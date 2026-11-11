#pragma once

/**
 * @file RuleEnginePipeline.hpp
 * @brief Settore 2 — Master Generative Rule Pipeline: Family 3 (2D Causal Markov Chain
 *        with Instantaneous Transient Markov Reset), Family 4 (Local Source-Filter
 *        Cross-Synthesis via Circular Permutation Matrix), Softmax + Spectral Sparsity
 *        Gating, Sub-150Hz Butterworth Band Lock, and C^1 ERB -> FFT Upsampling with
 *        2D Spectral Feathering.
 *
 * @details
 * Implements the complete real-time Settore 2 pipeline from Sections 1.2, 2.1, and 2.2
 * of the SpecTile Executive Architecture Plan:
 *  - Master Logit Composition on B = 64 psychoacoustic ERB bands:
 *      L_i(t, b) = ln(w_i + eps) + alpha_Geo * bar{Phi}^Geo_i(t, b)
 *                                + alpha_Desc * bar{Phi}^Desc_i(t, b)
 *                                + alpha_Mark * Phi^Mark_i(t, b)
 *                                + Lambda_lock,i(b)
 *  - Family 3 (Causal 2D Markov Continuity with Transient Markov Reset):
 *      Phi^Mark_i(t, b) = (1 - eta_reset * Omega_onset(t, b))
 *                       * [ kappa_t * tilde{M}_i(t-1, b)
 *                         + (kappa_f / 2) * (tilde{M}_i(t-1, b-1) + tilde{M}_i(t-1, b+1)) ]
 *                       + kappa_f * tilde{M}^causal_i(t, b-1)
 *  - Family 4 (Local Cross-Synthesis):
 *      Decouples envelope masks M^env_i(t, k) from fine-structure masks M^fine_j(t, k)
 *      via circular permutation matrix Pi_Delta controlled by chi_cross in [0, 1].
 *  - Sub-150Hz Butterworth Frequency Band Lock (4th-order Butterworth profile) +
 *    Quadrant 1 custom frequency band locks.
 *  - Temperature-scaled Softmax (tau in [0.03, 1.50]) + Spectral Sparsity Gating (eta_sparse).
 *  - C^1 Raised-Cosine ERB-to-FFT Upsampling (B -> K) + 2D Spectral Feathering
 *    strictly preserving the partition of unity sum_i M_i(t, k) = 1.
 *
 * Zero heap allocations, 64-byte AVX2/AVX-512/NEON aligned SoA buffers, C++20 noexcept.
 */

#include "SpectralGenome.hpp"
#include "GeometricTiler.hpp"
#include "FeatureCompetitor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace spectile::rules {

// ============================================================================
// Output Mask Matrix SoA Structure (Settore 2 -> Settore 3 & Settore 4)
// ============================================================================

/**
 * @brief Complete 64-byte aligned Structure-of-Arrays (SoA) Mask Output for a single
 *        STFT hop t, ready for Settore 3's LockFreeTripleBuffer<MaskMatrix2D>,
 *        Causal MS-PGHI Phase Glue, and Settore 4's GPU OKLCH Spectrogram Shader.
 */
struct alignas(64) MaskMatrix2D {
    /// Full-resolution FFT bin masks for Formant True Envelope E_i(t, k): sum_i M^env_i(t, k) == 1
    alignas(64) std::array<std::array<float, kPaddedFftBins>, kMaxSamples> envMasksFft{};
    /// Full-resolution FFT bin masks for Fine Structure F_j(t, k): sum_j M^fine_j(t, k) == 1
    alignas(64) std::array<std::array<float, kPaddedFftBins>, kMaxSamples> fineMasksFft{};

    /// ERB band-level envelope masks hat{M}^env_i(t, b) in [0, 1]
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> envMasksErb{};
    /// ERB band-level fine-structure masks hat{M}^fine_j(t, b) in [0, 1]
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> fineMasksErb{};

    /// Per-FFT-bin tile purity c_purity(t, k) = max_i M^env_i(t, k) in [1/N, 1.0] (for MS-PGHI Phase Glue)
    alignas(64) std::array<float, kPaddedFftBins> binPurity{};
    /// Per-FFT-bin dominant sample slot index i* = argmax_i M^env_i(t, k) (for Identity Phase-Locking & GPU)
    alignas(64) std::array<uint8_t, kPaddedFftBins> dominantSampleFft{};
    /// Per-ERB-band dominant sample slot index (for GPU Voronoi neon border shader)
    alignas(64) std::array<uint8_t, kMaxBands> dominantSampleErb{};
    /// Per-ERB-band onset activation gate Omega_onset(t, b) in [0, 1]
    alignas(64) std::array<float, kMaxBands> onsetGateErb{};

    std::size_t numActiveSamples = kMaxSamples;
    std::size_t numBands         = kDefaultBands;
    std::size_t numFftBins       = kDefaultFftBins;
};

// ============================================================================
// Frequency Band Lock Configuration (Quadrant 1 UI & Sub-150Hz Butterworth Lock)
// ============================================================================

inline constexpr std::size_t kMaxBandLocks = 4;

/**
 * @brief Configurable Frequency Band Lock region (Quadrant 1 of Deep Spectral Matrix).
 */
struct FrequencyBandLockRegion {
    bool        enabled      = false;
    float       lowFreqHz    = 20.0f;
    float       highFreqHz   = 150.0f;
    std::size_t ownerSlot    = 0;     ///< Sample slot index locked in [lowFreqHz, highFreqHz]
    float       lockStrength = 8.0f;  ///< Logit boost Lambda_lock in [0.0, 16.0]
};

// ============================================================================
// Psychoacoustic ERB Scale & Pre-Tabulated C^1 Upsampling Table (Section 1.2)
// ============================================================================

/**
 * @brief Pre-tabulated Glasberg & Moore (1990) ERB psychoacoustic mapping between
 *        B ERB bands and K FFT bins with C^1 raised-cosine interpolation weights.
 */
struct alignas(64) ErbProjectionTable {
    std::array<float,       kMaxBands>      bandCenterFreqHz{}; ///< Center frequency f_b of each ERB band
    std::array<uint16_t,    kPaddedFftBins> b0{};               ///< Lower adjacent ERB band index floor(xi(k))
    std::array<uint16_t,    kPaddedFftBins> b1{};               ///< Upper adjacent ERB band index min(b0 + 1, B - 1)
    std::array<float,       kPaddedFftBins> wUp{};              ///< C^1 raised-cosine weight 0.5*(1 - cos(pi*alpha(k)))
    std::array<float,       kPaddedFftBins> binFreqHz{};        ///< Frequency f_k = k * f_s / M of each FFT bin

    std::size_t numBands   = kDefaultBands;
    std::size_t numFftBins = kDefaultFftBins;
    float       sampleRate = 48000.0f;
    std::size_t fftSize    = 4096;

    /**
     * @brief Glasberg & Moore (1990) frequency (Hz) to ERB-rate (Cams):
     *        E_ERB(f) = 21.4 * log10(0.00437 * f + 1)
     */
    [[nodiscard]] static inline float hzToErbRate(float freqHz) noexcept {
        return 21.4f * std::log10(0.00437f * std::max(freqHz, 0.0f) + 1.0f);
    }

    /**
     * @brief Inverse ERB-rate (Cams) to frequency (Hz):
     *        f = (10^(E_ERB / 21.4) - 1) / 0.00437
     */
    [[nodiscard]] static inline float erbRateToHz(float erbCam) noexcept {
        return (std::pow(10.0f, erbCam / 21.4f) - 1.0f) / 0.00437f;
    }

    /**
     * @brief Equivalent Rectangular Bandwidth in Hz at frequency f:
     *        ERB(f) = 24.7 * (4.37 * f / 1000 + 1)
     */
    [[nodiscard]] static inline float erbBandwidthHz(float freqHz) noexcept {
        return 24.7f * (0.00437f * std::max(freqHz, 0.0f) + 1.0f);
    }

    /**
     * @brief Initializes the pre-tabulated ERB-to-FFT projection weights (Section 1.2).
     *        Called only on initialization or when FFT size / sample rate changes.
     */
    void configure(float fs, std::size_t fftLength, std::size_t bands = kDefaultBands) noexcept {
        sampleRate = std::max(fs, 8000.0f);
        fftSize    = std::clamp<std::size_t>(fftLength, 256, 4096);
        numFftBins = std::min<std::size_t>(fftSize / 2 + 1, kDefaultFftBins);
        numBands   = std::clamp<std::size_t>(bands, 8, kMaxBands);

        const float nyquistHz = 0.5f * sampleRate;
        const float erbMin    = hzToErbRate(20.0f);
        const float erbMax    = hzToErbRate(nyquistHz);
        const float erbSpan   = std::max(erbMax - erbMin, 1.0e-3f);

        // Compute center frequencies f_b of each of the B ERB bands
        for (std::size_t b = 0; b < numBands; ++b) {
            const float normB  = static_cast<float>(b) / static_cast<float>(numBands - 1);
            const float erbCam = erbMin + normB * erbSpan;
            bandCenterFreqHz[b]= erbRateToHz(erbCam);
        }
        for (std::size_t b = numBands; b < kMaxBands; ++b) {
            bandCenterFreqHz[b] = nyquistHz;
        }

        // Pre-tabulate continuous band coordinate xi(k), b0(k), b1(k), and C^1 weight w_up(k)
        const float binStepHz = sampleRate / static_cast<float>(fftSize);
        const float maxBandF  = static_cast<float>(numBands - 1);

        for (std::size_t k = 0; k < kPaddedFftBins; ++k) {
            const float fk = std::min(static_cast<float>(k) * binStepHz, nyquistHz);
            binFreqHz[k]   = fk;

            const float erbK = hzToErbRate(fk);
            const float xi   = std::clamp(((erbK - erbMin) / erbSpan) * maxBandF, 0.0f, maxBandF);
            const auto  idx0 = static_cast<uint16_t>(std::floor(xi));
            const auto  idx1 = static_cast<uint16_t>(std::min<std::size_t>(idx0 + 1U, numBands - 1));
            const float alpha= xi - static_cast<float>(idx0);

            b0[k]  = idx0;
            b1[k]  = idx1;
            // Exact C^1 raised-cosine interpolation weight (Section 1.2):
            //   w_up(k) = 0.5 * (1 - cos(pi * alpha(k)))
            wUp[k] = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * alpha));
        }
    }
};

// ============================================================================
// RuleEnginePipeline — Master Settore 2 Real-Time Engine
// ============================================================================

class RuleEnginePipeline {
public:
    RuleEnginePipeline() noexcept {
        erbTable_.configure(48000.0f, 4096, kDefaultBands);
        resetState();
    }

    /**
     * @brief Configures sample rate, FFT size M, and number of psychoacoustic ERB bands B.
     */
    void prepare(float sampleRate, std::size_t fftSize = 4096, std::size_t numBands = kDefaultBands) noexcept {
        erbTable_.configure(sampleRate, fftSize, numBands);
        resetState();
    }

    /**
     * @brief Resets all temporal Markov and 2D Spectral Feathering memory states.
     */
    void resetState() noexcept {
        hasPreviousFrame_ = false;
        const float uniformWeight = 1.0f / static_cast<float>(kMaxSamples);
        for (std::size_t i = 0; i < kMaxSamples; ++i) {
            prevEnvMasksErb_[i].fill(uniformWeight);
            prevEnvMasksFft_[i].fill(uniformWeight);
            prevFineMasksFft_[i].fill(uniformWeight);
        }
    }

    /**
     * @brief Sets the prior presence weights w_1 .. w_N for the N sample slots.
     */
    void setSlotPresenceWeights(const std::array<float, kMaxSamples>& weights) noexcept {
        slotWeights_ = weights;
    }

    /**
     * @brief Sets custom Frequency Band Lock dividers (Quadrant 1 of Deep Spectral Matrix).
     */
    void setCustomBandLocks(const std::array<FrequencyBandLockRegion, kMaxBandLocks>& locks) noexcept {
        customBandLocks_ = locks;
    }

    /**
     * @brief Accesses the underlying Family 1 (GeometricTiler) and Family 2 (FeatureCompetitor) engines.
     */
    [[nodiscard]] GeometricTiler&       geometricTiler() noexcept       { return geometricTiler_; }
    [[nodiscard]] const GeometricTiler& geometricTiler() const noexcept { return geometricTiler_; }

    [[nodiscard]] FeatureCompetitor&       featureCompetitor() noexcept       { return featureCompetitor_; }
    [[nodiscard]] const FeatureCompetitor& featureCompetitor() const noexcept { return featureCompetitor_; }

    [[nodiscard]] const ErbProjectionTable& erbProjectionTable() const noexcept { return erbTable_; }

    /**
     * @brief Executes the complete Settore 2 Rule Engine Pipeline for a single STFT hop t:
     *        1. Evaluates Family 1 (Phi^Geo_i: O(B) Parabolic Voronoi + Simplex 2D + von Mises Comb)
     *        2. Evaluates Family 2 (Phi^Desc_i: Onset/Continuant Split + Flatness + Anti-Masking Hole Filler)
     *        3. Evaluates Sub-150Hz Butterworth Frequency Band Lock + Quadrant 1 custom Band Locks
     *        4. Sweeps causally across b = 0..B-1 to evaluate Family 3 (2D Causal Markov Chain with
     *           Instantaneous Transient Markov Reset), Family 4 (Circular Permutation Cross-Synthesis),
     *           Temperature-Scaled Softmax, and Spectral Sparsity Gating
     *        5. Projects B -> K via C^1 Raised-Cosine Upsampling and applies separable 2D Spectral
     *           Feathering with exact Partition of Unity conservation (sum_i M_i = 1).
     *
     * @param params           Decoded genome & macro parameters
     * @param descriptors      Current frame ERB band descriptors from Settore 1
     * @param timeSeconds      Current continuous playback/STFT time in seconds
     * @param fundamentalF0Hz  Fundamental pitch f_0(t) in Hz (for harmonic comb alignment)
     * @param numActiveSamples Number of active sample slots N in [2, 8]
     * @param outMasks         Destination aligned SoA MaskMatrix2D structure
     */
    void processHop(
        const DecodedGenomeParams& params,
        const FrameErbDescriptors& descriptors,
        float timeSeconds,
        float fundamentalF0Hz,
        std::size_t numActiveSamples,
        MaskMatrix2D& outMasks) noexcept
    {
        const std::size_t N = std::clamp<std::size_t>(numActiveSamples, 2, kMaxSamples);
        const std::size_t B = erbTable_.numBands;
        const std::size_t K = erbTable_.numFftBins;

        outMasks.numActiveSamples = N;
        outMasks.numBands         = B;
        outMasks.numFftBins       = K;

        const float uniformPrior = 1.0f / static_cast<float>(N);

        // ====================================================================
        // Step 1: Evaluate Family 1 (Geometric Tiling: Voronoi O(B) + Simplex + Comb)
        // ====================================================================
        geometricTiler_.evaluateFrame(
            params,
            timeSeconds,
            N,
            B,
            erbTable_.bandCenterFreqHz,
            fundamentalF0Hz,
            geoScratch_);

        // ====================================================================
        // Step 2: Evaluate Family 2 (Descriptor Competition & Anti-Masking Hole Filler)
        // ====================================================================
        featureCompetitor_.evaluateFrame(
            params,
            descriptors,
            N,
            B,
            featScratch_);

        // Pre-compute log prior weights ln(w_i + eps)
        std::array<float, kMaxSamples> logPriorWeights{};
        for (std::size_t i = 0; i < N; ++i) {
            logPriorWeights[i] = std::log(std::max(slotWeights_[i], 1.0e-4f));
        }

        // Log-odds Markov persistence gains kappa_t(P_stay,t) and kappa_f(P_stay,f)
        const float kappaT = params.markovKappaTime;
        const float kappaF = params.markovKappaFreq;

        // Causal vertical Markov state tilde{M}^causal_i(t, b-1) during the upward scan b = 0..B-1
        std::array<float, kMaxSamples> causalFreqState{};
        causalFreqState.fill(0.0f);

        // ====================================================================
        // Step 3: Upward Causal Sweep across ERB Bands b = 0 .. B-1
        //         (Family 3 Markov + Transient Reset + Band Lock + Family 4 Cross-Synthesis
        //          + Softmax(tau) + Spectral Sparsity Gating)
        // ====================================================================
        for (std::size_t b = 0; b < B; ++b) {
            const float freqHz     = erbTable_.bandCenterFreqHz[b];
            const float omegaOnset = featScratch_.onsetActivationGate[b];
            outMasks.onsetGateErb[b] = omegaOnset;

            // --- (A) Transient Markov Reset Factor (Section 2.1, Family 3) ---
            // On transient attacks (Omega_onset -> 1), (1 - eta_reset * Omega_onset) -> 0
            // instantaneously zeroes the temporal Markov memory to guarantee razor-sharp attacks!
            const float transientMemoryKeep = hasPreviousFrame_
                ? std::clamp(1.0f - params.transientResetEta * omegaOnset, 0.0f, 1.0f)
                : 0.0f;

            const std::size_t bPrev = (b > 0)     ? (b - 1) : 0;
            const std::size_t bNext = (b + 1 < B) ? (b + 1) : (B - 1);

            // --- (B) Sub-150Hz 4th-Order Butterworth Frequency Band Lock ---
            //   H_sub(f_b) = 1 / sqrt(1 + (f_b / f_sub)^8)
            const float fRatio   = freqHz / std::max(params.subBassCutoffHz, 40.0f);
            const float fRatio2  = fRatio * fRatio;
            const float fRatio4  = fRatio2 * fRatio2;
            const float fRatio8  = fRatio4 * fRatio4;
            const float hButter  = 1.0f / std::sqrt(1.0f + fRatio8);
            const std::size_t subOwner = params.subBassOwner % N;

            // Evaluate Master Envelope Logits L^env_i(t, b) for each active sample i
            std::array<float, kMaxSamples> logitsEnv{};

            for (std::size_t i = 0; i < N; ++i) {
                // Zero-centered previous frame masks tilde{M}_i(t-1, b) = M_i(t-1, b) - 1/N
                const float prevCenter = prevEnvMasksErb_[i][b]     - uniformPrior;
                const float prevLower  = prevEnvMasksErb_[i][bPrev] - uniformPrior;
                const float prevUpper  = prevEnvMasksErb_[i][bNext] - uniformPrior;

                // Exact Family 3 Equation (Section 2.1):
                //   Phi^Mark_i(t, b) = (1 - eta_reset * Omega_onset(t, b))
                //                    * [ kappa_t * tilde{M}_i(t-1, b)
                //                      + (kappa_f / 2) * (tilde{M}_i(t-1, b-1) + tilde{M}_i(t-1, b+1)) ]
                //                    + kappa_f * tilde{M}^causal_i(t, b-1)
                const float temporalMarkov = transientMemoryKeep
                    * (kappaT * prevCenter + 0.5f * kappaF * (prevLower + prevUpper));
                const float causalFreqMarkov = (b > 0) ? (kappaF * causalFreqState[i]) : 0.0f;
                const float phiMarkov = temporalMarkov + causalFreqMarkov;

                // Sub-150Hz Butterworth Band Lock + Custom Quadrant 1 Band Locks
                float lambdaLock = (i == subOwner)
                    ? (+params.subBassLockLambda * hButter)
                    : (-params.subBassLockLambda * hButter);

                for (const FrequencyBandLockRegion& lock : customBandLocks_) {
                    if (lock.enabled && freqHz >= lock.lowFreqHz && freqHz <= lock.highFreqHz) {
                        const std::size_t owner = lock.ownerSlot % N;
                        lambdaLock += (i == owner) ? (+lock.lockStrength) : (-lock.lockStrength);
                    }
                }

                // Master Logit Composition Equation (Section 2.1):
                logitsEnv[i] = logPriorWeights[i]
                             + params.alphaGeo  * geoScratch_.normalizedLogits[i][b]
                             + params.alphaDesc * featScratch_.normalizedLogits[i][b]
                             + params.alphaMark * phiMarkov
                             + lambdaLock;
            }

            // --- (C) Family 4: Local Cross-Synthesis via Circular Permutation Matrix Pi_Delta ---
            // Attenuate cross-synthesis inside the locked sub-bass region (1 - hButter)
            // so the sub-bass fundamental (< 150 Hz) maintains coherent envelope and fine structure.
            const float chiEffective = params.crossSynthChi * (1.0f - 0.85f * hButter);
            const std::size_t deltaShift = std::clamp<std::size_t>(params.crossShiftDelta, 1, N - 1);

            std::array<float, kMaxSamples> logitsFine{};
            for (std::size_t j = 0; j < N; ++j) {
                const std::size_t permutedIdx = (j + N - (deltaShift % N)) % N;
                logitsFine[j] = (1.0f - chiEffective) * logitsEnv[j]
                              + chiEffective          * logitsEnv[permutedIdx];
            }

            // --- (D) Temperature-Scaled Softmax & Spectral Sparsity Gating ---
            std::array<float, kMaxSamples> bandEnvMask{};
            std::array<float, kMaxSamples> bandFineMask{};

            applySoftmaxAndSparsityGate(logitsEnv,  N, params.softmaxTempTau, params.sparsityGateEta, bandEnvMask);
            applySoftmaxAndSparsityGate(logitsFine, N, params.softmaxTempTau, params.sparsityGateEta, bandFineMask);

            // Store ERB masks, update causal frequency Markov state tilde{M}^causal_i(t, b),
            // and find dominant sample index for this ERB band
            float   maxWeight = -1.0f;
            uint8_t domIdx    = 0;
            for (std::size_t i = 0; i < N; ++i) {
                outMasks.envMasksErb[i][b]  = bandEnvMask[i];
                outMasks.fineMasksErb[i][b] = bandFineMask[i];
                prevEnvMasksErb_[i][b]      = bandEnvMask[i];
                causalFreqState[i]          = bandEnvMask[i] - uniformPrior;

                if (bandEnvMask[i] > maxWeight) {
                    maxWeight = bandEnvMask[i];
                    domIdx    = static_cast<uint8_t>(i);
                }
            }
            for (std::size_t i = N; i < kMaxSamples; ++i) {
                outMasks.envMasksErb[i][b]  = 0.0f;
                outMasks.fineMasksErb[i][b] = 0.0f;
                prevEnvMasksErb_[i][b]      = 0.0f;
            }
            outMasks.dominantSampleErb[b] = domIdx;
        }

        // ====================================================================
        // Step 4: Fluid C^1 Raised-Cosine Upsampling (B -> K) + 2D Spectral Feathering
        //         (Section 1.2 & Section 3.3)
        // ====================================================================
        upsampleAndFeather2D(params, N, K, outMasks);
        hasPreviousFrame_ = true;
    }

private:
    ErbProjectionTable                              erbTable_{};
    GeometricTiler                                  geometricTiler_{};
    FeatureCompetitor                               featureCompetitor_{};
    GeometricTilingOutput                           geoScratch_{};
    FeatureCompetitionOutput                        featScratch_{};

    std::array<float, kMaxSamples>                  slotWeights_{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::array<FrequencyBandLockRegion, kMaxBandLocks> customBandLocks_{};

    /// Previous frame ERB mask state M^env_i(t-1, b) for Family 3 Causal 2D Markov Chain
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples>      prevEnvMasksErb_{};
    /// Previous frame feathered FFT masks for temporal 2D Spectral Feathering
    alignas(64) std::array<std::array<float, kPaddedFftBins>, kMaxSamples> prevEnvMasksFft_{};
    alignas(64) std::array<std::array<float, kPaddedFftBins>, kMaxSamples> prevFineMasksFft_{};
    bool hasPreviousFrame_ = false;

    /**
     * @brief Evaluates Temperature-Scaled Softmax (tau in [0.03, 1.50]) followed by
     *        Spectral Sparsity Gating (eta_sparse) with exact partition-of-unity renormalization.
     */
    static inline void applySoftmaxAndSparsityGate(
        const std::array<float, kMaxSamples>& logits,
        std::size_t N,
        float temperatureTau,
        float sparsityGateEta,
        std::array<float, kMaxSamples>& outProbs) noexcept
    {
        const float invTau = 1.0f / std::clamp(temperatureTau, 0.03f, 1.50f);

        // Find max logit for numerical stability
        float maxLogit = logits[0];
        for (std::size_t i = 1; i < N; ++i) {
            maxLogit = std::max(maxLogit, logits[i]);
        }

        // Compute Softmax probabilities P_i
        float sumExp  = 0.0f;
        float maxProb = 0.0f;
        for (std::size_t i = 0; i < N; ++i) {
            const float scaled = std::clamp((logits[i] - maxLogit) * invTau, -40.0f, 0.0f);
            const float p      = std::exp(scaled);
            outProbs[i]        = p;
            sumExp            += p;
        }

        const float invSumExp = 1.0f / std::max(sumExp, 1.0e-12f);
        for (std::size_t i = 0; i < N; ++i) {
            outProbs[i] *= invSumExp;
            maxProb      = std::max(maxProb, outProbs[i]);
        }

        // Spectral Sparsity Gating (Section 2.1):
        // Suppresses marginal sample contributions below eta_sparse * max_m P_m(t, b)
        // to prevent the "Spectral Mud Paradox", then renormalizes to sum_i hat{M}_i = 1.
        const float gateThreshold = std::clamp(sparsityGateEta, 0.0f, 0.90f) * maxProb;
        if (gateThreshold > 1.0e-5f) {
            float gatedSum = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                const float gated = std::max(0.0f, outProbs[i] - gateThreshold);
                outProbs[i] = gated;
                gatedSum   += gated;
            }
            if (gatedSum > 1.0e-8f) {
                const float invGatedSum = 1.0f / gatedSum;
                for (std::size_t i = 0; i < N; ++i) {
                    outProbs[i] *= invGatedSum;
                }
            }
        }
    }

    /**
     * @brief Performs C^1 Raised-Cosine Upsampling from B ERB bands to K FFT bins
     *        (Section 1.2) followed by Separable 2D Spectral Feathering (temporal causal IIR
     *        + symmetric 1D Gaussian frequency kernel) while strictly preserving
     *        sum_{i=1}^N M^env_i(t, k) == 1 and sum_{j=1}^N M^fine_j(t, k) == 1.
     */
    void upsampleAndFeather2D(
        const DecodedGenomeParams& params,
        std::size_t N,
        std::size_t K,
        MaskMatrix2D& outMasks) noexcept
    {
        // Temporary unfeathered FFT bin buffers (aligned on stack)
        alignas(64) std::array<std::array<float, kPaddedFftBins>, kMaxSamples> rawEnvFft{};
        alignas(64) std::array<std::array<float, kPaddedFftBins>, kMaxSamples> rawFineFft{};

        // 1. C^1 Raised-Cosine Upsampling B -> K (2 multiplies + 1 add per bin, Section 1.2)
        //    Since sum_i hat{M}_i(t, b0) = 1 and sum_i hat{M}_i(t, b1) = 1, the convex
        //    combination (1 - w_up)*hat{M}_i(b0) + w_up*hat{M}_i(b1) identically sums to 1!
        const float meanOnset = computeMeanOnsetGate(outMasks.onsetGateErb, outMasks.numBands);
        // Reduce temporal feathering lag during sharp transient attacks
        const float alphaTime = hasPreviousFrame_
            ? (params.featherTimeAlpha * (1.0f - 0.80f * params.transientResetEta * meanOnset))
            : 0.0f;
        const float oneMinusAlpha = 1.0f - alphaTime;

        for (std::size_t i = 0; i < N; ++i) {
            const auto& erbEnv  = outMasks.envMasksErb[i];
            const auto& erbFine = outMasks.fineMasksErb[i];
            auto&       prevEnv = prevEnvMasksFft_[i];
            auto&       prevFine= prevFineMasksFft_[i];

            for (std::size_t k = 0; k < K; ++k) {
                const uint16_t idx0 = erbTable_.b0[k];
                const uint16_t idx1 = erbTable_.b1[k];
                const float    w1   = erbTable_.wUp[k];
                const float    w0   = 1.0f - w1;

                const float upEnv  = w0 * erbEnv[idx0]  + w1 * erbEnv[idx1];
                const float upFine = w0 * erbFine[idx0] + w1 * erbFine[idx1];

                // Causal temporal feathering (1st dimension of 2D Spectral Feathering)
                const float tempEnv  = oneMinusAlpha * upEnv  + alphaTime * prevEnv[k];
                const float tempFine = oneMinusAlpha * upFine + alphaTime * prevFine[k];

                rawEnvFft[i][k]  = tempEnv;
                rawFineFft[i][k] = tempFine;
                prevEnv[k]       = tempEnv;
                prevFine[k]      = tempFine;
            }
            for (std::size_t k = K; k < kPaddedFftBins; ++k) {
                rawEnvFft[i][k]  = 0.0f;
                rawFineFft[i][k] = 0.0f;
            }
        }

        // 2. Separable 1D Gaussian Frequency Feathering along FFT bins k = 0 .. K-1
        //    (2nd dimension of 2D Spectral Feathering, radius R_k <= 6 bins)
        const float sigmaBins = std::clamp(params.featherFreqSigma, 0.0f, 12.0f);
        if (sigmaBins < 0.35f) {
            // Fast path: no frequency blur needed; copy directly and compute purity/dominant ID
            for (std::size_t i = 0; i < N; ++i) {
                outMasks.envMasksFft[i]  = rawEnvFft[i];
                outMasks.fineMasksFft[i] = rawFineFft[i];
            }
        } else {
            // Build normalized symmetric 1D Gaussian kernel of half-width R_k in [1, 6]
            constexpr int kMaxRadius = 6;
            const int radius = std::clamp(static_cast<int>(std::ceil(2.0f * sigmaBins)), 1, kMaxRadius);
            std::array<float, 2 * kMaxRadius + 1> kernel{};
            const float invTwoSigmaSq = 1.0f / (2.0f * sigmaBins * sigmaBins);
            float kernelSum = 0.0f;
            for (int r = -radius; r <= radius; ++r) {
                const float w = std::exp(-static_cast<float>(r * r) * invTwoSigmaSq);
                kernel[static_cast<std::size_t>(r + radius)] = w;
                kernelSum += w;
            }
            const float invKernelSum = 1.0f / kernelSum;
            for (int r = -radius; r <= radius; ++r) {
                kernel[static_cast<std::size_t>(r + radius)] *= invKernelSum;
            }

            // Convolve each sample's mask with clamped boundary reflection (preserving sum_i M_i = 1)
            const int kMaxInt = static_cast<int>(K) - 1;
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t k = 0; k < K; ++k) {
                    const int kInt = static_cast<int>(k);
                    float accEnv  = 0.0f;
                    float accFine = 0.0f;
                    for (int r = -radius; r <= radius; ++r) {
                        const int   kNbr = std::clamp(kInt + r, 0, kMaxInt);
                        const float kw   = kernel[static_cast<std::size_t>(r + radius)];
                        accEnv  += kw * rawEnvFft[i][static_cast<std::size_t>(kNbr)];
                        accFine += kw * rawFineFft[i][static_cast<std::size_t>(kNbr)];
                    }
                    outMasks.envMasksFft[i][k]  = accEnv;
                    outMasks.fineMasksFft[i][k] = accFine;
                }
                for (std::size_t k = K; k < kPaddedFftBins; ++k) {
                    outMasks.envMasksFft[i][k]  = 0.0f;
                    outMasks.fineMasksFft[i][k] = 0.0f;
                }
            }
        }

        // Zero unused sample slots N .. kMaxSamples-1
        for (std::size_t i = N; i < kMaxSamples; ++i) {
            outMasks.envMasksFft[i].fill(0.0f);
            outMasks.fineMasksFft[i].fill(0.0f);
        }

        // 3. Exact Partition-of-Unity Renormalization + Tile Purity c_purity(t, k) & Dominant ID i*(t, k)
        for (std::size_t k = 0; k < K; ++k) {
            float sumEnv  = 0.0f;
            float sumFine = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                sumEnv  += outMasks.envMasksFft[i][k];
                sumFine += outMasks.fineMasksFft[i][k];
            }
            const float invSumEnv  = 1.0f / std::max(sumEnv,  1.0e-12f);
            const float invSumFine = 1.0f / std::max(sumFine, 1.0e-12f);

            float   maxM   = -1.0f;
            uint8_t domIdx = 0;
            for (std::size_t i = 0; i < N; ++i) {
                const float mEnv  = outMasks.envMasksFft[i][k]  * invSumEnv;
                const float mFine = outMasks.fineMasksFft[i][k] * invSumFine;
                outMasks.envMasksFft[i][k]  = mEnv;
                outMasks.fineMasksFft[i][k] = mFine;
                if (mEnv > maxM) {
                    maxM   = mEnv;
                    domIdx = static_cast<uint8_t>(i);
                }
            }
            outMasks.binPurity[k]         = std::clamp(maxM, 0.0f, 1.0f);
            outMasks.dominantSampleFft[k] = domIdx;
        }

        for (std::size_t k = K; k < kPaddedFftBins; ++k) {
            outMasks.binPurity[k]         = 1.0f;
            outMasks.dominantSampleFft[k] = 0;
        }
    }

    [[nodiscard]] static constexpr float computeMeanOnsetGate(
        const std::array<float, kMaxBands>& onsetGate,
        std::size_t B) noexcept
    {
        float sum = 0.0f;
        for (std::size_t b = 0; b < B; ++b) {
            sum += onsetGate[b];
        }
        return sum / static_cast<float>( std::max<std::size_t>(B, 1) );
    }
};

} // namespace spectile::rules
