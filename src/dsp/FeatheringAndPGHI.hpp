#pragma once

#include "../core/SpectralTypesSoA.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

namespace spectile::dsp {

// ============================================================================
// FlatMaxHeap<MaxCapacity> — Static Pre-Allocated Binary Max-Heap for MS-PGHI
// ============================================================================
struct PGHIHeapNode {
    float         magnitude{0.0f};
    std::uint16_t binIndex{0};
    std::uint8_t  isCurrentFrame{1}; // 0 = seed at (t-1, k), 1 = active bin at (t, k)
    std::uint8_t  reserved{0};
};

/**
 * @brief Zero-allocation static binary Max-Heap ordered by spectral magnitude.
 *        Used by Real-Time Causal MS-PGHI inside processBlock() (O(K_act log K_act)).
 */
template <std::size_t MaxCapacity = 8192>
class alignas(kCacheLineBytes) FlatMaxHeap {
public:
    constexpr FlatMaxHeap() noexcept = default;

    void clear() noexcept {
        size_ = 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return MaxCapacity;
    }

    bool push(float magnitude, std::uint16_t binIndex, std::uint8_t isCurrentFrame = 1) noexcept {
        if (size_ >= MaxCapacity) {
            return false;
        }
        std::size_t idx = size_++;
        nodes_[idx]     = PGHIHeapNode{magnitude, binIndex, isCurrentFrame, 0};

        // Sift up
        while (idx > 0) {
            const std::size_t parent = (idx - 1) >> 1u;
            if (nodes_[parent].magnitude >= nodes_[idx].magnitude) {
                break;
            }
            std::swap(nodes_[parent], nodes_[idx]);
            idx = parent;
        }
        return true;
    }

    [[nodiscard]] PGHIHeapNode pop() noexcept {
        const PGHIHeapNode topNode = nodes_[0];
        --size_;
        if (size_ > 0) {
            nodes_[0] = nodes_[size_];
            siftDown(0);
        }
        return topNode;
    }

private:
    void siftDown(std::size_t idx) noexcept {
        const std::size_t half = size_ >> 1u;
        while (idx < half) {
            std::size_t left    = (idx << 1u) + 1u;
            std::size_t right   = left + 1u;
            std::size_t largest = left;

            if (right < size_ && nodes_[right].magnitude > nodes_[left].magnitude) {
                largest = right;
            }
            if (nodes_[idx].magnitude >= nodes_[largest].magnitude) {
                break;
            }
            std::swap(nodes_[idx], nodes_[largest]);
            idx = largest;
        }
    }

    alignas(kCacheLineBytes) std::array<PGHIHeapNode, MaxCapacity> nodes_{};
    std::size_t size_{0};
};

// ============================================================================
// SpectralFeathering2D — Separable 2D Gaussian Freq + Causal IIR / Lookahead FIR Time
// ============================================================================
/**
 * @brief Separable 2D Spectral Feathering filter (Section 1.1 Stage 7 & Section 3.3.1).
 *        Convolves the mask matrix M_i(t, k) along frequency with a symmetric normalized
 *        Gaussian kernel and along time with either a causal IIR smoother or a short
 *        FIR lookahead kernel, strictly preserving \sum_{i=1}^N M_i(t, k) = 1.
 */
class alignas(kCacheLineBytes) SpectralFeathering2D {
public:
    static constexpr std::size_t kMaxFreqKernelRadius = 16;
    static constexpr std::size_t kMaxFreqKernelSize   = 2 * kMaxFreqKernelRadius + 1;
    static constexpr std::size_t kMaxLookaheadFrames  = 5;

    SpectralFeathering2D() {
        prepare(kDefaultNumBins);
    }

    explicit SpectralFeathering2D(std::size_t numBins) {
        prepare(numBins);
    }

    void prepare(std::size_t numBins) {
        numBins_    = std::max<std::size_t>(2, numBins);
        paddedBins_ = roundUpToSimdLanes(numBins_);

        tempFreqScratch_.assign(paddedBins_, 0.0f);
        prevEnvState_.assign(kMaxSampleSlots * paddedBins_, 0.0f);
        prevFineState_.assign(kMaxSampleSlots * paddedBins_, 0.0f);

        for (std::size_t f = 0; f < kMaxLookaheadFrames; ++f) {
            firEnvHistory_[f].assign(kMaxSampleSlots * paddedBins_, 0.0f);
            firFineHistory_[f].assign(kMaxSampleSlots * paddedBins_, 0.0f);
        }

        resetState();
        setParameters(1.5f, 0.35f, false);
    }

    void resetState() noexcept {
        std::fill(prevEnvState_.begin(), prevEnvState_.end(), 0.0f);
        std::fill(prevFineState_.begin(), prevFineState_.end(), 0.0f);
        std::fill_n(prevEnvState_.data(), paddedBins_, 1.0f);
        std::fill_n(prevFineState_.data(), paddedBins_, 1.0f);

        for (std::size_t f = 0; f < kMaxLookaheadFrames; ++f) {
            std::fill(firEnvHistory_[f].begin(), firEnvHistory_[f].end(), 0.0f);
            std::fill(firFineHistory_[f].begin(), firFineHistory_[f].end(), 0.0f);
            std::fill_n(firEnvHistory_[f].data(), paddedBins_, 1.0f);
            std::fill_n(firFineHistory_[f].data(), paddedBins_, 1.0f);
        }
        firWriteCursor_ = 0;
    }

    /**
     * @brief Updates the 2D feathering parameters and precomputes the normalized Gaussian kernel.
     * @param sigmaFreqBins Standard deviation along FFT bins (0 = bypass frequency feathering).
     * @param timeSmoothCoeff Temporal smoothing coefficient in [0, 0.98].
     * @param useFirLookahead If true, uses a 5-tap Gaussian FIR filter across recent frames;
     *                        if false, uses a zero-latency causal 1-pole IIR filter.
     */
    void setParameters(float sigmaFreqBins, float timeSmoothCoeff, bool useFirLookahead = false) noexcept {
        sigmaFreqBins_   = std::clamp(sigmaFreqBins, 0.0f, 8.0f);
        timeSmoothCoeff_ = std::clamp(timeSmoothCoeff, 0.0f, 0.98f);
        useFirLookahead_ = useFirLookahead;

        // Precompute symmetric 1D Gaussian frequency kernel
        if (sigmaFreqBins_ < 0.15f) {
            freqRadius_    = 0;
            freqKernel_[0] = 1.0f;
        } else {
            freqRadius_ = std::min<std::size_t>(
                static_cast<std::size_t>(std::ceil(3.0f * sigmaFreqBins_)), kMaxFreqKernelRadius);
            const float invTwoSigmaSq = 1.0f / (2.0f * sigmaFreqBins_ * sigmaFreqBins_);
            float       sumW          = 0.0f;

            for (std::ptrdiff_t d = -static_cast<std::ptrdiff_t>(freqRadius_);
                 d <= static_cast<std::ptrdiff_t>(freqRadius_);
                 ++d) {
                const float w = std::exp(-static_cast<float>(d * d) * invTwoSigmaSq);
                freqKernel_[static_cast<std::size_t>(d + static_cast<std::ptrdiff_t>(freqRadius_))] = w;
                sumW += w;
            }
            const float invSum = 1.0f / std::max(sumW, 1.0e-12f);
            for (std::size_t i = 0; i < 2 * freqRadius_ + 1; ++i) {
                freqKernel_[i] *= invSum;
            }
        }

        // Precompute temporal FIR weights
        const float sigmaT = std::max(0.25f, timeSmoothCoeff_ * 2.0f);
        float       sumFir = 0.0f;
        for (std::size_t tap = 0; tap < kMaxLookaheadFrames; ++tap) {
            const float dist = static_cast<float>(tap) - 2.0f;
            const float w    = std::exp(-(dist * dist) / (2.0f * sigmaT * sigmaT));
            timeFirWeights_[tap] = w;
            sumFir += w;
        }
        const float invSumFir = 1.0f / std::max(sumFir, 1.0e-12f);
        for (float& w : timeFirWeights_) {
            w *= invSumFir;
        }
    }

    /**
     * @brief Applies separable 2D Spectral Feathering to both M^{env} and M^{fine} in MaskMatrix2D
     *        while guaranteeing exact conservation of \sum_{i=1}^N M_i(t, k) = 1.
     */
    void process(MaskMatrix2D& masks, std::size_t activeSlots = kMaxSampleSlots) noexcept {
        const std::size_t slots = std::clamp<std::size_t>(activeSlots, 1, kMaxSampleSlots);
        const std::size_t K     = std::min(numBins_, masks.numBins);

        // Stage 1: Symmetric 1D Gaussian Convolution along Frequency k
        if (freqRadius_ > 0) {
            for (std::size_t s = 0; s < slots; ++s) {
                convolveFrequencyGaussian(masks.binEnvSlot(s), K);
                convolveFrequencyGaussian(masks.binFineSlot(s), K);
            }
        }

        // Stage 2: Temporal Smoothing (Causal IIR or Lookahead FIR)
        if (useFirLookahead_) {
            const std::size_t writeSlot = firWriteCursor_;
            firWriteCursor_             = (firWriteCursor_ + 1) % kMaxLookaheadFrames;

            for (std::size_t s = 0; s < slots; ++s) {
                auto envSpan  = masks.binEnvSlot(s);
                auto fineSpan = masks.binFineSlot(s);
                float* histEnvWrite  = firEnvHistory_[writeSlot].data() + s * paddedBins_;
                float* histFineWrite = firFineHistory_[writeSlot].data() + s * paddedBins_;

                for (std::size_t k = 0; k < K; ++k) {
                    histEnvWrite[k]  = envSpan[k];
                    histFineWrite[k] = fineSpan[k];
                }

                for (std::size_t k = 0; k < K; ++k) {
                    float accEnv  = 0.0f;
                    float accFine = 0.0f;
                    for (std::size_t tap = 0; tap < kMaxLookaheadFrames; ++tap) {
                        const std::size_t frameIdx = (writeSlot + kMaxLookaheadFrames - tap) % kMaxLookaheadFrames;
                        const float       w        = timeFirWeights_[tap];
                        accEnv  += w * firEnvHistory_[frameIdx][s * paddedBins_ + k];
                        accFine += w * firFineHistory_[frameIdx][s * paddedBins_ + k];
                    }
                    envSpan[k]  = accEnv;
                    fineSpan[k] = accFine;
                }
            }
        } else if (timeSmoothCoeff_ > 1.0e-4f) {
            const float alpha    = timeSmoothCoeff_;
            const float oneMinus = 1.0f - alpha;

            for (std::size_t s = 0; s < slots; ++s) {
                auto   envSpan   = masks.binEnvSlot(s);
                auto   fineSpan  = masks.binFineSlot(s);
                float* stateEnv  = prevEnvState_.data() + s * paddedBins_;
                float* stateFine = prevFineState_.data() + s * paddedBins_;

                for (std::size_t k = 0; k < K; ++k) {
                    const float smEnv  = oneMinus * envSpan[k] + alpha * stateEnv[k];
                    const float smFine = oneMinus * fineSpan[k] + alpha * stateFine[k];
                    stateEnv[k]  = smEnv;
                    stateFine[k] = smFine;
                    envSpan[k]   = smEnv;
                    fineSpan[k]  = smFine;
                }
            }
        }

        // Enforce exact Partition of Unity and update per-bin tilePurity & dominantSampleId
        masks.updatePurityAndNormalizeBins(slots);
    }

private:
    void convolveFrequencyGaussian(std::span<float> binSlab, std::size_t K) noexcept {
        const auto r      = static_cast<std::ptrdiff_t>(freqRadius_);
        const auto maxIdx = static_cast<std::ptrdiff_t>(K) - 1;

        for (std::size_t k = 0; k < K; ++k) {
            const auto center = static_cast<std::ptrdiff_t>(k);
            float      acc    = 0.0f;
            for (std::ptrdiff_t d = -r; d <= r; ++d) {
                // Reflective boundary condition preserves constant sum across all sample slots
                std::ptrdiff_t idx = center + d;
                if (idx < 0) {
                    idx = -idx;
                } else if (idx > maxIdx) {
                    idx = 2 * maxIdx - idx;
                }
                idx = std::clamp<std::ptrdiff_t>(idx, 0, maxIdx);
                acc += freqKernel_[static_cast<std::size_t>(d + r)] * binSlab[static_cast<std::size_t>(idx)];
            }
            tempFreqScratch_[k] = acc;
        }

        for (std::size_t k = 0; k < K; ++k) {
            binSlab[k] = tempFreqScratch_[k];
        }
    }

    std::size_t numBins_{kDefaultNumBins};
    std::size_t paddedBins_{kDefaultPaddedBins};
    std::size_t freqRadius_{2};
    float       sigmaFreqBins_{1.5f};
    float       timeSmoothCoeff_{0.35f};
    bool        useFirLookahead_{false};
    std::size_t firWriteCursor_{0};

    std::array<float, kMaxFreqKernelSize>       freqKernel_{};
    std::array<float, kMaxLookaheadFrames>      timeFirWeights_{};
    AlignedVector<float>                        tempFreqScratch_;
    AlignedVector<float>                        prevEnvState_;
    AlignedVector<float>                        prevFineState_;
    std::array<AlignedVector<float>, kMaxLookaheadFrames> firEnvHistory_{};
    std::array<AlignedVector<float>, kMaxLookaheadFrames> firFineHistory_{};
};

// ============================================================================
// CausalMSPGHIReconstructor — Real-Time Multi-Source PGHI + Identity Phase Locking
// ============================================================================
/**
 * @brief Real-Time Causal Multi-Source Phase Gradient Heap Integration (MS-PGHI)
 *        with Laroche & Dolson Identity Phase Locking and Spectral Freeze dispersion
 *        (Section 1.4.1 & Section 3.2.2).
 *
 * Zero heap allocations in processFrame(). Uses a pre-allocated FlatMaxHeap<8192>.
 */
class alignas(kCacheLineBytes) CausalMSPGHIReconstructor {
public:
    static constexpr float       kPi           = std::numbers::pi_v<float>;
    static constexpr float       kTwoPi        = 2.0f * kPi;
    static constexpr std::size_t kMaxHeapNodes = 8192;

    enum class PhaseMode : std::uint8_t {
        RawSource          = 0,
        IdentityPhaseLock  = 1,
        HybridMSPGHI       = 2,
        SpectralFreezeBlur = 3
    };

    CausalMSPGHIReconstructor() {
        prepare(kDefaultFFTSize, kDefaultHopSize);
    }

    CausalMSPGHIReconstructor(std::size_t fftSize, std::size_t hopSize) {
        prepare(fftSize, hopSize);
    }

    void prepare(std::size_t fftSize, std::size_t hopSize) {
        fftSize_    = std::max<std::size_t>(64, fftSize);
        hopSize_    = std::max<std::size_t>(16, hopSize);
        numBins_    = (fftSize_ / 2) + 1;
        paddedBins_ = roundUpToSimdLanes(numBins_);

        // Time-frequency Gaussian window aspect ratio \kappa_{TF} = R * M / \lambda^2
        // For matching Gaussian analysis/synthesis window, \lambda \approx M / sqrt(8 * pi)
        const float mFloat = static_cast<float>(fftSize_);
        const float rFloat = static_cast<float>(hopSize_);
        const float lambda = mFloat / 5.0132565f;
        kappaTF_           = (rFloat * mFloat) / (lambda * lambda);

        prevLogMag_.assign(paddedBins_, -18.42068f);
        prevPhase_.assign(paddedBins_, 0.0f);
        prevFusedOmega_.assign(paddedBins_, 0.0f);
        prevMagnitude_.assign(paddedBins_, 0.0f);

        fusedOmega_.assign(paddedBins_, 0.0f);
        fusedTau_.assign(paddedBins_, 0.0f);
        lockedSourcePhase_.assign(paddedBins_, 0.0f);
        binVisited_.assign(paddedBins_, 0u);

        reset();
    }

    void reset() noexcept {
        std::fill(prevLogMag_.begin(), prevLogMag_.end(), -18.42068f);
        std::fill(prevPhase_.begin(), prevPhase_.end(), 0.0f);
        std::fill(prevFusedOmega_.begin(), prevFusedOmega_.end(), 0.0f);
        std::fill(prevMagnitude_.begin(), prevMagnitude_.end(), 0.0f);
        heap_.clear();
    }

    /**
     * @brief Master Polyphonic Source-Filter & HPR Magnitude Recombination (Section 1.4.2 Eq.):
     *
     *   |S_out(t, k)| = E_{rms}(t)^{1-\gamma}
     *                   * (\sum_{i=1}^N M_i^{env}(t, k) \tilde{E}_{v,i}(\tau_i, k))^\gamma
     *                   * (\sum_{j=1}^N M_j^{fine}(t, k) (g_H \tilde{F}_{H,v,j} + g_P \tilde{F}_{P,v,j} + g_R \tilde{F}_{R,v,j}))
     *
     * Also extracts the dominant sample's native phase \phi_{i^*}, instantaneous frequency \omega_{i^*}^{IF},
     * and group delay \tau_{i^*}^{GD} for MS-PGHI Phase Glue fusion.
     */
    static void synthesizeChimericMagnitude(
        std::span<const SpectralSampleData* const> samplePool,
        std::span<const std::size_t>               frameIndices,
        std::span<const float>                     pitchRatios,
        const MaskMatrix2D&                        masks,
        SpectralFrameSoA&                          outFrame,
        float                                      envelopeExponentGamma = 0.85f,
        float                                      formantKeytrackKappa  = 0.0f,
        float                                      formantShiftZeta      = 1.0f,
        float                                      gainHarmonic          = 1.0f,
        float                                      gainPercussive        = 1.0f,
        float                                      gainResidual          = 1.0f) noexcept {
        const std::size_t K     = std::min(outFrame.numBins, masks.numBins);
        const std::size_t slots = std::min(samplePool.size(), kMaxSampleSlots);
        const float       gamma = std::clamp(envelopeExponentGamma, 0.0f, 1.0f);
        const float       zeta  = std::clamp(formantShiftZeta, 0.25f, 4.0f);
        const float       kappa = std::clamp(formantKeytrackKappa, 0.0f, 1.0f);

        float rmsAccumSq = 0.0f;

        for (std::size_t k = 0; k < K; ++k) {
            float envMix  = 0.0f;
            float fineMix = 0.0f;
            float hMix    = 0.0f;
            float pMix    = 0.0f;
            float rMix    = 0.0f;

            const float kFloat = static_cast<float>(k);

            for (std::size_t i = 0; i < slots; ++i) {
                const SpectralSampleData* sample = samplePool[i];
                if (sample == nullptr || sample->numFrames == 0) {
                    continue;
                }

                const float wEnv  = masks.binMaskEnv[i * masks.paddedBins + k];
                const float wFine = masks.binMaskFine[i * masks.paddedBins + k];
                if (wEnv <= 1.0e-7f && wFine <= 1.0e-7f) {
                    continue;
                }

                const std::size_t tIdx = (i < frameIndices.size()) ? frameIndices[i] : 0;
                const float       rho  = (i < pitchRatios.size()) ? std::max(0.0625f, pitchRatios[i]) : 1.0f;

                // Scaled bin coordinates for Formant Envelope and Fine Structure (Section 1.4.2)
                const float kFineCoord = kFloat / rho;
                const float kEnvCoord  = kFloat / (zeta * std::pow(rho, kappa));

                if (wEnv > 1.0e-7f) {
                    const float eVal = SpectralSampleData::sampleHermiteAntiAliased(
                        sample->frameEnvelope(tIdx), kEnvCoord, sample->numBins);
                    envMix += wEnv * eVal;
                }

                if (wFine > 1.0e-7f) {
                    const float fH = gainHarmonic * SpectralSampleData::sampleHermiteAntiAliased(
                                                        sample->frameFineHarmonic(tIdx), kFineCoord, sample->numBins);
                    const float fP = gainPercussive * SpectralSampleData::sampleHermiteAntiAliased(
                                                          sample->frameFinePercussive(tIdx), kFineCoord, sample->numBins);
                    const float fR = gainResidual * SpectralSampleData::sampleHermiteAntiAliased(
                                                        sample->frameFineResidual(tIdx), kFineCoord, sample->numBins);

                    hMix    += wFine * fH;
                    pMix    += wFine * fP;
                    rMix    += wFine * fR;
                    fineMix += wFine * (fH + fP + fR);
                }
            }

            const float poweredEnv = (envMix > 1.0e-7f) ? std::pow(envMix, gamma) : 0.0f;
            const float outMag     = poweredEnv * fineMix;

            outFrame.spectralEnvelope[k] = envMix;
            outFrame.fineStructure[k]    = fineMix;
            outFrame.harmonicMag[k]      = poweredEnv * hMix;
            outFrame.percussiveMag[k]    = poweredEnv * pMix;
            outFrame.residualMag[k]      = poweredEnv * rMix;
            outFrame.magnitude[k]        = outMag;

            rmsAccumSq += outMag * outMag;

            // Populate dominant source phase and pitch-scaled instantaneous frequency \omega_{i^*}^{IF}
            const std::uint8_t domSlot = masks.dominantSampleId[k];
            if (domSlot < slots && samplePool[domSlot] != nullptr && samplePool[domSlot]->numFrames > 0) {
                const SpectralSampleData* domSample = samplePool[domSlot];
                const std::size_t domT   = (domSlot < frameIndices.size()) ? frameIndices[domSlot] : 0;
                const float       domRho = (domSlot < pitchRatios.size()) ? std::max(0.0625f, pitchRatios[domSlot]) : 1.0f;
                const std::size_t srcBin = std::min<std::size_t>(
                    static_cast<std::size_t>(kFloat / domRho + 0.5f), domSample->numBins - 1);

                outFrame.phase[k]      = domSample->framePhase(domT)[srcBin];
                outFrame.instFreq[k]   = domSample->frameInstFreq(domT)[srcBin] * domRho;
                outFrame.groupDelay[k] = domSample->frameGroupDelay(domT)[srcBin];
            }
        }

        // Apply E_{rms}(t)^{1 - \gamma} energy compensation factor
        const float rmsEnergy = std::sqrt(rmsAccumSq / static_cast<float>(K));
        if (rmsEnergy > 1.0e-6f && gamma < 0.999f) {
            const float rmsComp = std::pow(rmsEnergy, 1.0f - gamma);
            for (std::size_t k = 0; k < K; ++k) {
                outFrame.magnitude[k]     *= rmsComp;
                outFrame.harmonicMag[k]   *= rmsComp;
                outFrame.percussiveMag[k] *= rmsComp;
                outFrame.residualMag[k]   *= rmsComp;
            }
        }
    }

    /**
     * @brief Reconstructs a coherent phase spectrum \phi(t, k) and populates the Cartesian
     *        (real, imag) buffers of SpectralFrameSoA in real time.
     *
     * Combines:
     *   1. Cauchy-Riemann Log-Magnitude Phase Gradients (\hat{\omega}_{PGHI}, \hat{\tau}_{PGHI})
     *   2. Laroche & Dolson (1999) Identity Phase Locking around local spectral peaks
     *   3. Boundary-aware Phase Glue (\eta(t, k) = clamp(\beta_{glue} * (2 - c_{purity}(t, k)), 0, 1))
     *   4. Static FlatMaxHeap integration above -60 dB tolerance threshold
     *   5. Stochastic phase dispersion for Spectral Freeze
     */
    void reconstructPhase(SpectralFrameSoA&   frame,
                          const MaskMatrix2D& masks,
                          float               phaseGlueBeta  = 0.75f,
                          float               freezeAmount   = 0.0f,
                          float               toleranceDb    = 60.0f,
                          PhaseMode           mode           = PhaseMode::HybridMSPGHI) noexcept {
        const std::size_t K          = std::min(frame.numBins, numBins_);
        const float       betaGlue   = std::clamp(phaseGlueBeta, 0.0f, 1.0f);
        const float       freeze     = std::clamp(freezeAmount, 0.0f, 1.0f);
        const float       eps        = 1.0e-8f;
        const float       omegaStepK = kTwoPi * static_cast<float>(hopSize_) / static_cast<float>(fftSize_);

        // --------------------------------------------------------------------
        // Stage 1: Compute Log-Magnitude L(t, k) and find peak magnitude
        // --------------------------------------------------------------------
        float maxMag = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            const float mag        = std::max(0.0f, frame.magnitude[k]);
            maxMag                 = std::max(maxMag, mag);
            frame.logMagnitude[k]  = std::log(mag + eps);
        }

        // --------------------------------------------------------------------
        // Stage 2: Compute PGHI Gradients (\hat{\omega}_{PGHI}, \hat{\tau}_{PGHI})
        //          and fuse with Native Instantaneous Frequency via Phase Glue \eta(t, k)
        // --------------------------------------------------------------------
        const float halfKappa    = 0.5f * kappaTF_;
        const float invTwoKappa  = 1.0f / std::max(1.0e-5f, 2.0f * kappaTF_);

        for (std::size_t k = 0; k < K; ++k) {
            const std::size_t kPrev = (k > 0) ? (k - 1) : 0;
            const std::size_t kNext = (k + 1 < K) ? (k + 1) : (K - 1);

            const float dLdK = frame.logMagnitude[kNext] - frame.logMagnitude[kPrev];
            const float dLdT = frame.logMagnitude[k] - prevLogMag_[k];

            frame.dLogMagDf[k] = 0.5f * dLdK;
            frame.dLogMagDt[k] = dLdT;

            // Cauchy-Riemann PGHI gradient estimates (Section 1.4.1 Eq. 1)
            const float omegaPGHI = omegaStepK * static_cast<float>(k) + halfKappa * dLdK;
            const float tauPGHI   = -invTwoKappa * dLdT;

            // Tile purity c_{purity}(t, k) \in [1/N, 1] and Phase Glue blending weight \eta(t, k)
            const float cPurity = (k < masks.tilePurity.size()) ? masks.tilePurity[k] : 1.0f;
            const float eta     = std::clamp(betaGlue * (2.0f - cPurity), 0.0f, 1.0f);

            const float nativeOmega = frame.instFreq[k];
            const float nativeTau   = frame.groupDelay[k];

            fusedOmega_[k] = (1.0f - eta) * nativeOmega + eta * omegaPGHI;
            fusedTau_[k]   = (1.0f - eta) * nativeTau   + eta * tauPGHI;
        }

        // --------------------------------------------------------------------
        // Stage 3: Identity Phase Locking (Laroche & Dolson, 1999) for pure tiles
        // --------------------------------------------------------------------
        applyIdentityPhaseLocking(frame, K);

        // --------------------------------------------------------------------
        // Stage 4: Causal MS-PGHI Integration on Pre-Allocated FlatMaxHeap
        // --------------------------------------------------------------------
        if (mode == PhaseMode::HybridMSPGHI && maxMag > 1.0e-7f) {
            const float magThreshold = maxMag * std::pow(10.0f, -std::abs(toleranceDb) / 20.0f);

            heap_.clear();
            std::fill_n(binVisited_.data(), K, std::uint8_t{0});

            // Insert active bins from previous frame (t - 1, k) as known phase seeds
            for (std::size_t k = 0; k < K; ++k) {
                if (frame.magnitude[k] >= magThreshold) {
                    const float seedMag = std::max(prevMagnitude_[k], frame.magnitude[k] * 0.5f);
                    heap_.push(seedMag, static_cast<std::uint16_t>(k), 0u);
                } else {
                    // Below -60 dB threshold: advance phase linearly without heap overhead
                    frame.phase[k] = wrapPhasePi(prevPhase_[k] + fusedOmega_[k]);
                    binVisited_[k] = 1u;
                }
            }

            // Heap integration across time-frequency neighbors
            while (!heap_.empty()) {
                const PGHIHeapNode node = heap_.pop();
                const std::size_t  k    = node.binIndex;

                if (node.isCurrentFrame == 0u) {
                    // Propagate along time: (t - 1, k) -> (t, k) via trapezoidal rule
                    if (binVisited_[k] == 0u) {
                        const float integratedPhase = wrapPhasePi(
                            prevPhase_[k] + 0.5f * (prevFusedOmega_[k] + fusedOmega_[k]));

                        // Blend with Identity Phase-Locked phase inside pure tiles
                        const float cPurity = (k < masks.tilePurity.size()) ? masks.tilePurity[k] : 1.0f;
                        const float eta     = std::clamp(betaGlue * (2.0f - cPurity), 0.0f, 1.0f);

                        frame.phase[k] = slerpAngle(lockedSourcePhase_[k], integratedPhase, eta);
                        binVisited_[k] = 1u;

                        heap_.push(frame.magnitude[k], static_cast<std::uint16_t>(k), 1u);
                    }
                } else {
                    // Propagate along frequency: (t, k) -> (t, k + 1) and (t, k - 1)
                    if (k + 1 < K && binVisited_[k + 1] == 0u) {
                        frame.phase[k + 1] = wrapPhasePi(
                            frame.phase[k] + 0.5f * (fusedTau_[k] + fusedTau_[k + 1]));
                        binVisited_[k + 1] = 1u;
                        heap_.push(frame.magnitude[k + 1], static_cast<std::uint16_t>(k + 1), 1u);
                    }
                    if (k > 0 && binVisited_[k - 1] == 0u) {
                        frame.phase[k - 1] = wrapPhasePi(
                            frame.phase[k] - 0.5f * (fusedTau_[k] + fusedTau_[k - 1]));
                        binVisited_[k - 1] = 1u;
                        heap_.push(frame.magnitude[k - 1], static_cast<std::uint16_t>(k - 1), 1u);
                    }
                }
            }
        } else if (mode == PhaseMode::IdentityPhaseLock) {
            for (std::size_t k = 0; k < K; ++k) {
                frame.phase[k] = lockedSourcePhase_[k];
            }
        }

        // --------------------------------------------------------------------
        // Stage 5: Stochastic Phase Dispersion for Spectral Freeze / Blur
        //          & Polar-to-Cartesian SIMD-Friendly Conversion
        // --------------------------------------------------------------------
        if (freeze > 1.0e-4f || mode == PhaseMode::SpectralFreezeBlur) {
            const float effFreeze = (mode == PhaseMode::SpectralFreezeBlur) ? std::max(0.85f, freeze) : freeze;
            float       prevRand  = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                // Smoothly correlated random phase walk across adjacent bins prevents harsh metallic noise
                const float rawRand  = nextUniformMinusPiToPi();
                const float corrRand = 0.65f * rawRand + 0.35f * prevRand;
                prevRand             = rawRand;

                const float expectedAdvance = prevPhase_[k] + omegaStepK * static_cast<float>(k) + corrRand * effFreeze;
                frame.phase[k] = slerpAngle(frame.phase[k], wrapPhasePi(expectedAdvance), effFreeze);
            }
        }

        // Save history for next hop (t -> t - 1) and compute Cartesian (Re, Im)
        for (std::size_t k = 0; k < K; ++k) {
            const float phi    = frame.phase[k];
            const float mag    = frame.magnitude[k];
            prevLogMag_[k]     = frame.logMagnitude[k];
            prevPhase_[k]      = phi;
            prevFusedOmega_[k] = fusedOmega_[k];
            prevMagnitude_[k]  = mag;

            frame.real[k] = mag * std::cos(phi);
            frame.imag[k] = mag * std::sin(phi);
        }

        // Zero-pad tail SIMD lanes
        for (std::size_t k = K; k < frame.paddedBins; ++k) {
            frame.real[k] = 0.0f;
            frame.imag[k] = 0.0f;
        }
    }

    [[nodiscard]] static inline float wrapPhasePi(float angle) noexcept {
        float x = std::fmod(angle + kPi, kTwoPi);
        if (x < 0.0f) {
            x += kTwoPi;
        }
        return x - kPi;
    }

private:
    /**
     * @brief Laroche & Dolson (1999) Identity Phase Locking:
     *        Identifies local spectral peaks k_{peak} and locks the phase of surrounding
     *        bins in each peak's region of influence to preserve vertical phase coherence
     *        of sinusoids under time-stretching and pitch-shifting.
     */
    void applyIdentityPhaseLocking(const SpectralFrameSoA& frame, std::size_t K) noexcept {
        // Advance peak phases and lock neighboring bins to their dominant peak
        std::size_t currentPeak = 0;
        float       peakPhase   = wrapPhasePi(prevPhase_[0] + fusedOmega_[0]);

        for (std::size_t k = 0; k < K; ++k) {
            // Check if k is a 3-point local maximum
            const bool isLocalPeak = (k > 0 && k + 1 < K) &&
                                     (frame.magnitude[k] > frame.magnitude[k - 1]) &&
                                     (frame.magnitude[k] >= frame.magnitude[k + 1]);
            if (isLocalPeak) {
                currentPeak = k;
                peakPhase   = wrapPhasePi(prevPhase_[k] + fusedOmega_[k]);
            } else if (k > currentPeak + 1 && k + 1 < K) {
                // Switch region of influence at local spectral valley
                if (frame.magnitude[k] < frame.magnitude[k - 1] &&
                    frame.magnitude[k] <= frame.magnitude[k + 1]) {
                    // Look ahead for the next local peak
                    std::size_t nextPeak = k + 1;
                    while (nextPeak + 1 < K && frame.magnitude[nextPeak] < frame.magnitude[nextPeak + 1]) {
                        ++nextPeak;
                    }
                    currentPeak = nextPeak;
                    peakPhase   = wrapPhasePi(prevPhase_[currentPeak] + fusedOmega_[currentPeak]);
                }
            }

            // Identity Phase Locking relation: \phi(t, k) = \phi(t, k_p) + (\phi_{src}(t, k) - \phi_{src}(t, k_p))
            const float sourcePhaseDiff = frame.phase[k] - frame.phase[currentPeak];
            lockedSourcePhase_[k]       = wrapPhasePi(peakPhase + sourcePhaseDiff);
        }
    }

    /**
     * @brief Shortest-arc circular interpolation between two angles in [-pi, pi].
     */
    [[nodiscard]] static float slerpAngle(float a, float b, float t) noexcept {
        const float diff = wrapPhasePi(b - a);
        return wrapPhasePi(a + std::clamp(t, 0.0f, 1.0f) * diff);
    }

    /**
     * @brief Lock-free, real-time safe XorShift64* PRNG returning uniform values in [-pi, pi].
     */
    [[nodiscard]] float nextUniformMinusPiToPi() noexcept {
        prngState_ ^= prngState_ >> 12u;
        prngState_ ^= prngState_ << 25u;
        prngState_ ^= prngState_ >> 27u;
        const std::uint64_t val  = prngState_ * 0x2545F4914F6CDD1DULL;
        const float         unit = static_cast<float>(val >> 40u) * (1.0f / 16777216.0f); // [0, 1)
        return (unit * kTwoPi) - kPi;
    }

    std::size_t fftSize_{kDefaultFFTSize};
    std::size_t hopSize_{kDefaultHopSize};
    std::size_t numBins_{kDefaultNumBins};
    std::size_t paddedBins_{kDefaultPaddedBins};
    float       kappaTF_{5.0f};
    std::uint64_t prngState_{0x853C49E6748FEA9BULL};

    FlatMaxHeap<kMaxHeapNodes>   heap_;
    AlignedVector<float>         prevLogMag_;
    AlignedVector<float>         prevPhase_;
    AlignedVector<float>         prevFusedOmega_;
    AlignedVector<float>         prevMagnitude_;

    AlignedVector<float>         fusedOmega_;
    AlignedVector<float>         fusedTau_;
    AlignedVector<float>         lockedSourcePhase_;
    AlignedVector<std::uint8_t>  binVisited_;
};

} // namespace spectile::dsp
