#pragma once

#include "../core/SpectralTypesSoA.hpp"
#include "ERBScaleMapper.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

namespace spectile::dsp {

// ============================================================================
// Configuration Parameters for 3-Way HPR & Iterative Cepstral True Envelope
// ============================================================================
struct HPRConfig {
    std::size_t timeMedianLength{17};   // L_t = 17 frames (~180 ms at R=512, 48kHz)
    std::size_t freqMedianLength{17};   // L_k = 17 bins
    float       dominanceMargin{2.0f};  // \beta_{HPR} = 2.0
    float       sigmoidSteepness{4.0f}; // \gamma_m = 4.0
    float       epsilon{1.0e-8f};
};

struct TrueEnvelopeConfig {
    std::size_t numIterations{8};            // I_{TE} = 8 cepstral iterations
    std::size_t minCepstralOrder{24};        // Minimum Q_{max}
    std::size_t maxCepstralOrder{120};       // Maximum Q_{max}
    float       logMagFloor{1.0e-4f};        // -80 dB floor before ln()
    float       overshootDelta{0.02f};       // \delta_{overshoot} in Nepers
    float       tikhonovLambda{1.0e-6f};     // \lambda_{Tikh} regularization parameter
};

// ============================================================================
// HPRAndCepstrumProcessor — 3-Way HPR + Anti-Gibbs True Envelope & Tikhonov F_i
// ============================================================================
class alignas(kCacheLineBytes) HPRAndCepstrumProcessor {
public:
    static constexpr float       kPi               = std::numbers::pi_v<float>;
    static constexpr std::size_t kMaxMedianKernel  = 33;

    HPRAndCepstrumProcessor() {
        prepare(kDefaultFFTSize, 48000.0);
    }

    explicit HPRAndCepstrumProcessor(std::size_t fftSize, double sampleRate = 48000.0) {
        prepare(fftSize, sampleRate);
    }

    /**
     * @brief Pre-allocates all aligned scratch buffers and Radix-2 FFT twiddle tables.
     */
    void prepare(std::size_t fftSize, double sampleRate) {
        // Ensure power-of-two FFT size for symmetric DCT-I / Cepstral FFT
        fftSize_    = std::bit_ceil(std::max<std::size_t>(64, fftSize));
        sampleRate_ = std::max(8000.0, sampleRate);
        numBins_    = (fftSize_ / 2) + 1;
        paddedBins_ = roundUpToSimdLanes(numBins_);

        horizMedianScratch_.assign(paddedBins_, 0.0f);
        vertMedianScratch_.assign(paddedBins_, 0.0f);

        // Cepstral buffers over M = 2 * (K - 1) = fftSize_ points (even symmetry)
        cepLogTarget_.assign(fftSize_, 0.0f);
        cepCurrentEnv_.assign(fftSize_, 0.0f);
        cepFftReal_.assign(fftSize_, 0.0f);
        cepFftImag_.assign(fftSize_, 0.0f);
        cepWindow_.assign(fftSize_, 0.0f);

        precomputeFftTables(fftSize_);
    }

    // ========================================================================
    // 1. 3-Way HPR (Harmonic-Percussive-Residual) Decomposition (Section 1.3.1)
    // ========================================================================

    /**
     * @brief Computes the 3-way soft Wiener masks (W_H, W_P, W_R) and decomposed magnitudes
     *        (H, P, R) for a single frame given horizontal (\tilde{H}) and vertical (\tilde{P})
     *        median estimates.
     *
     * Guarantees exact conservation: H(t,k) + P(t,k) + R(t,k) == |S(t,k)|.
     */
    static void computeSoftWienerHPR(std::span<const float> magnitude,
                                     std::span<const float> horizMedian,
                                     std::span<const float> vertMedian,
                                     std::span<float>       outHarmonic,
                                     std::span<float>       outPercussive,
                                     std::span<float>       outResidual,
                                     std::size_t            numBins,
                                     const HPRConfig&       cfg = {}) noexcept {
        const float eps   = cfg.epsilon;
        const float beta  = cfg.dominanceMargin;
        const float gamma = cfg.sigmoidSteepness;

        for (std::size_t k = 0; k < numBins; ++k) {
            const float sMag   = std::max(0.0f, magnitude[k]);
            const float hTilde = std::max(0.0f, horizMedian[k]);
            const float pTilde = std::max(0.0f, vertMedian[k]);

            const float rH = hTilde / (pTilde + eps);
            const float rP = pTilde / (hTilde + eps);

            const float sigH = fastSigmoid(gamma * (rH - beta));
            const float sigP = fastSigmoid(gamma * (rP - beta));

            const float wH = (hTilde * hTilde) * sigH;
            const float wP = (pTilde * pTilde) * sigP;
            const float wR = (sMag * sMag) * (1.0f - sigH) * (1.0f - sigP);

            const float wSum = wH + wP + wR;
            float maskH = 0.0f;
            float maskP = 0.0f;
            float maskR = 1.0f;

            if (wSum > eps) {
                const float invSum = 1.0f / wSum;
                maskH = wH * invSum;
                maskP = wP * invSum;
                maskR = std::max(0.0f, 1.0f - (maskH + maskP));
            }

            outHarmonic[k]   = sMag * maskH;
            outPercussive[k] = sMag * maskP;
            outResidual[k]   = sMag * maskR;
        }
    }

    /**
     * @brief Performs 1D vertical median filtering along the frequency axis k for a single frame.
     */
    static void medianFilterFrequency1D(std::span<const float> inMag,
                                        std::span<float>       outVertMedian,
                                        std::size_t            numBins,
                                        std::size_t            kernelLen = 17) noexcept {
        const std::size_t effLen = std::clamp<std::size_t>(kernelLen | 1u, 3, kMaxMedianKernel);
        const auto        halfW  = static_cast<std::ptrdiff_t>(effLen / 2);
        const auto        maxIdx = static_cast<std::ptrdiff_t>(numBins) - 1;

        std::array<float, kMaxMedianKernel> window{};

        for (std::size_t k = 0; k < numBins; ++k) {
            const auto center = static_cast<std::ptrdiff_t>(k);
            for (std::size_t m = 0; m < effLen; ++m) {
                const std::ptrdiff_t idx = std::clamp<std::ptrdiff_t>(
                    center + static_cast<std::ptrdiff_t>(m) - halfW, 0, maxIdx);
                window[m] = inMag[static_cast<std::size_t>(idx)];
            }
            auto midIt = window.begin() + static_cast<std::ptrdiff_t>(effLen / 2);
            std::nth_element(window.begin(), midIt, window.begin() + static_cast<std::ptrdiff_t>(effLen));
            outVertMedian[k] = *midIt;
        }
    }

    /**
     * @brief Decomposes an entire multi-frame SpectralSampleData into 3-way HPR components
     *        using 2D directional median filtering (L_t along time, L_k along frequency)
     *        followed by 3-way soft Wiener masking.
     */
    void decomposeSampleHPR(SpectralSampleData& sample, const HPRConfig& cfg = {}) {
        if (sample.numFrames == 0 || sample.numBins == 0) {
            return;
        }
        if (sample.fftSize != fftSize_) {
            prepare(sample.fftSize, sample.sampleRate);
        }

        const std::size_t T      = sample.numFrames;
        const std::size_t K      = sample.numBins;
        const std::size_t effLt  = std::clamp<std::size_t>(cfg.timeMedianLength | 1u, 1, kMaxMedianKernel);
        const std::size_t effLk  = std::clamp<std::size_t>(cfg.freqMedianLength | 1u, 3, kMaxMedianKernel);
        const auto        halfLt = static_cast<std::ptrdiff_t>(effLt / 2);
        const auto        maxT   = static_cast<std::ptrdiff_t>(T) - 1;

        std::array<float, kMaxMedianKernel> timeWindow{};

        for (std::size_t t = 0; t < T; ++t) {
            const auto tCenter = static_cast<std::ptrdiff_t>(t);

            // 1. Horizontal (time-axis) median filter per bin k
            for (std::size_t k = 0; k < K; ++k) {
                for (std::size_t m = 0; m < effLt; ++m) {
                    const std::ptrdiff_t tSample = std::clamp<std::ptrdiff_t>(
                        tCenter + static_cast<std::ptrdiff_t>(m) - halfLt, 0, maxT);
                    timeWindow[m] = sample.frameMagnitude(static_cast<std::size_t>(tSample))[k];
                }
                auto midIt = timeWindow.begin() + static_cast<std::ptrdiff_t>(effLt / 2);
                std::nth_element(timeWindow.begin(), midIt, timeWindow.begin() + static_cast<std::ptrdiff_t>(effLt));
                horizMedianScratch_[k] = *midIt;
            }

            // 2. Vertical (frequency-axis) median filter for frame t
            medianFilterFrequency1D(sample.frameMagnitude(t), vertMedianScratch_, K, effLk);

            // 3. Soft 3-Way Wiener Masking
            computeSoftWienerHPR(sample.frameMagnitude(t),
                                 horizMedianScratch_,
                                 vertMedianScratch_,
                                 sample.mutableFrameHarmonic(t),
                                 sample.mutableFramePercussive(t),
                                 sample.mutableFrameResidual(t),
                                 K,
                                 cfg);
        }
    }

    // ========================================================================
    // 2. Iterative Cepstral True Envelope & Tikhonov Fine Structure (Section 1.3.2)
    // ========================================================================

    /**
     * @brief Computes the adaptive maximum cepstral order Q_{max}:
     *        Q_{max} = clamp(floor(f_s / (2.2 * f_0)), 24, 120)
     */
    [[nodiscard]] std::size_t computeAdaptiveCepstralOrder(float                     fundamentalHz,
                                                           const TrueEnvelopeConfig& cfg = {}) const noexcept {
        const float f0   = std::clamp(fundamentalHz, 20.0f, 4000.0f);
        const auto  rawQ = static_cast<std::size_t>(sampleRate_ / (2.2 * static_cast<double>(f0)));
        return std::clamp(rawQ, cfg.minCepstralOrder, cfg.maxCepstralOrder);
    }

    /**
     * @brief Extracts the True Spectral Envelope E(t, k) via iterative cepstral liftering
     *        with even-symmetry DCT-I boundary extension (anti-Gibbs at 0 Hz and Nyquist),
     *        and computes the whitened Fine Structure F(t, k) in [0, 1] via Tikhonov
     *        regularized deconvolution:
     *          F_i(t, k) = clamp(|S_i(t, k)| * E_i(t, k) / (E_i(t, k)^2 + \lambda_{Tikh}), 0, 1)
     */
    void extractTrueEnvelopeAndFineStructure(std::span<const float>    magnitude,
                                             std::span<const float>    harmonicMag,
                                             std::span<const float>    percussiveMag,
                                             std::span<const float>    residualMag,
                                             std::span<float>          outEnvelope,
                                             std::span<float>          outFineStructure,
                                             std::span<float>          outFineHarmonic,
                                             std::span<float>          outFinePercussive,
                                             std::span<float>          outFineResidual,
                                             float                     fundamentalHz = 220.0f,
                                             const TrueEnvelopeConfig& cfg           = {}) noexcept {
        const std::size_t M    = fftSize_;
        const std::size_t K    = numBins_;
        const std::size_t qMax = std::min(computeAdaptiveCepstralOrder(fundamentalHz, cfg), (M / 2) - 1);

        // 1. Build symmetric log-magnitude target \tilde{A}_0(m) over M = 2*(K - 1) points
        const float floorVal = std::max(1.0e-7f, cfg.logMagFloor);
        for (std::size_t k = 0; k < K; ++k) {
            const float mag    = (k < magnitude.size()) ? std::max(magnitude[k], floorVal) : floorVal;
            const float logMag = std::log(mag);
            cepLogTarget_[k]   = logMag;
            cepCurrentEnv_[k]  = logMag;
        }
        // Mirror extension \tilde{A}_0(M - m) = A_0(m) for m = 1 .. K-2 (exact even symmetry)
        for (std::size_t k = 1; k < K - 1; ++k) {
            cepLogTarget_[M - k]  = cepLogTarget_[k];
            cepCurrentEnv_[M - k] = cepCurrentEnv_[k];
        }

        // 2. Build raised-cosine low-pass cepstral lifter window W_{cep}(q)
        std::fill(cepWindow_.begin(), cepWindow_.end(), 0.0f);
        cepWindow_[0] = 1.0f;
        for (std::size_t q = 1; q <= qMax; ++q) {
            const float w = 0.5f * (1.0f + std::cos(kPi * static_cast<float>(q) / static_cast<float>(qMax + 1)));
            cepWindow_[q]     = w;
            cepWindow_[M - q] = w;
        }

        // 3. Iterative True Envelope Refinement (I_{TE} = 8 iterations)
        const std::size_t iters = std::max<std::size_t>(1, cfg.numIterations);
        for (std::size_t iter = 0; iter < iters; ++iter) {
            for (std::size_t m = 0; m < M; ++m) {
                cepFftReal_[m] = cepCurrentEnv_[m];
                cepFftImag_[m] = 0.0f;
            }

            // Forward FFT: Log-Spectrum -> Cepstrum
            radix2Fft(cepFftReal_, cepFftImag_, false);

            // Liftering in Quefrency domain: \hat{c}_j(q) = W_{cep}(q) * c_j(q)
            for (std::size_t q = 0; q < M; ++q) {
                cepFftReal_[q] *= cepWindow_[q];
                cepFftImag_[q] *= cepWindow_[q];
            }

            // Inverse FFT: Liftered Cepstrum -> Smooth Log-Envelope V_j(m)
            radix2Fft(cepFftReal_, cepFftImag_, true);

            // Upper-envelope constraint: \tilde{A}_{j+1}(m) = max(\tilde{A}_0(m), V_j(m) - \delta_{overshoot})
            for (std::size_t m = 0; m < M; ++m) {
                cepCurrentEnv_[m] = std::max(cepLogTarget_[m], cepFftReal_[m] - cfg.overshootDelta);
            }
        }

        // 4. Final pass to guarantee C^\infty band-limited smoothness of V_{I_{TE}}(k)
        for (std::size_t m = 0; m < M; ++m) {
            cepFftReal_[m] = cepCurrentEnv_[m];
            cepFftImag_[m] = 0.0f;
        }
        radix2Fft(cepFftReal_, cepFftImag_, false);
        for (std::size_t q = 0; q < M; ++q) {
            cepFftReal_[q] *= cepWindow_[q];
            cepFftImag_[q] *= cepWindow_[q];
        }
        radix2Fft(cepFftReal_, cepFftImag_, true);

        // 5. Exponentiate E_i(t,k) = exp(V_8(k)) and apply Tikhonov Regularized Deconvolution for F_i(t,k) in [0, 1]
        const float lambdaTikh = std::max(1.0e-12f, cfg.tikhonovLambda);
        const bool  hasHPR     = (harmonicMag.size() >= K &&
                                  percussiveMag.size() >= K &&
                                  residualMag.size() >= K);

        for (std::size_t k = 0; k < K; ++k) {
            const float sMag = (k < magnitude.size()) ? std::max(0.0f, magnitude[k]) : 0.0f;
            // Enforce E_i(t, k) >= |S_i(t, k)| so Tikhonov ratio is strictly inside [0, 1]
            const float env  = std::max({std::exp(cepFftReal_[k]), sMag, floorVal});
            const float denom = (env * env) + lambdaTikh;
            const float invDeconv = env / denom;

            const float fTot = std::clamp(sMag * invDeconv, 0.0f, 1.0f);

            if (k < outEnvelope.size())      { outEnvelope[k]      = env; }
            if (k < outFineStructure.size()) { outFineStructure[k] = fTot; }

            if (hasHPR) {
                const float h = std::max(0.0f, harmonicMag[k]);
                const float p = std::max(0.0f, percussiveMag[k]);
                const float r = std::max(0.0f, residualMag[k]);

                if (k < outFineHarmonic.size())   { outFineHarmonic[k]   = std::clamp(h * invDeconv, 0.0f, 1.0f); }
                if (k < outFinePercussive.size()) { outFinePercussive[k] = std::clamp(p * invDeconv, 0.0f, 1.0f); }
                if (k < outFineResidual.size())   { outFineResidual[k]   = std::clamp(r * invDeconv, 0.0f, 1.0f); }
            } else {
                if (k < outFineHarmonic.size())   { outFineHarmonic[k]   = fTot; }
                if (k < outFinePercussive.size()) { outFinePercussive[k] = 0.0f; }
                if (k < outFineResidual.size())   { outFineResidual[k]   = 0.0f; }
            }
        }
    }

    /**
     * @brief Runs the complete offline pre-analysis pipeline (HPR + True Envelope + Tikhonov Fine Structure
     *        + ERB Band Descriptors) on a SpectralSampleData instance.
     */
    void analyzeSampleComplete(SpectralSampleData&       sample,
                               const ERBScaleMapper&     erbMapper,
                               const HPRConfig&          hprCfg = {},
                               const TrueEnvelopeConfig& teCfg  = {}) {
        if (sample.numFrames == 0 || sample.numBins == 0) {
            return;
        }
        if (sample.fftSize != fftSize_) {
            prepare(sample.fftSize, sample.sampleRate);
        }

        // Step 1: 2D HPR Separation across all frames
        decomposeSampleHPR(sample, hprCfg);

        // Step 2: Per-frame True Envelope, Tikhonov Fine Structure, and ERB Band Descriptors
        for (std::size_t t = 0; t < sample.numFrames; ++t) {
            extractTrueEnvelopeAndFineStructure(
                sample.frameMagnitude(t),
                sample.frameHarmonic(t),
                sample.framePercussive(t),
                sample.frameResidual(t),
                sample.mutableFrameEnvelope(t),
                sample.mutableFrameFineStructure(t),
                sample.mutableFrameFineHarmonic(t),
                sample.mutableFrameFinePercussive(t),
                sample.mutableFrameFineResidual(t),
                sample.fundamentalHz,
                teCfg);

            const std::size_t prevT = (t > 0) ? (t - 1) : 0;
            erbMapper.computeBandDescriptors(
                sample.frameMagnitude(t),
                sample.frameMagnitude(prevT),
                sample.frameHarmonic(t),
                sample.framePercussive(t),
                sample.frameResidual(t),
                sample.mutableFrameERBEnergy(t),
                sample.mutableFrameERBFlatness(t),
                sample.mutableFrameERBFlux(t),
                sample.mutableFrameERBHarmonicity(t));
        }
    }

private:
    [[nodiscard]] static float fastSigmoid(float x) noexcept {
        const float clamped = std::clamp(x, -20.0f, 20.0f);
        return 1.0f / (1.0f + std::exp(-clamped));
    }

    void precomputeFftTables(std::size_t n) {
        const std::size_t log2N = static_cast<std::size_t>(std::countr_zero(n));
        bitRevTable_.assign(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t rev = 0;
            for (std::size_t b = 0; b < log2N; ++b) {
                rev = (rev << 1u) | ((i >> b) & 1u);
            }
            bitRevTable_[i] = static_cast<std::uint32_t>(rev);
        }

        twiddleCos_.assign(n / 2, 0.0f);
        twiddleSin_.assign(n / 2, 0.0f);
        for (std::size_t k = 0; k < n / 2; ++k) {
            const float angle = -2.0f * kPi * static_cast<float>(k) / static_cast<float>(n);
            twiddleCos_[k]    = std::cos(angle);
            twiddleSin_[k]    = std::sin(angle);
        }
    }

    /**
     * @brief In-place Radix-2 Cooley-Tukey FFT/IFFT using pre-tabulated bit-reversal and twiddle factors
     *        (zero dynamic allocations).
     */
    void radix2Fft(AlignedVector<float>& real, AlignedVector<float>& imag, bool inverse) noexcept {
        const std::size_t n = fftSize_;

        // Bit-reversal permutation
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t j = bitRevTable_[i];
            if (i < j) {
                std::swap(real[i], real[j]);
                std::swap(imag[i], imag[j]);
            }
        }

        const float sign = inverse ? -1.0f : 1.0f;

        // Butterfly stages
        for (std::size_t len = 2; len <= n; len <<= 1u) {
            const std::size_t halfLen    = len >> 1u;
            const std::size_t twiddleStep = n / len;

            for (std::size_t i = 0; i < n; i += len) {
                for (std::size_t j = 0; j < halfLen; ++j) {
                    const std::size_t twIdx = j * twiddleStep;
                    const float       wCos  = twiddleCos_[twIdx];
                    const float       wSin  = sign * twiddleSin_[twIdx];

                    const std::size_t uIdx = i + j;
                    const std::size_t vIdx = uIdx + halfLen;

                    const float tReal = wCos * real[vIdx] - wSin * imag[vIdx];
                    const float tImag = wCos * imag[vIdx] + wSin * real[vIdx];

                    real[vIdx] = real[uIdx] - tReal;
                    imag[vIdx] = imag[uIdx] - tImag;
                    real[uIdx] += tReal;
                    imag[uIdx] += tImag;
                }
            }
        }

        if (inverse) {
            const float invN = 1.0f / static_cast<float>(n);
            for (std::size_t i = 0; i < n; ++i) {
                real[i] *= invN;
                imag[i] *= invN;
            }
        }
    }

    std::size_t fftSize_{kDefaultFFTSize};
    double      sampleRate_{48000.0};
    std::size_t numBins_{kDefaultNumBins};
    std::size_t paddedBins_{kDefaultPaddedBins};

    AlignedVector<float> horizMedianScratch_;
    AlignedVector<float> vertMedianScratch_;

    AlignedVector<float> cepLogTarget_;
    AlignedVector<float> cepCurrentEnv_;
    AlignedVector<float> cepFftReal_;
    AlignedVector<float> cepFftImag_;
    AlignedVector<float> cepWindow_;

    AlignedVector<std::uint32_t> bitRevTable_;
    AlignedVector<float>         twiddleCos_;
    AlignedVector<float>         twiddleSin_;
};

} // namespace spectile::dsp
