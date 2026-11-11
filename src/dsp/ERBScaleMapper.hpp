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
// Psychoacoustic & Musical Frequency Scale Modes
// ============================================================================
enum class FrequencyScaleMode : std::uint8_t {
    ERB      = 0, // Glasberg & Moore (1990) Equivalent Rectangular Bandwidth
    Bark     = 1, // Traunmüller (1990) Critical Band Rate
    Harmonic = 2, // Fundamental-anchored quasi-harmonic partial scale
    Linear   = 3  // Uniform Hertz scale
};

// ============================================================================
// ERBScaleMapper — Psychoacoustic K <-> B Mapping & Cosine-Tapered Projection
// ============================================================================
/**
 * @brief Implements exact psychoacoustic scale transformations (ERB, Bark, Harmonic, Linear),
 *        K -> B spectral descriptor aggregation (erbEnergy, erbFlatness, erbFlux, erbHarmonic),
 *        and C^1-continuous Cosine-Tapered B -> K mask upsampling preserving the Partition of Unity.
 */
class alignas(kCacheLineBytes) ERBScaleMapper {
public:
    static constexpr float kPi      = std::numbers::pi_v<float>;
    static constexpr float kEpsilon = 1.0e-8f;

    ERBScaleMapper() {
        configure(kDefaultFFTSize, 48000.0, kMaxERBBands, FrequencyScaleMode::ERB, 110.0f);
    }

    ERBScaleMapper(std::size_t        fftSize,
                   double             sampleRate,
                   std::size_t        numBands    = kMaxERBBands,
                   FrequencyScaleMode scaleMode   = FrequencyScaleMode::ERB,
                   float              refPitchHz  = 110.0f) {
        configure(fftSize, sampleRate, numBands, scaleMode, refPitchHz);
    }

    // ========================================================================
    // Exact Psychoacoustic Scale Formulas
    // ========================================================================

    /**
     * @brief Glasberg & Moore (1990) Equivalent Rectangular Bandwidth in Hz:
     *        ERB(f) = 24.7 * (4.37 * (f / 1000) + 1)
     */
    [[nodiscard]] static constexpr float hzToERBBandwidth(float freqHz) noexcept {
        const float f = std::max(0.0f, freqHz);
        return 24.7f * (0.00437f * f + 1.0f);
    }

    /**
     * @brief Glasberg & Moore (1990) ERB-rate scale in Cams:
     *        E_{ERB}(f) = 21.4 * log10(0.00437 * f + 1)
     */
    [[nodiscard]] static float hzToERBRate(float freqHz) noexcept {
        const float f = std::max(0.0f, freqHz);
        return 21.4f * std::log10(0.00437f * f + 1.0f);
    }

    /**
     * @brief Inverse Glasberg & Moore (1990) ERB-rate (Cams -> Hz):
     *        f = (10^(E_{ERB} / 21.4) - 1) / 0.00437
     */
    [[nodiscard]] static float erbRateToHz(float erbCams) noexcept {
        const float e = std::max(0.0f, erbCams);
        return (std::pow(10.0f, e / 21.4f) - 1.0f) / 0.00437f;
    }

    /**
     * @brief Traunmüller (1990) Critical Band Rate in Barks:
     *        z(f) = 26.81 * f / (1960 + f) - 0.53 (with end-band corrections)
     */
    [[nodiscard]] static constexpr float hzToBark(float freqHz) noexcept {
        const float f = std::max(0.0f, freqHz);
        float z = (26.81f * f) / (1960.0f + f) - 0.53f;
        if (z < 2.0f) {
            z += 0.15f * (2.0f - z);
        } else if (z > 20.1f) {
            z += 0.22f * (z - 20.1f);
        }
        return std::max(0.0f, z);
    }

    /**
     * @brief Inverse Traunmüller (1990) Bark -> Hz:
     */
    [[nodiscard]] static constexpr float barkToHz(float bark) noexcept {
        float z = std::clamp(bark, 0.0f, 26.2f);
        if (z < 2.0f) {
            z = (z - 0.30f) / 0.85f;
        } else if (z > 20.1f) {
            z = (z + 4.422f) / 1.22f;
        }
        const float zShift = z + 0.53f;
        const float denom  = std::max(0.05f, 26.81f - zShift);
        return std::max(0.0f, 1960.0f * zShift / denom);
    }

    /**
     * @brief Quasi-Harmonic log-partial scale anchored at refPitchHz:
     *        H(f) = log2(1 + f / f_0)
     */
    [[nodiscard]] static float hzToHarmonicScale(float freqHz, float refPitchHz = 110.0f) noexcept {
        const float f0 = std::max(10.0f, refPitchHz);
        const float f  = std::max(0.0f, freqHz);
        return std::log2(1.0f + (f / f0));
    }

    /**
     * @brief Inverse Quasi-Harmonic scale -> Hz:
     */
    [[nodiscard]] static float harmonicScaleToHz(float harmCoord, float refPitchHz = 110.0f) noexcept {
        const float f0 = std::max(10.0f, refPitchHz);
        const float h  = std::max(0.0f, harmCoord);
        return f0 * (std::exp2(h) - 1.0f);
    }

    // ========================================================================
    // Configuration & Pre-Tabulation of Lookup Tables
    // ========================================================================
    void configure(std::size_t        fftSize,
                   double             sampleRate,
                   std::size_t        numBands   = kMaxERBBands,
                   FrequencyScaleMode scaleMode  = FrequencyScaleMode::ERB,
                   float              refPitchHz = 110.0f) {
        fftSize_     = std::max<std::size_t>(64, fftSize);
        sampleRate_  = std::max(8000.0, sampleRate);
        numBins_     = (fftSize_ / 2) + 1;
        paddedBins_  = roundUpToSimdLanes(numBins_);
        numBands_    = std::clamp<std::size_t>(numBands, 2, kMaxERBBands);
        scaleMode_   = scaleMode;
        refPitchHz_  = std::max(10.0f, refPitchHz);

        binFreqHz_.assign(paddedBins_, 0.0f);
        binContBandCoord_.assign(paddedBins_, 0.0f);
        lowerBandIdx_.assign(paddedBins_, 0);
        upperBandIdx_.assign(paddedBins_, 0);
        upsampleWeight_.assign(paddedBins_, 0.0f);

        bandCenterHz_.fill(0.0f);
        bandNormWeight_.fill(0.0f);

        const float nyquistHz = static_cast<float>(sampleRate_ * 0.5);
        const float minScale  = hzToScaleUnit(0.0f);
        const float maxScale  = std::max(minScale + 1.0e-5f, hzToScaleUnit(nyquistHz));
        const float invSpan   = 1.0f / (maxScale - minScale);

        // Compute band center frequencies in Hz
        for (std::size_t b = 0; b < numBands_; ++b) {
            const float normBand = static_cast<float>(b) / static_cast<float>(numBands_ - 1);
            const float scaleVal = minScale + normBand * (maxScale - minScale);
            bandCenterHz_[b]     = scaleUnitToHz(scaleVal);
        }

        // Pre-tabulate continuous band coordinate \xi(k), b0(k), b1(k), and cosine weight w_up(k)
        const float binHzStep = static_cast<float>(sampleRate_) / static_cast<float>(fftSize_);
        for (std::size_t k = 0; k < numBins_; ++k) {
            const float freqHz = static_cast<float>(k) * binHzStep;
            binFreqHz_[k]      = freqHz;

            const float scaleVal = hzToScaleUnit(freqHz);
            const float normPos  = std::clamp((scaleVal - minScale) * invSpan, 0.0f, 1.0f);
            const float xi       = normPos * static_cast<float>(numBands_ - 1);

            binContBandCoord_[k] = xi;

            const auto b0 = std::min<std::size_t>(static_cast<std::size_t>(xi), numBands_ - 1);
            const auto b1 = std::min<std::size_t>(b0 + 1, numBands_ - 1);
            const float alpha = std::clamp(xi - static_cast<float>(b0), 0.0f, 1.0f);

            // C^1-continuous raised-cosine weight: w_up(k) = 0.5 * (1 - cos(pi * alpha(k)))
            const float wUp = 0.5f * (1.0f - std::cos(kPi * alpha));

            lowerBandIdx_[k]   = static_cast<std::uint16_t>(b0);
            upperBandIdx_[k]   = static_cast<std::uint16_t>(b1);
            upsampleWeight_[k] = wUp;

            // Accumulate dual-band aggregation weights for K -> B reduction
            bandNormWeight_[b0] += (1.0f - wUp);
            bandNormWeight_[b1] += wUp;
        }

        // Pad tail elements up to paddedBins_
        for (std::size_t k = numBins_; k < paddedBins_; ++k) {
            binFreqHz_[k]        = nyquistHz;
            binContBandCoord_[k] = static_cast<float>(numBands_ - 1);
            lowerBandIdx_[k]     = static_cast<std::uint16_t>(numBands_ - 1);
            upperBandIdx_[k]     = static_cast<std::uint16_t>(numBands_ - 1);
            upsampleWeight_[k]   = 0.0f;
        }

        // Convert accumulated band weight sums into reciprocal normalization factors
        for (std::size_t b = 0; b < numBands_; ++b) {
            if (bandNormWeight_[b] > 1.0e-6f) {
                bandNormWeight_[b] = 1.0f / bandNormWeight_[b];
            } else {
                bandNormWeight_[b] = 1.0f;
            }
        }
    }

    // ========================================================================
    // K -> B Aggregation of the 4 Spectral Descriptors
    // ========================================================================
    /**
     * @brief Aggregates K FFT bins into B psychoacoustic bands to compute the 4 descriptors:
     *        1. erbEnergy   : Normalized per-band RMS energy \hat{E}_i(t, b)
     *        2. erbFlatness : Wiener entropy (geometric mean / arithmetic mean) in [0, 1]
     *        3. erbFlux     : Half-wave rectified spectral flux in [0, 1]
     *        4. erbHarmonic : Harmonic dominance ratio H^2 / (H^2 + P^2 + R^2 + eps) in [0, 1]
     */
    void computeBandDescriptors(std::span<const float> magnitude,
                                std::span<const float> prevMagnitude,
                                std::span<const float> harmonicMag,
                                std::span<const float> percussiveMag,
                                std::span<const float> residualMag,
                                std::span<float>       outErbEnergy,
                                std::span<float>       outErbFlatness,
                                std::span<float>       outErbFlux,
                                std::span<float>       outErbHarmonic) const noexcept {
        std::array<float, kMaxERBBands> sumMag{};
        std::array<float, kMaxERBBands> sumMagSq{};
        std::array<float, kMaxERBBands> sumLogMag{};
        std::array<float, kMaxERBBands> sumPosFlux{};
        std::array<float, kMaxERBBands> sumHarmSq{};
        std::array<float, kMaxERBBands> sumTotalHPRSq{};
        std::array<float, kMaxERBBands> sumWeights{};

        const bool hasPrev = (prevMagnitude.size() >= numBins_);
        const bool hasHPR  = (harmonicMag.size() >= numBins_ &&
                              percussiveMag.size() >= numBins_ &&
                              residualMag.size() >= numBins_);

        for (std::size_t k = 0; k < numBins_; ++k) {
            const float mag    = std::max(magnitude[k], 0.0f);
            const float magEps = std::max(mag, kEpsilon);
            const float logMag = std::log(magEps);
            const float magSq  = mag * mag;

            const float prevM = hasPrev ? std::max(prevMagnitude[k], 0.0f) : mag;
            const float dMag  = std::max(0.0f, mag - prevM);

            float hSq   = magSq;
            float totSq = magSq;
            if (hasHPR) {
                const float h = harmonicMag[k];
                const float p = percussiveMag[k];
                const float r = residualMag[k];
                hSq   = h * h;
                totSq = hSq + (p * p) + (r * r);
            }

            const std::size_t b0  = lowerBandIdx_[k];
            const std::size_t b1  = upperBandIdx_[k];
            const float       w1  = upsampleWeight_[k];
            const float       w0  = 1.0f - w1;

            // Band b0 contribution
            sumMag[b0]        += w0 * mag;
            sumMagSq[b0]      += w0 * magSq;
            sumLogMag[b0]     += w0 * logMag;
            sumPosFlux[b0]    += w0 * dMag;
            sumHarmSq[b0]     += w0 * hSq;
            sumTotalHPRSq[b0] += w0 * totSq;
            sumWeights[b0]    += w0;

            // Band b1 contribution
            sumMag[b1]        += w1 * mag;
            sumMagSq[b1]      += w1 * magSq;
            sumLogMag[b1]     += w1 * logMag;
            sumPosFlux[b1]    += w1 * dMag;
            sumHarmSq[b1]     += w1 * hSq;
            sumTotalHPRSq[b1] += w1 * totSq;
            sumWeights[b1]    += w1;
        }

        // In case a very low frequency band is narrower than 1 FFT bin, interpolate from nearest populated band
        for (std::size_t b = 0; b < numBands_; ++b) {
            if (sumWeights[b] <= 1.0e-6f) {
                const std::size_t kNear = std::min<std::size_t>(
                    static_cast<std::size_t>(bandCenterHz_[b] * static_cast<float>(fftSize_) / static_cast<float>(sampleRate_)),
                    numBins_ - 1);
                const float mag    = std::max(magnitude[kNear], 0.0f);
                const float magEps = std::max(mag, kEpsilon);
                const float prevM  = hasPrev ? std::max(prevMagnitude[kNear], 0.0f) : mag;

                sumMag[b]        = mag;
                sumMagSq[b]      = mag * mag;
                sumLogMag[b]     = std::log(magEps);
                sumPosFlux[b]    = std::max(0.0f, mag - prevM);
                sumHarmSq[b]     = hasHPR ? (harmonicMag[kNear] * harmonicMag[kNear]) : (mag * mag);
                sumTotalHPRSq[b] = hasHPR ? (sumHarmSq[b] + percussiveMag[kNear] * percussiveMag[kNear]
                                             + residualMag[kNear] * residualMag[kNear])
                                          : (mag * mag);
                sumWeights[b]    = 1.0f;
            }

            const float invW      = 1.0f / sumWeights[b];
            const float meanMag   = sumMag[b] * invW;
            const float rmsEnergy = std::sqrt(std::max(0.0f, sumMagSq[b] * invW));
            const float geomMean  = std::exp(sumLogMag[b] * invW);

            const float flatness  = (meanMag > kEpsilon)
                                        ? std::clamp(geomMean / (meanMag + kEpsilon), 0.0f, 1.0f)
                                        : 1.0f;
            const float flux      = (sumPosFlux[b] * invW) / (meanMag + 1.0e-4f);
            const float harmRatio = (sumTotalHPRSq[b] > kEpsilon)
                                        ? std::clamp(sumHarmSq[b] / (sumTotalHPRSq[b] + kEpsilon), 0.0f, 1.0f)
                                        : 0.0f;

            if (b < outErbEnergy.size())   { outErbEnergy[b]   = rmsEnergy; }
            if (b < outErbFlatness.size()) { outErbFlatness[b] = flatness; }
            if (b < outErbFlux.size())     { outErbFlux[b]     = std::clamp(flux, 0.0f, 10.0f); }
            if (b < outErbHarmonic.size()) { outErbHarmonic[b] = harmRatio; }
        }
    }

    // ========================================================================
    // Smooth Cosine-Tapered Projection B -> K (Section 1.2)
    // ========================================================================
    /**
     * @brief Projects a single sample's band mask \hat{M}_i(t, b) (size B) to FFT bins M_i^{raw}(t, k) (size K):
     *        M_i^{raw}(t, k) = (1 - w_up(k)) * \hat{M}_i(t, b0(k)) + w_up(k) * \hat{M}_i(t, b1(k))
     *
     * Cost: 2 multiplications and 1 addition per bin (O(K), SIMD-friendly).
     */
    void projectBandsToBins(std::span<const float> bandValues,
                            std::span<float>       outBinValues) const noexcept {
        const std::size_t limitK = std::min(numBins_, outBinValues.size());
        for (std::size_t k = 0; k < limitK; ++k) {
            const std::size_t b0  = lowerBandIdx_[k];
            const std::size_t b1  = upperBandIdx_[k];
            const float       wUp = upsampleWeight_[k];

            const float v0 = (b0 < bandValues.size()) ? bandValues[b0] : 0.0f;
            const float v1 = (b1 < bandValues.size()) ? bandValues[b1] : v0;

            outBinValues[k] = (1.0f - wUp) * v0 + wUp * v1;
        }
        for (std::size_t k = limitK; k < outBinValues.size(); ++k) {
            outBinValues[k] = 0.0f;
        }
    }

    /**
     * @brief Projects all active sample slots in a MaskMatrix2D from ERB bands (B) to FFT bins (K),
     *        strictly preserving the Partition of Unity: \sum_{i=1}^N M_i(t, k) == 1 for all k.
     */
    void projectBandsToBins(MaskMatrix2D& masks, std::size_t activeSlots = kMaxSampleSlots) const noexcept {
        const std::size_t slots = std::clamp<std::size_t>(activeSlots, 1, kMaxSampleSlots);

        // First ensure the input ERB band masks form a valid partition of unity
        for (std::size_t b = 0; b < numBands_; ++b) {
            float sumEnv  = 0.0f;
            float sumFine = 0.0f;
            for (std::size_t i = 0; i < slots; ++i) {
                sumEnv  += std::max(0.0f, masks.erbMaskEnv[i * kMaxERBBands + b]);
                sumFine += std::max(0.0f, masks.erbMaskFine[i * kMaxERBBands + b]);
            }

            const float invEnv  = (sumEnv > 1.0e-12f) ? (1.0f / sumEnv) : 0.0f;
            const float invFine = (sumFine > 1.0e-12f) ? (1.0f / sumFine) : 0.0f;

            for (std::size_t i = 0; i < slots; ++i) {
                masks.erbMaskEnv[i * kMaxERBBands + b] =
                    (sumEnv > 1.0e-12f) ? (std::max(0.0f, masks.erbMaskEnv[i * kMaxERBBands + b]) * invEnv)
                                        : ((i == 0) ? 1.0f : 0.0f);
                masks.erbMaskFine[i * kMaxERBBands + b] =
                    (sumFine > 1.0e-12f) ? (std::max(0.0f, masks.erbMaskFine[i * kMaxERBBands + b]) * invFine)
                                         : ((i == 0) ? 1.0f : 0.0f);
            }
            for (std::size_t i = slots; i < kMaxSampleSlots; ++i) {
                masks.erbMaskEnv[i * kMaxERBBands + b]  = 0.0f;
                masks.erbMaskFine[i * kMaxERBBands + b] = 0.0f;
            }
        }

        // Cosine-tapered linear convex combination automatically preserves \sum_i M_i(t, k) = 1
        for (std::size_t i = 0; i < slots; ++i) {
            projectBandsToBins(masks.erbEnvSlot(i), masks.binEnvSlot(i));
            projectBandsToBins(masks.erbFineSlot(i), masks.binFineSlot(i));
        }
        for (std::size_t i = slots; i < kMaxSampleSlots; ++i) {
            std::fill(masks.binEnvSlot(i).begin(), masks.binEnvSlot(i).end(), 0.0f);
            std::fill(masks.binFineSlot(i).begin(), masks.binFineSlot(i).end(), 0.0f);
        }

        masks.updatePurityAndNormalizeBins(slots);
    }

    // ========================================================================
    // Getters & Inspection Accessors
    // ========================================================================
    [[nodiscard]] std::size_t        fftSize() const noexcept    { return fftSize_; }
    [[nodiscard]] double             sampleRate() const noexcept { return sampleRate_; }
    [[nodiscard]] std::size_t        numBins() const noexcept    { return numBins_; }
    [[nodiscard]] std::size_t        paddedBins() const noexcept { return paddedBins_; }
    [[nodiscard]] std::size_t        numBands() const noexcept   { return numBands_; }
    [[nodiscard]] FrequencyScaleMode scaleMode() const noexcept  { return scaleMode_; }

    [[nodiscard]] std::span<const float>         binFrequenciesHz() const noexcept { return {binFreqHz_.data(), numBins_}; }
    [[nodiscard]] std::span<const float>         bandCenterFrequenciesHz() const noexcept { return {bandCenterHz_.data(), numBands_}; }
    [[nodiscard]] std::span<const std::uint16_t> lowerBandIndices() const noexcept { return {lowerBandIdx_.data(), numBins_}; }
    [[nodiscard]] std::span<const std::uint16_t> upperBandIndices() const noexcept { return {upperBandIdx_.data(), numBins_}; }
    [[nodiscard]] std::span<const float>         upsampleWeights() const noexcept  { return {upsampleWeight_.data(), numBins_}; }

private:
    [[nodiscard]] float hzToScaleUnit(float freqHz) const noexcept {
        switch (scaleMode_) {
            case FrequencyScaleMode::ERB:      return hzToERBRate(freqHz);
            case FrequencyScaleMode::Bark:     return hzToBark(freqHz);
            case FrequencyScaleMode::Harmonic: return hzToHarmonicScale(freqHz, refPitchHz_);
            case FrequencyScaleMode::Linear:   return std::max(0.0f, freqHz);
        }
        return hzToERBRate(freqHz);
    }

    [[nodiscard]] float scaleUnitToHz(float scaleUnit) const noexcept {
        switch (scaleMode_) {
            case FrequencyScaleMode::ERB:      return erbRateToHz(scaleUnit);
            case FrequencyScaleMode::Bark:     return barkToHz(scaleUnit);
            case FrequencyScaleMode::Harmonic: return harmonicScaleToHz(scaleUnit, refPitchHz_);
            case FrequencyScaleMode::Linear:   return std::max(0.0f, scaleUnit);
        }
        return erbRateToHz(scaleUnit);
    }

    std::size_t        fftSize_{kDefaultFFTSize};
    double             sampleRate_{48000.0};
    std::size_t        numBins_{kDefaultNumBins};
    std::size_t        paddedBins_{kDefaultPaddedBins};
    std::size_t        numBands_{kMaxERBBands};
    FrequencyScaleMode scaleMode_{FrequencyScaleMode::ERB};
    float              refPitchHz_{110.0f};

    AlignedVector<float>         binFreqHz_;
    AlignedVector<float>         binContBandCoord_;
    AlignedVector<std::uint16_t> lowerBandIdx_;
    AlignedVector<std::uint16_t> upperBandIdx_;
    AlignedVector<float>         upsampleWeight_;

    std::array<float, kMaxERBBands> bandCenterHz_{};
    std::array<float, kMaxERBBands> bandNormWeight_{};
};

} // namespace spectile::dsp
