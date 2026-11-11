#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace spectile::core {

// ============================================================================
// Architectural Constants (64-Byte Cache-Line & SIMD Alignment)
// ============================================================================
inline constexpr std::size_t kCacheLineBytes       = 64;
inline constexpr std::size_t kSimdFloatLanes       = 16; // 16 x float32 = 64 bytes (AVX-512 / 2x AVX2 / 4x NEON)
inline constexpr std::size_t kMaxSampleSlots       = 8;
inline constexpr std::size_t kMaxERBBands          = 64;
inline constexpr std::size_t kDefaultFFTSize       = 4096;
inline constexpr std::size_t kDefaultHopSize       = 512;
inline constexpr std::size_t kDefaultNumBins       = (kDefaultFFTSize / 2) + 1; // 2049
inline constexpr std::size_t kGenomeDimension      = 24;
inline constexpr std::size_t kNumEvolutionBranches = 6;
inline constexpr std::size_t kGpuDisplayBands      = 512;

// HPR Component Bitflags per Sample Slot
inline constexpr std::uint8_t kHPRFlagHarmonic   = 0x01;
inline constexpr std::uint8_t kHPRFlagPercussive = 0x02;
inline constexpr std::uint8_t kHPRFlagResidual   = 0x04;
inline constexpr std::uint8_t kHPRFlagAll        = kHPRFlagHarmonic | kHPRFlagPercussive | kHPRFlagResidual;

/**
 * @brief Rounds a bin or element count up to the nearest multiple of kSimdFloatLanes (16 floats = 64 bytes).
 *        For K = 2049, returns K_padded = 2064, eliminating scalar tail loops in SIMD kernels.
 */
[[nodiscard]] constexpr std::size_t roundUpToSimdLanes(std::size_t count) noexcept {
    return ((count + kSimdFloatLanes - 1) / kSimdFloatLanes) * kSimdFloatLanes;
}

inline constexpr std::size_t kDefaultPaddedBins = roundUpToSimdLanes(kDefaultNumBins); // 2064

// ============================================================================
// 64-Byte Aligned Allocator for SIMD SoA Buffers
// ============================================================================
template <typename T, std::size_t Alignment = kCacheLineBytes>
class AlignedAllocator {
public:
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");

    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal                        = std::true_type;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    constexpr AlignedAllocator() noexcept = default;
    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        const std::size_t byteCount = n * sizeof(T);
        void* ptr = ::operator new(byteCount, std::align_val_t{Alignment});
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t /*n*/) noexcept {
        if (ptr != nullptr) {
            ::operator delete(ptr, std::align_val_t{Alignment});
        }
    }

    template <typename U, std::size_t OtherAlignment>
    constexpr bool operator==(const AlignedAllocator<U, OtherAlignment>&) const noexcept {
        return Alignment == OtherAlignment;
    }

    template <typename U, std::size_t OtherAlignment>
    constexpr bool operator!=(const AlignedAllocator<U, OtherAlignment>&) const noexcept {
        return Alignment != OtherAlignment;
    }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, kCacheLineBytes>>;

// ============================================================================
// IEEE-754 binary16 (float16) & int16 Quantization Utilities (12 B/bin Storage)
// ============================================================================
[[nodiscard]] inline std::uint16_t floatToHalf(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t  exp  = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t       mant = bits & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<std::uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

[[nodiscard]] inline float halfToFloat(std::uint16_t half) noexcept {
    const std::uint32_t sign = (static_cast<std::uint32_t>(half & 0x8000u)) << 16;
    const std::uint32_t exp  = (half >> 10) & 0x1Fu;
    const std::uint32_t mant = half & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) {
            return std::bit_cast<float>(sign);
        }
        float m = static_cast<float>(mant) * (1.0f / 1024.0f);
        float v = std::ldexp(m, -14);
        return (sign != 0) ? -v : v;
    }
    if (exp == 31) {
        const std::uint32_t outBits = sign | 0x7F800000u | (mant << 13);
        return std::bit_cast<float>(outBits);
    }
    const std::uint32_t outExp  = (exp + 127 - 15) << 23;
    const std::uint32_t outMant = mant << 13;
    return std::bit_cast<float>(sign | outExp | outMant);
}

/**
 * @brief Compact 12-byte/bin storage format described in Section 1.1:
 *        - ln(E_i) in float16 (2 B)
 *        - (F_{H,i}, F_{P,i}, F_{R,i}) in 3 x float16 (6 B)
 *        - Phase phi_i and Instantaneous Frequency omega_i^{IF} quantized in 2 x int16 (4 B)
 */
struct CompactBinRecord {
    std::uint16_t logEnvelopeFp16{0};
    std::uint16_t fineHarmonicFp16{0};
    std::uint16_t finePercussiveFp16{0};
    std::uint16_t fineResidualFp16{0};
    std::int16_t  phaseInt16{0};
    std::int16_t  instFreqInt16{0};
};
static_assert(sizeof(CompactBinRecord) == 12, "CompactBinRecord must occupy exactly 12 bytes per bin");

// ============================================================================
// SpectralFrameSoA — Single STFT Frame in 64-Byte Aligned Structure-of-Arrays
// ============================================================================
struct alignas(kCacheLineBytes) SpectralFrameSoA {
    std::size_t fftSize{kDefaultFFTSize};
    std::size_t numBins{kDefaultNumBins};
    std::size_t paddedBins{kDefaultPaddedBins};

    // Polar & Derivative Slabs
    AlignedVector<float> magnitude;        // |S(t, k)|
    AlignedVector<float> phase;            // \phi(t, k) in [-pi, pi]
    AlignedVector<float> instFreq;         // \omega^{IF}(t, k) in rad/hop
    AlignedVector<float> groupDelay;       // \tau^{GD}(t, k)

    // 3-Way HPR Magnitudes
    AlignedVector<float> harmonicMag;      // H(t, k)
    AlignedVector<float> percussiveMag;    // P(t, k)
    AlignedVector<float> residualMag;      // R(t, k)

    // Source-Filter Decomposition (True Envelope E & Fine Structure F)
    AlignedVector<float> spectralEnvelope; // E(t, k)
    AlignedVector<float> fineStructure;    // F(t, k) in [0, 1.05]
    AlignedVector<float> fineHarmonic;     // F_H(t, k) = H(t, k) / E(t, k)
    AlignedVector<float> finePercussive;   // F_P(t, k) = P(t, k) / E(t, k)
    AlignedVector<float> fineResidual;     // F_R(t, k) = R(t, k) / E(t, k)

    // Log-Magnitude & Time-Frequency Gradients for Causal MS-PGHI
    AlignedVector<float> logMagnitude;     // L(t, k) = ln(|S_out(t, k)| + eps)
    AlignedVector<float> dLogMagDf;        // \partial L / \partial k
    AlignedVector<float> dLogMagDt;        // \partial L / \partial t

    // Cartesian Complex Slabs for SIMD IFFT
    AlignedVector<float> real;             // |S| cos(\phi)
    AlignedVector<float> imag;             // |S| sin(\phi)

    SpectralFrameSoA() {
        resize(kDefaultFFTSize);
    }

    explicit SpectralFrameSoA(std::size_t inFftSize) {
        resize(inFftSize);
    }

    void resize(std::size_t inFftSize) {
        fftSize    = inFftSize;
        numBins    = (inFftSize / 2) + 1;
        paddedBins = roundUpToSimdLanes(numBins);

        magnitude.assign(paddedBins, 0.0f);
        phase.assign(paddedBins, 0.0f);
        instFreq.assign(paddedBins, 0.0f);
        groupDelay.assign(paddedBins, 0.0f);

        harmonicMag.assign(paddedBins, 0.0f);
        percussiveMag.assign(paddedBins, 0.0f);
        residualMag.assign(paddedBins, 0.0f);

        spectralEnvelope.assign(paddedBins, 1.0f);
        fineStructure.assign(paddedBins, 0.0f);
        fineHarmonic.assign(paddedBins, 0.0f);
        finePercussive.assign(paddedBins, 0.0f);
        fineResidual.assign(paddedBins, 0.0f);

        logMagnitude.assign(paddedBins, -18.42068f); // ln(1e-8)
        dLogMagDf.assign(paddedBins, 0.0f);
        dLogMagDt.assign(paddedBins, 0.0f);

        real.assign(paddedBins, 0.0f);
        imag.assign(paddedBins, 0.0f);
    }

    void clear() noexcept {
        std::fill(magnitude.begin(), magnitude.end(), 0.0f);
        std::fill(phase.begin(), phase.end(), 0.0f);
        std::fill(instFreq.begin(), instFreq.end(), 0.0f);
        std::fill(groupDelay.begin(), groupDelay.end(), 0.0f);
        std::fill(harmonicMag.begin(), harmonicMag.end(), 0.0f);
        std::fill(percussiveMag.begin(), percussiveMag.end(), 0.0f);
        std::fill(residualMag.begin(), residualMag.end(), 0.0f);
        std::fill(spectralEnvelope.begin(), spectralEnvelope.end(), 1.0f);
        std::fill(fineStructure.begin(), fineStructure.end(), 0.0f);
        std::fill(fineHarmonic.begin(), fineHarmonic.end(), 0.0f);
        std::fill(finePercussive.begin(), finePercussive.end(), 0.0f);
        std::fill(fineResidual.begin(), fineResidual.end(), 0.0f);
        std::fill(logMagnitude.begin(), logMagnitude.end(), -18.42068f);
        std::fill(dLogMagDf.begin(), dLogMagDf.end(), 0.0f);
        std::fill(dLogMagDt.begin(), dLogMagDt.end(), 0.0f);
        std::fill(real.begin(), real.end(), 0.0f);
        std::fill(imag.begin(), imag.end(), 0.0f);
    }

    [[nodiscard]] std::span<float>       magnitudeSpan() noexcept       { return {magnitude.data(), paddedBins}; }
    [[nodiscard]] std::span<const float> magnitudeSpan() const noexcept { return {magnitude.data(), paddedBins}; }
    [[nodiscard]] std::span<float>       phaseSpan() noexcept           { return {phase.data(), paddedBins}; }
    [[nodiscard]] std::span<const float> phaseSpan() const noexcept     { return {phase.data(), paddedBins}; }
};

// ============================================================================
// SpectralSampleData — Immutable Pre-Analyzed Multi-Frame Sample Dataset
// ============================================================================
struct alignas(kCacheLineBytes) SpectralSampleData {
    std::uint32_t sampleId{0};
    std::string   sampleName;
    std::string   relativePath;
    std::string   sha256Hash;

    double        sampleRate{48000.0};
    std::size_t   fftSize{kDefaultFFTSize};
    std::size_t   hopSize{kDefaultHopSize};
    std::size_t   numFrames{0};
    std::size_t   numBins{kDefaultNumBins};
    std::size_t   paddedBins{kDefaultPaddedBins};
    std::size_t   numERBBands{kMaxERBBands};

    float         fundamentalHz{220.0f}; // Detected pitch f_{0,i}
    float         rmsEnergy{1.0f};

    // Flattened 2D SoA Slabs: [frameIndex * paddedBins + binIndex]
    AlignedVector<float> magnitude;        // |S_i(t, k)|
    AlignedVector<float> phase;            // \phi_i(t, k)
    AlignedVector<float> instFreq;         // \omega_i^{IF}(t, k)
    AlignedVector<float> groupDelay;       // \tau_i^{GD}(t, k)

    // 3-Way HPR Decomposition
    AlignedVector<float> harmonic;         // H_i(t, k)
    AlignedVector<float> percussive;       // P_i(t, k)
    AlignedVector<float> residual;         // R_i(t, k)

    // Cepstral True Envelope & Fine Structure (E_i * F_i = |S_i|)
    AlignedVector<float> envelope;         // E_i(t, k)
    AlignedVector<float> fineStructure;    // F_i(t, k)
    AlignedVector<float> fineHarmonic;     // F_{H,i}(t, k)
    AlignedVector<float> finePercussive;   // F_{P,i}(t, k)
    AlignedVector<float> fineResidual;     // F_{R,i}(t, k)

    // Per-Frame ERB Band Descriptors: [frameIndex * kMaxERBBands + bandIndex]
    AlignedVector<float> erbEnergy;        // \hat{E}_i(t, b)
    AlignedVector<float> erbFlatness;      // Spectral Flatness (Wiener entropy) in [0, 1]
    AlignedVector<float> erbFlux;          // Rectified Half-Wave Spectral Flux
    AlignedVector<float> erbHarmonicity;   // Harmonic ratio H / (H + P + R) per ERB band

    // Optional Compact 12-Byte/Bin Storage Buffer
    AlignedVector<CompactBinRecord> compactStorage;

    void allocate(std::size_t inNumFrames,
                  std::size_t inFftSize = kDefaultFFTSize,
                  std::size_t inHopSize = kDefaultHopSize,
                  double      inSampleRate = 48000.0,
                  std::size_t inNumERBBands = kMaxERBBands) {
        sampleRate  = inSampleRate;
        fftSize     = inFftSize;
        hopSize     = inHopSize;
        numFrames   = inNumFrames;
        numBins     = (inFftSize / 2) + 1;
        paddedBins  = roundUpToSimdLanes(numBins);
        numERBBands = std::min(inNumERBBands, kMaxERBBands);

        const std::size_t totalBinElems  = numFrames * paddedBins;
        const std::size_t totalBandElems = numFrames * kMaxERBBands;

        magnitude.assign(totalBinElems, 0.0f);
        phase.assign(totalBinElems, 0.0f);
        instFreq.assign(totalBinElems, 0.0f);
        groupDelay.assign(totalBinElems, 0.0f);

        harmonic.assign(totalBinElems, 0.0f);
        percussive.assign(totalBinElems, 0.0f);
        residual.assign(totalBinElems, 0.0f);

        envelope.assign(totalBinElems, 1.0f);
        fineStructure.assign(totalBinElems, 0.0f);
        fineHarmonic.assign(totalBinElems, 0.0f);
        finePercussive.assign(totalBinElems, 0.0f);
        fineResidual.assign(totalBinElems, 0.0f);

        erbEnergy.assign(totalBandElems, 0.0f);
        erbFlatness.assign(totalBandElems, 0.0f);
        erbFlux.assign(totalBandElems, 0.0f);
        erbHarmonicity.assign(totalBandElems, 0.0f);
    }

    // Read-Only Frame Accessors (std::span<const float>)
    [[nodiscard]] std::span<const float> frameMagnitude(std::size_t t) const noexcept {
        return {magnitude.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> framePhase(std::size_t t) const noexcept {
        return {phase.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameInstFreq(std::size_t t) const noexcept {
        return {instFreq.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameGroupDelay(std::size_t t) const noexcept {
        return {groupDelay.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameHarmonic(std::size_t t) const noexcept {
        return {harmonic.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> framePercussive(std::size_t t) const noexcept {
        return {percussive.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameResidual(std::size_t t) const noexcept {
        return {residual.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameEnvelope(std::size_t t) const noexcept {
        return {envelope.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameFineStructure(std::size_t t) const noexcept {
        return {fineStructure.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameFineHarmonic(std::size_t t) const noexcept {
        return {fineHarmonic.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameFinePercussive(std::size_t t) const noexcept {
        return {finePercussive.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> frameFineResidual(std::size_t t) const noexcept {
        return {fineResidual.data() + clampFrame(t) * paddedBins, paddedBins};
    }

    // Read-Only ERB Band Accessors (std::span<const float>)
    [[nodiscard]] std::span<const float> frameERBEnergy(std::size_t t) const noexcept {
        return {erbEnergy.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<const float> frameERBFlatness(std::size_t t) const noexcept {
        return {erbFlatness.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<const float> frameERBFlux(std::size_t t) const noexcept {
        return {erbFlux.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<const float> frameERBHarmonicity(std::size_t t) const noexcept {
        return {erbHarmonicity.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }

    // Mutable Frame Accessors for Offline Pre-Analysis Worker
    [[nodiscard]] std::span<float> mutableFrameMagnitude(std::size_t t) noexcept {
        return {magnitude.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFramePhase(std::size_t t) noexcept {
        return {phase.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameInstFreq(std::size_t t) noexcept {
        return {instFreq.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameGroupDelay(std::size_t t) noexcept {
        return {groupDelay.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameHarmonic(std::size_t t) noexcept {
        return {harmonic.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFramePercussive(std::size_t t) noexcept {
        return {percussive.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameResidual(std::size_t t) noexcept {
        return {residual.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameEnvelope(std::size_t t) noexcept {
        return {envelope.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameFineStructure(std::size_t t) noexcept {
        return {fineStructure.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameFineHarmonic(std::size_t t) noexcept {
        return {fineHarmonic.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameFinePercussive(std::size_t t) noexcept {
        return {finePercussive.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameFineResidual(std::size_t t) noexcept {
        return {fineResidual.data() + clampFrame(t) * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> mutableFrameERBEnergy(std::size_t t) noexcept {
        return {erbEnergy.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<float> mutableFrameERBFlatness(std::size_t t) noexcept {
        return {erbFlatness.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<float> mutableFrameERBFlux(std::size_t t) noexcept {
        return {erbFlux.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<float> mutableFrameERBHarmonicity(std::size_t t) noexcept {
        return {erbHarmonicity.data() + clampFrame(t) * kMaxERBBands, numERBBands};
    }

    /**
     * @brief Catmull-Rom / Hermite cubic interpolation with anti-aliasing zeroing above Nyquist
     *        for Formant-Preserving Polyphonic Synthesis (Section 1.4.2).
     */
    [[nodiscard]] static float sampleHermiteAntiAliased(std::span<const float> slab,
                                                        float                  fractionalBin,
                                                        std::size_t            validNumBins) noexcept {
        if (fractionalBin < 0.0f || fractionalBin >= static_cast<float>(validNumBins - 1)) {
            return 0.0f; // Anti-aliasing cutoff beyond Nyquist
        }
        const auto  k1   = static_cast<std::ptrdiff_t>(fractionalBin);
        const float frac = fractionalBin - static_cast<float>(k1);

        const std::ptrdiff_t maxIdx = static_cast<std::ptrdiff_t>(validNumBins) - 1;
        const std::ptrdiff_t k0     = std::max<std::ptrdiff_t>(0, k1 - 1);
        const std::ptrdiff_t k2     = std::min<std::ptrdiff_t>(maxIdx, k1 + 1);
        const std::ptrdiff_t k3     = std::min<std::ptrdiff_t>(maxIdx, k1 + 2);

        const float y0 = slab[static_cast<std::size_t>(k0)];
        const float y1 = slab[static_cast<std::size_t>(k1)];
        const float y2 = slab[static_cast<std::size_t>(k2)];
        const float y3 = slab[static_cast<std::size_t>(k3)];

        const float c0 = y1;
        const float c1 = 0.5f * (y2 - y0);
        const float c2 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);

        const float val = ((c3 * frac + c2) * frac + c1) * frac + c0;
        return std::max(0.0f, val);
    }

private:
    [[nodiscard]] std::size_t clampFrame(std::size_t t) const noexcept {
        if (numFrames == 0) {
            return 0;
        }
        return (t < numFrames) ? t : (numFrames - 1);
    }
};

// ============================================================================
// MaskMatrix2D — Dual Envelope/Fine-Structure Spectral Mask State (SoA)
// ============================================================================
struct alignas(kCacheLineBytes) MaskMatrix2D {
    std::size_t numSlots{kMaxSampleSlots};
    std::size_t numERBBands{kMaxERBBands};
    std::size_t numBins{kDefaultNumBins};
    std::size_t paddedBins{kDefaultPaddedBins};

    // ERB Band-Level Masks: [slot * kMaxERBBands + band]
    AlignedVector<float> erbMaskEnv;       // \hat{M}^{env}_i(t, b)
    AlignedVector<float> erbMaskFine;      // \hat{M}^{fine}_j(t, b)

    // Full FFT Bin-Level Masks: [slot * paddedBins + bin]
    AlignedVector<float> binMaskEnv;       // M^{env}_i(t, k)
    AlignedVector<float> binMaskFine;      // M^{fine}_j(t, k)

    // Per-Bin Purity & Dominant Sample Tracking for MS-PGHI Phase Glue
    AlignedVector<float>         tilePurity;       // c_{purity}(t, k) = max_i M_i(t, k) in [1/N, 1]
    AlignedVector<std::uint8_t>  dominantSampleId; // i^*(t, k) = argmax_i M_i(t, k)

    MaskMatrix2D() {
        allocate(kDefaultNumBins, kMaxERBBands);
    }

    explicit MaskMatrix2D(std::size_t inNumBins, std::size_t inNumERBBands = kMaxERBBands) {
        allocate(inNumBins, inNumERBBands);
    }

    void allocate(std::size_t inNumBins, std::size_t inNumERBBands = kMaxERBBands) {
        numSlots    = kMaxSampleSlots;
        numERBBands = std::min(inNumERBBands, kMaxERBBands);
        numBins     = inNumBins;
        paddedBins  = roundUpToSimdLanes(inNumBins);

        erbMaskEnv.assign(numSlots * kMaxERBBands, 0.0f);
        erbMaskFine.assign(numSlots * kMaxERBBands, 0.0f);
        binMaskEnv.assign(numSlots * paddedBins, 0.0f);
        binMaskFine.assign(numSlots * paddedBins, 0.0f);
        tilePurity.assign(paddedBins, 1.0f);
        dominantSampleId.assign(paddedBins, 0);

        // Initialize slot 0 to 1.0 so partition of unity holds out-of-the-box
        std::fill_n(erbMaskEnv.data(), kMaxERBBands, 1.0f);
        std::fill_n(erbMaskFine.data(), kMaxERBBands, 1.0f);
        std::fill_n(binMaskEnv.data(), paddedBins, 1.0f);
        std::fill_n(binMaskFine.data(), paddedBins, 1.0f);
    }

    [[nodiscard]] std::span<float> erbEnvSlot(std::size_t slot) noexcept {
        return {erbMaskEnv.data() + slot * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<const float> erbEnvSlot(std::size_t slot) const noexcept {
        return {erbMaskEnv.data() + slot * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<float> erbFineSlot(std::size_t slot) noexcept {
        return {erbMaskFine.data() + slot * kMaxERBBands, numERBBands};
    }
    [[nodiscard]] std::span<const float> erbFineSlot(std::size_t slot) const noexcept {
        return {erbMaskFine.data() + slot * kMaxERBBands, numERBBands};
    }

    [[nodiscard]] std::span<float> binEnvSlot(std::size_t slot) noexcept {
        return {binMaskEnv.data() + slot * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> binEnvSlot(std::size_t slot) const noexcept {
        return {binMaskEnv.data() + slot * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<float> binFineSlot(std::size_t slot) noexcept {
        return {binMaskFine.data() + slot * paddedBins, paddedBins};
    }
    [[nodiscard]] std::span<const float> binFineSlot(std::size_t slot) const noexcept {
        return {binMaskFine.data() + slot * paddedBins, paddedBins};
    }

    /**
     * @brief Enforces exact Partition of Unity (\sum_{i=1}^N M_i(t, k) = 1)
     *        and updates tilePurity and dominantSampleId per bin.
     */
    void updatePurityAndNormalizeBins(std::size_t activeSlots = kMaxSampleSlots) noexcept {
        const std::size_t slots = std::clamp<std::size_t>(activeSlots, 1, kMaxSampleSlots);
        for (std::size_t k = 0; k < numBins; ++k) {
            float sumEnv  = 0.0f;
            float sumFine = 0.0f;
            for (std::size_t i = 0; i < slots; ++i) {
                sumEnv  += binMaskEnv[i * paddedBins + k];
                sumFine += binMaskFine[i * paddedBins + k];
            }

            const float invSumEnv  = (sumEnv > 1.0e-12f) ? (1.0f / sumEnv) : 0.0f;
            const float invSumFine = (sumFine > 1.0e-12f) ? (1.0f / sumFine) : 0.0f;

            float        maxWeight = -1.0f;
            std::uint8_t bestSlot  = 0;

            for (std::size_t i = 0; i < slots; ++i) {
                float wEnv  = (sumEnv > 1.0e-12f)
                                  ? (binMaskEnv[i * paddedBins + k] * invSumEnv)
                                  : ((i == 0) ? 1.0f : 0.0f);
                float wFine = (sumFine > 1.0e-12f)
                                  ? (binMaskFine[i * paddedBins + k] * invSumFine)
                                  : ((i == 0) ? 1.0f : 0.0f);

                binMaskEnv[i * paddedBins + k]  = wEnv;
                binMaskFine[i * paddedBins + k] = wFine;

                if (wFine > maxWeight) {
                    maxWeight = wFine;
                    bestSlot  = static_cast<std::uint8_t>(i);
                }
            }
            for (std::size_t i = slots; i < kMaxSampleSlots; ++i) {
                binMaskEnv[i * paddedBins + k]  = 0.0f;
                binMaskFine[i * paddedBins + k] = 0.0f;
            }

            tilePurity[k]       = std::clamp(maxWeight, 0.0f, 1.0f);
            dominantSampleId[k] = bestSlot;
        }
    }
};

// ============================================================================
// GenomeState — Trivially Copyable 24-Dimensional Spectral Genome & Macro State
// ============================================================================
struct alignas(kCacheLineBytes) GenomeState {
    std::array<float, kGenomeDimension>  genes{};
    std::array<float, kMaxSampleSlots>   slotWeights{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::array<std::uint8_t, kMaxSampleSlots> slotLocked{0, 0, 0, 0, 0, 0, 0, 0};
    std::array<std::uint8_t, kMaxSampleSlots> slotHPRFlags{
        kHPRFlagAll, kHPRFlagAll, kHPRFlagAll, kHPRFlagAll,
        kHPRFlagAll, kHPRFlagAll, kHPRFlagAll, kHPRFlagAll
    };

    std::uint32_t archetypeId{0};           // 0..5 (A1..A6 musical archetypes)
    float         branchRadius{0.0f};       // \rho \in [0, 1] radial scrubbing distance
    float         branchAngle{0.0f};        // \vartheta \in [0, 2\pi) polar branch angle

    // 3 Macro Controls (Level 1 Simple View)
    float         macroTileSize{0.5f};      // \Delta t \times \Delta f scale
    float         macroRuleOrderVsMutation{0.5f};
    float         macroPhaseAndSpectralGlue{0.75f};

    // Source-Filter & Cross-Synthesis Controls (Level 2 Surgical View)
    float         crossSynthesisAmount{0.0f}; // \chi_{cross} \in [0, 1]
    float         envelopeExponent{0.85f};    // \gamma \in [0, 1]
    float         softmaxTemperature{0.18f};  // \tau \in [0.03, 1.50]
    float         sparsityGate{0.10f};        // \eta_{sparse} \in [0, 0.95]

    std::uint64_t generationCounter{1};
};
static_assert(std::is_trivially_copyable_v<GenomeState>,
              "GenomeState must be trivially copyable for lock-free atomic/triple-buffer exchange");

// ============================================================================
// SpectralColumnPacket — Wait-Free SPSC Telemetry Packet for 60/120 FPS GPU UI
// ============================================================================
/**
 * @brief Compact GPU column payload (~6.65 KB per hop as specified in Section 4.2):
 *        512 logarithmic display bands x 13 bytes:
 *        - magDbFp16 + transientFluxFp16 (4 B, uploaded to GL_RG16F)
 *        - maskWeights[8] quantized to uint8 (8 B, uploaded to two GL_RGBA8 textures)
 *        - dominantSlotId uint8 (1 B, uploaded to GL_R8UI for 4-neighbor Voronoi neon edges)
 */
#pragma pack(push, 1)
struct GpuBandTexel13B {
    std::uint16_t magDbFp16{0};
    std::uint16_t transientFluxFp16{0};
    std::array<std::uint8_t, kMaxSampleSlots> maskWeights{255, 0, 0, 0, 0, 0, 0, 0};
    std::uint8_t  dominantSlotId{0};
};
#pragma pack(pop)
static_assert(sizeof(GpuBandTexel13B) == 13, "Each GPU display band texel must be exactly 13 bytes");

struct alignas(kCacheLineBytes) SpectralColumnPacket {
    std::uint64_t hopIndex{0};
    float         playheadNormalized{0.0f};
    float         meanTilePurity{1.0f};
    std::array<float, 4> bandLockFrequenciesHz{150.0f, 600.0f, 2400.0f, 8000.0f};
    std::array<GpuBandTexel13B, kGpuDisplayBands> bands{};

    /**
     * @brief Populates the 512 logarithmic GPU display bands from the current STFT frame and MaskMatrix2D.
     */
    void packFromFrame(const SpectralFrameSoA& frame,
                       const MaskMatrix2D&     masks,
                       double                  sampleRate,
                       std::uint64_t           currentHopIndex,
                       float                   currentPlayheadNorm = 0.0f) noexcept {
        hopIndex           = currentHopIndex;
        playheadNormalized = std::clamp(currentPlayheadNorm, 0.0f, 1.0f);

        const float minFreqHz = 20.0f;
        const float maxFreqHz = std::max(minFreqHz + 100.0f, static_cast<float>(sampleRate * 0.5));
        const float logRatio  = std::log(maxFreqHz / minFreqHz);
        const float binHz     = static_cast<float>(sampleRate) / static_cast<float>(frame.fftSize);

        float purityAccum = 0.0f;

        for (std::size_t b = 0; b < kGpuDisplayBands; ++b) {
            const float normY  = static_cast<float>(b) / static_cast<float>(kGpuDisplayBands - 1);
            const float freqHz = minFreqHz * std::exp(normY * logRatio);
            const std::size_t k = std::clamp<std::size_t>(
                static_cast<std::size_t>(freqHz / binHz + 0.5f), 0, frame.numBins - 1);

            const float mag   = std::max(frame.magnitude[k], 1.0e-6f);
            const float magDb = 20.0f * std::log10(mag);
            const float flux  = std::clamp(std::max(0.0f, frame.dLogMagDt[k]) * 0.25f, 0.0f, 1.0f);

            GpuBandTexel13B& texel  = bands[b];
            texel.magDbFp16         = floatToHalf(magDb);
            texel.transientFluxFp16 = floatToHalf(flux);
            texel.dominantSlotId    = masks.dominantSampleId[k];

            for (std::size_t s = 0; s < kMaxSampleSlots; ++s) {
                const float w = std::clamp(masks.binMaskFine[s * masks.paddedBins + k], 0.0f, 1.0f);
                texel.maskWeights[s] = static_cast<std::uint8_t>(w * 255.0f + 0.5f);
            }

            purityAccum += masks.tilePurity[k];
        }

        meanTilePurity = purityAccum / static_cast<float>(kGpuDisplayBands);
    }
};
static_assert(std::is_trivially_copyable_v<SpectralColumnPacket>,
              "SpectralColumnPacket must be trivially copyable for lock-free SPSC ring-buffer transport");

} // namespace spectile::core

namespace spectile::dsp {
// Bring core SoA types and constants into spectile::dsp for seamless cross-namespace usage
using spectile::core::AlignedAllocator;
using spectile::core::AlignedVector;
using spectile::core::CompactBinRecord;
using spectile::core::GenomeState;
using spectile::core::GpuBandTexel13B;
using spectile::core::kCacheLineBytes;
using spectile::core::kDefaultFFTSize;
using spectile::core::kDefaultHopSize;
using spectile::core::kDefaultNumBins;
using spectile::core::kDefaultPaddedBins;
using spectile::core::kGenomeDimension;
using spectile::core::kGpuDisplayBands;
using spectile::core::kHPRFlagAll;
using spectile::core::kHPRFlagHarmonic;
using spectile::core::kHPRFlagPercussive;
using spectile::core::kHPRFlagResidual;
using spectile::core::kMaxERBBands;
using spectile::core::kMaxSampleSlots;
using spectile::core::kNumEvolutionBranches;
using spectile::core::kSimdFloatLanes;
using spectile::core::MaskMatrix2D;
using spectile::core::roundUpToSimdLanes;
using spectile::core::SpectralColumnPacket;
using spectile::core::SpectralFrameSoA;
using spectile::core::SpectralSampleData;
} // namespace spectile::dsp
