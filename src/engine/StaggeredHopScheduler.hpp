#pragma once
// ==============================================================================
// SpecTile — src/engine/StaggeredHopScheduler.hpp
// Settore 3: Real-Time Staggered Sub-Hop Spectral Pipeline (4 x 128 Micro-Stages)
// ==============================================================================
// Eliminates periodic STFT CPU spikes by partitioning each R_a = 512 hop into
// 4 deterministic micro-stages of 128 samples:
//   - Stage 0 (0..127):   SIMD Source-Filter + HPR Magnitude Recombination
//   - Stage 1 (128..255): Log-Magnitude L(t,k) & Cauchy-Riemann Gradients
//   - Stage 2 (256..383): Static Binary Max-Heap Causal MS-PGHI + Phase Glue
//   - Stage 3 (384..511): SIMD Polar-to-Cartesian + IFFT + Circular WOLA
// ==============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if __has_include(<xsimd/xsimd.hpp>)
    #include <xsimd/xsimd.hpp>
    #define SPECTILE_HAS_XSIMD 1
#else
    #define SPECTILE_HAS_XSIMD 0
#endif

namespace spectile {

// ==============================================================================
// 1. Global Compile-Time Constants & SIMD Alignment Parameters
// ==============================================================================
inline constexpr std::size_t kCacheLineBytes     = 64;
inline constexpr std::size_t kMaxSources         = 8;
inline constexpr std::size_t kDefaultFftSize     = 4096;
inline constexpr std::size_t kDefaultHopSize     = 512;
inline constexpr std::size_t kNumMicroStages     = 4;
inline constexpr std::size_t kMicroStageSamples  = kDefaultHopSize / kNumMicroStages; // 128
inline constexpr std::size_t kNumBins            = (kDefaultFftSize / 2) + 1;         // 2049
inline constexpr std::size_t kNumBinsPadded      = 2064; // Multiple of 16 floats (64B aligned)
inline constexpr std::size_t kNumErbBands        = 64;
inline constexpr std::size_t kGenomeDim          = 24;
inline constexpr std::size_t kNumBranches        = 6;
inline constexpr std::size_t kGpuDisplayBands    = 512;
inline constexpr float       kPi                 = 3.14159265358979323846f;
inline constexpr float       kTwoPi              = 6.28318530717958647692f;
inline constexpr float       kEps                = 1.0e-7f;
inline constexpr float       kLogMagFloor        = 1.0e-4f; // -80 dB floor for True Envelope / PGHI
inline constexpr float       kPghiHeapTolRatio   = 1.0e-3f; // -60 dB below frame peak

// ==============================================================================
// 2. IEEE-754 binary16 (float16) & Phase int16 Quantization Utilities (12B/bin)
// ==============================================================================
[[nodiscard]] inline uint16_t fp32ToFp16(float value) noexcept {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t       mant = bits & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

[[nodiscard]] inline float fp16ToFp32(uint16_t half) noexcept {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t exp  = (half >> 10) & 0x1Fu;
    const uint32_t mant = half & 0x03FFu;

    if (exp == 0) {
        if (mant == 0) {
            return std::bit_cast<float>(sign);
        }
        float val = static_cast<float>(mant) * (1.0f / 16777216.0f);
        return (sign != 0) ? -val : val;
    }
    if (exp == 31) {
        return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13));
    }
    const uint32_t bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    return std::bit_cast<float>(bits);
}

[[nodiscard]] inline int16_t quantizePhaseInt16(float radians) noexcept {
    float wrapped = std::remainder(radians, kTwoPi);
    float norm    = std::clamp(wrapped / kPi, -1.0f, 1.0f);
    return static_cast<int16_t>( std::lrintf(norm * 32767.0f) );
}

[[nodiscard]] inline float dequantizePhaseInt16(int16_t q) noexcept {
    return static_cast<float>(q) * (kPi / 32767.0f);
}

[[nodiscard]] inline float wrapPhaseRadians(float radians) noexcept {
    return radians - kTwoPi * std::floor((radians + kPi) * (1.0f / kTwoPi));
}

// ==============================================================================
// 3. Core Memory Contracts (SoA Aligned 64-Byte Structures & Lock-Free Buffers)
// ==============================================================================
#ifndef SPECTILE_CORE_DATA_CONTRACTS_DEFINED
#define SPECTILE_CORE_DATA_CONTRACTS_DEFINED

// Compact 12-Byte/bin storage format for RAM efficiency (~22.18 MB per 10s sample)
struct PackedSpectralBin12B {
    uint16_t logEnvFp16;   // ln E_i(t,k) in IEEE-754 binary16 (2 B)
    uint16_t fineHFp16;    // F_{H,i}(t,k) in [0, 1.05] (2 B)
    uint16_t finePFp16;    // F_{P,i}(t,k) in [0, 1.05] (2 B)
    uint16_t fineRFp16;    // F_{R,i}(t,k) in [0, 1.05] (2 B)
    int16_t  phaseQ15;     // phi_i(t,k) quantized on [-pi, pi] (2 B)
    int16_t  instFreqQ15;  // omega_i^IF(t,k) quantized on [-pi, pi] (2 B)
};
static_assert(sizeof(PackedSpectralBin12B) == 12, "PackedSpectralBin12B must be exactly 12 bytes");

// Structure-of-Arrays (SoA) 64-byte aligned frame for L1-cache SIMD FMA processing
struct alignas(kCacheLineBytes) SpectralFrameSoA {
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> env{};        // E_i(t, k) True Envelope
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> fineH{};      // F_{H,i}(t, k) Harmonic fine structure
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> fineP{};      // F_{P,i}(t, k) Percussive fine structure
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> fineR{};      // F_{R,i}(t, k) Residual fine structure
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> phase{};      // phi_i(t, k) native phase
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> instFreq{};   // omega_i^IF(t, k) instantaneous freq
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> groupDelay{}; // tau_i^GD(t, k) local group delay

    // Psychoacoustic ERB band descriptors (B = 64)
    alignas(kCacheLineBytes) std::array<float, kNumErbBands> erbEnergy{};
    alignas(kCacheLineBytes) std::array<float, kNumErbBands> erbFlatness{};
    alignas(kCacheLineBytes) std::array<float, kNumErbBands> erbFlux{};

    float rmsEnergy    = 0.0f;
    float detectedF0Hz = 220.0f;

    void clear() noexcept {
        env.fill(kLogMagFloor);
        fineH.fill(0.0f);
        fineP.fill(0.0f);
        fineR.fill(0.0f);
        phase.fill(0.0f);
        instFreq.fill(0.0f);
        groupDelay.fill(0.0f);
        erbEnergy.fill(0.0f);
        erbFlatness.fill(0.5f);
        erbFlux.fill(0.0f);
        rmsEnergy    = 0.0f;
        detectedF0Hz = 220.0f;
    }
};

// Immutable Pre-Analyzed Sample Data published via Lock-Free RCU
struct SpectralSampleData {
    std::size_t                     slotIndex       = 0;
    double                          sampleRate      = 48000.0;
    std::size_t                     fftSize         = kDefaultFftSize;
    std::size_t                     hopSize         = kDefaultHopSize;
    std::size_t                     numFrames       = 0;
    std::size_t                     numSamples      = 0;
    float                           referenceF0Hz   = 220.0f;
    float                           durationSeconds = 0.0f;
    std::string                     filePath;
    std::string                     fileName;
    std::string                     sha256Hash;
    std::vector<float>              rawAudioMono;
    std::vector<SpectralFrameSoA>   frames;
    std::vector<PackedSpectralBin12B> compactStorage; // Optional 12B/bin storage

    [[nodiscard]] bool isValid() const noexcept {
        return !frames.empty() && numFrames == frames.size();
    }

    // Read a time frame with wrap-around looping in [loopStart, loopEnd]
    [[nodiscard]] const SpectralFrameSoA& getFrameWrapped(
        std::size_t frameIdx,
        float loopStartNorm = 0.0f,
        float loopEndNorm   = 1.0f) const noexcept
    {
        if (frames.empty()) {
            static const SpectralFrameSoA kEmptyFrame{};
            return kEmptyFrame;
        }
        const std::size_t total = frames.size();
        const std::size_t start = std::min<std::size_t>(
            static_cast<std::size_t>(std::clamp(loopStartNorm, 0.0f, 0.99f) * static_cast<float>(total)),
            total - 1);
        const std::size_t end = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::clamp(loopEndNorm, 0.01f, 1.0f) * static_cast<float>(total)),
            start + 1,
            total);
        const std::size_t span = std::max<std::size_t>(1, end - start);
        const std::size_t idx  = start + ((frameIdx >= start ? (frameIdx - start) : frameIdx) % span);
        return frames[std::min(idx, total - 1)];
    }
};

// Dual Source-Filter Mask Matrix M_i^env(t,k) and M_j^fine(t,k) aligned to 64 bytes
struct alignas(kCacheLineBytes) MaskMatrix2D {
    alignas(kCacheLineBytes) std::array<std::array<float, kNumBinsPadded>, kMaxSources> envMask{};
    alignas(kCacheLineBytes) std::array<std::array<float, kNumBinsPadded>, kMaxSources> fineMask{};
    uint64_t frameSequenceId = 0;

    void setUniform(std::size_t activeSources = kMaxSources) noexcept {
        const std::size_t count = std::clamp<std::size_t>(activeSources, 1, kMaxSources);
        const float weight = 1.0f / static_cast<float>(count);
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const float val = (i < count) ? weight : 0.0f;
            envMask[i].fill(val);
            fineMask[i].fill(val);
        }
    }
};

// Wait-Free Triple Buffer for MaskMatrix2D / Real-Time State Handoff
template <typename T>
class LockFreeTripleBuffer {
public:
    LockFreeTripleBuffer() : buffers_(3) {}

    T& writeBuffer() noexcept {
        return buffers_[writeIdx_];
    }

    void publishWrite() noexcept {
        const uint8_t nextShared = writeIdx_ | kDirtyBit;
        const uint8_t prevShared = sharedState_.exchange(nextShared, std::memory_order_acq_rel);
        writeIdx_ = prevShared & kIndexMask;
    }

    void publish(const T& value) noexcept {
        buffers_[writeIdx_] = value;
        publishWrite();
    }

    [[nodiscard]] const T& readLatest() noexcept {
        const uint8_t currentShared = sharedState_.load(std::memory_order_acquire);
        if ((currentShared & kDirtyBit) != 0) {
            const uint8_t prevShared = sharedState_.exchange(readIdx_, std::memory_order_acq_rel);
            readIdx_ = prevShared & kIndexMask;
        }
        return buffers_[readIdx_];
    }

private:
    static constexpr uint8_t kIndexMask = 0x03u;
    static constexpr uint8_t kDirtyBit  = 0x80u;

    std::vector<T>       buffers_;
    uint8_t              writeIdx_    = 0;
    uint8_t              readIdx_     = 1;
    std::atomic<uint8_t> sharedState_{2};
};

// Wait-Free SPSC Ring Buffer for GPU Spectrogram Streaming (6.65 KB/hop)
template <typename T, std::size_t CapacityPow2 = 64>
class SpscRingBuffer {
    static_assert((CapacityPow2 & (CapacityPow2 - 1)) == 0, "CapacityPow2 must be a power of 2");
public:
    SpscRingBuffer() : storage_(CapacityPow2) {}

    bool tryPush(const T& item) noexcept {
        const std::size_t w = writePos_.load(std::memory_order_relaxed);
        const std::size_t r = readPos_.load(std::memory_order_acquire);
        if (w - r >= CapacityPow2) {
            return false; // Full — drop frame rather than block audio thread
        }
        storage_[w & (CapacityPow2 - 1)] = item;
        writePos_.store(w + 1, std::memory_order_release);
        return true;
    }

    bool tryPop(T& outItem) noexcept {
        const std::size_t r = readPos_.load(std::memory_order_relaxed);
        const std::size_t w = writePos_.load(std::memory_order_acquire);
        if (r == w) {
            return false;
        }
        outItem = storage_[r & (CapacityPow2 - 1)];
        readPos_.store(r + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t sizeApprox() const noexcept {
        const std::size_t w = writePos_.load(std::memory_order_acquire);
        const std::size_t r = readPos_.load(std::memory_order_acquire);
        return w - r;
    }

private:
    alignas(kCacheLineBytes) std::atomic<std::size_t> writePos_{0};
    alignas(kCacheLineBytes) std::atomic<std::size_t> readPos_{0};
    std::vector<T>                                    storage_;
};

// Compact 6.65 KB GPU Telemetry Column Packet sent every STFT hop (~5.8-10.6 ms)
#pragma pack(push, 1)
struct GpuColumnPacket {
    // 512 bands x 2 fp16 (GL_RG16F: Magnitude dB + Spectral Flux Transience) = 2048 B
    std::array<std::array<uint16_t, 2>, kGpuDisplayBands> magDbAndTransience{};
    // 512 bands x 8 uint8 (2x GL_RGBA8: Mask weights M_1..M_8 in [0..255]) = 4096 B
    std::array<std::array<uint8_t, kMaxSources>, kGpuDisplayBands> maskWeightsRgba{};
    // 512 bands x 1 uint8 (GL_R8UI: Dominant sample index argmax_i M_i) = 512 B
    std::array<uint8_t, kGpuDisplayBands> dominantSampleId{};
};
#pragma pack(pop)
static_assert(sizeof(GpuColumnPacket) == 6656, "GpuColumnPacket must be exactly 6656 bytes (6.65 KB)");

#endif // SPECTILE_CORE_DATA_CONTRACTS_DEFINED

namespace engine {

// ==============================================================================
// 4. Phase Reconstruction Mode & Per-Source HPR / Synthesis Parameters
// ==============================================================================
enum class PhaseMode : uint8_t {
    RawSource           = 0, // Unmodified phase from dominant source i*(k)
    IdentityPhaseLocked = 1, // Laroche & Dolson (1999) horizontal/vertical phase locking
    CausalMS_PGHI       = 2, // Prusa et al. (2017) Multi-Source PGHI + Phase Glue
    SpectralFreeze      = 3  // Stochastic phase dispersion / infinite spectral sustain
};

struct SourceSlotRuntimeParams {
    float weight         = 1.0f; // w_i in [0, 1]
    float gainLinear     = 1.0f;
    float pan            = 0.0f; // [-1, +1]
    float pitchSemitones = 0.0f; // [-24, +24]
    float loopStartNorm  = 0.0f;
    float loopEndNorm    = 1.0f;
    bool  enableH        = true; // [H] Harmonic component toggle
    bool  enableP        = true; // [P] Percussive component toggle
    bool  enableR        = true; // [R] Residual component toggle
    bool  slotLocked     = false;

    [[nodiscard]] float hGain() const noexcept { return enableH ? 1.0f : 0.0f; }
    [[nodiscard]] float pGain() const noexcept { return enableP ? 1.0f : 0.0f; }
    [[nodiscard]] float rGain() const noexcept { return enableR ? 1.0f : 0.0f; }
};

struct HopSynthesisParams {
    float     crossSynthGamma = 0.85f; // Exponent gamma in [0, 1] for Source-Filter Eq.
    float     phaseGlueBeta   = 0.85f; // Phase Glue beta_glue in [0, 1]
    float     subLockCutoffHz = 140.0f;
    int       subLockSourceId = 0;     // -1 = disabled, 0..7 = locked to slot S_{id+1}
    PhaseMode phaseMode       = PhaseMode::CausalMS_PGHI;
    double    sampleRate      = 48000.0;
    std::array<SourceSlotRuntimeParams, kMaxSources> slots{};
};

// ==============================================================================
// 5. Pre-Allocated Static Binary Max-Heap for Real-Time Causal MS-PGHI
// ==============================================================================
template <std::size_t MaxCapacity = kNumBinsPadded>
class StaticPghiMaxHeap {
public:
    struct Node {
        float    magnitude = 0.0f;
        uint16_t binIndex  = 0;
        uint8_t  isPrevFrame = 0; // 1 = seed from frame t-1, 0 = bin at frame t
    };

    void clear() noexcept {
        size_ = 0;
    }

    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

    void push(float magnitude, uint16_t binIndex, bool isPrevFrame) noexcept {
        if (size_ >= MaxCapacity) {
            return;
        }
        std::size_t idx = size_++;
        nodes_[idx] = Node{magnitude, binIndex, static_cast<uint8_t>(isPrevFrame ? 1 : 0)};
        siftUp(idx);
    }

    [[nodiscard]] Node pop() noexcept {
        const Node top = nodes_[0];
        --size_;
        if (size_ > 0) {
            nodes_[0] = nodes_[size_];
            siftDown(0);
        }
        return top;
    }

private:
    void siftUp(std::size_t idx) noexcept {
        while (idx > 0) {
            const std::size_t parent = (idx - 1) >> 1;
            if (nodes_[idx].magnitude <= nodes_[parent].magnitude) {
                break;
            }
            std::swap(nodes_[idx], nodes_[parent]);
            idx = parent;
        }
    }

    void siftDown(std::size_t idx) noexcept {
        while (true) {
            const std::size_t left  = (idx << 1) + 1;
            const std::size_t right = left + 1;
            std::size_t largest     = idx;

            if (left < size_ && nodes_[left].magnitude > nodes_[largest].magnitude) {
                largest = left;
            }
            if (right < size_ && nodes_[right].magnitude > nodes_[largest].magnitude) {
                largest = right;
            }
            if (largest == idx) {
                break;
            }
            std::swap(nodes_[idx], nodes_[largest]);
            idx = largest;
        }
    }

    std::array<Node, MaxCapacity> nodes_{};
    std::size_t                   size_ = 0;
};

// ==============================================================================
// 6. Real-Time Allocation-Free Radix-2 SIMD Inverse FFT Engine (M = 4096)
// ==============================================================================
class RealTimeFastIfft4096 {
public:
    static constexpr std::size_t kN    = kDefaultFftSize; // 4096
    static constexpr std::size_t kLogN = 12;

    RealTimeFastIfft4096() noexcept {
        initializeTables();
    }

    void initializeTables() noexcept {
        // Precompute bit-reversal permutation and twiddle factors once at init
        for (std::size_t i = 0; i < kN; ++i) {
            std::size_t rev = 0;
            std::size_t val = i;
            for (std::size_t b = 0; b < kLogN; ++b) {
                rev = (rev << 1) | (val & 1u);
                val >>= 1;
            }
            bitRev_[i] = static_cast<uint16_t>(rev);
        }

        for (std::size_t i = 0; i < kN / 2; ++i) {
            const float angle = kTwoPi * static_cast<float>(i) / static_cast<float>(kN);
            twiddleCos_[i] = std::cos(angle);
            twiddleSin_[i] = std::sin(angle); // Positive sign for Inverse FFT
        }
    }

    // Executes Inverse Real-Complex FFT from conjugate-symmetric spectrum (K = 2049)
    // into time-domain output buffer (M = 4096). Zero heap allocations.
    void executeInverseReal(
        std::span<const float, kNumBinsPadded> specReal,
        std::span<const float, kNumBinsPadded> specImag,
        std::span<float, kN>                   timeOut) noexcept
    {
        // 1. Construct Hermitian-symmetric full spectrum in bit-reversed order
        workReal_[bitRev_[0]]      = specReal[0];
        workImag_[bitRev_[0]]      = 0.0f;
        workReal_[bitRev_[kN / 2]] = specReal[kN / 2];
        workImag_[bitRev_[kN / 2]] = 0.0f;

        for (std::size_t k = 1; k < kN / 2; ++k) {
            const float re = specReal[k];
            const float im = specImag[k];
            const uint16_t rPos = bitRev_[k];
            const uint16_t rNeg = bitRev_[kN - k];
            workReal_[rPos] = re;
            workImag_[rPos] = im;
            workReal_[rNeg] = re;
            workImag_[rNeg] = -im;
        }

        // 2. In-place Cooley-Tukey Radix-2 Decimation-in-Time Butterfly stages
        for (std::size_t s = 1; s <= kLogN; ++s) {
            const std::size_t m      = std::size_t{1} << s;
            const std::size_t halfM  = m >> 1;
            const std::size_t stride = kN / m;

            for (std::size_t k = 0; k < kN; k += m) {
                for (std::size_t j = 0; j < halfM; ++j) {
                    const std::size_t twIdx = j * stride;
                    const float wRe = twiddleCos_[twIdx];
                    const float wIm = twiddleSin_[twIdx];

                    const std::size_t uIdx = k + j;
                    const std::size_t vIdx = uIdx + halfM;

                    const float vRe = workReal_[vIdx];
                    const float vIm = workImag_[vIdx];

                    const float tRe = wRe * vRe - wIm * vIm;
                    const float tIm = wRe * vIm + wIm * vRe;

                    const float uRe = workReal_[uIdx];
                    const float uIm = workImag_[uIdx];

                    workReal_[uIdx] = uRe + tRe;
                    workImag_[uIdx] = uIm + tIm;
                    workReal_[vIdx] = uRe - tRe;
                    workImag_[vIdx] = uIm - tIm;
                }
            }
        }

        // 3. Scale by 1 / N
        constexpr float kInvN = 1.0f / static_cast<float>(kN);
        for (std::size_t n = 0; n < kN; ++n) {
            timeOut[n] = workReal_[n] * kInvN;
        }
    }

private:
    alignas(kCacheLineBytes) std::array<float, kN>        workReal_{};
    alignas(kCacheLineBytes) std::array<float, kN>        workImag_{};
    alignas(kCacheLineBytes) std::array<float, kN / 2>    twiddleCos_{};
    alignas(kCacheLineBytes) std::array<float, kN / 2>    twiddleSin_{};
    alignas(kCacheLineBytes) std::array<uint16_t, kN>     bitRev_{};
};

// ==============================================================================
// 7. StaggeredSpectralPipeline — 4-Stage Sub-Hop Real-Time Pipeline
// ==============================================================================
class StaggeredSpectralPipeline {
public:
    static constexpr std::size_t kWolaBufferSize = kDefaultFftSize * 2; // 8192 power-of-2 ring
    static constexpr std::size_t kWolaMask       = kWolaBufferSize - 1;

    explicit StaggeredSpectralPipeline(std::size_t initialPhaseOffsetSamples = 0) noexcept {
        initializeGaussianSynthesisWindow();
        reset(initialPhaseOffsetSamples);
    }

    void reset(std::size_t initialPhaseOffsetSamples = 0) noexcept {
        hopSampleCounter_ = initialPhaseOffsetSamples % kDefaultHopSize;
        hopFrameCounter_  = 0;
        wolaReadPos_      = 0;
        wolaWriteAnchor_  = initialPhaseOffsetSamples % kDefaultHopSize;

        outMag_.fill(0.0f);
        prevLogMag_.fill( std::log(kLogMagFloor) );
        currLogMag_.fill( std::log(kLogMagFloor) );
        prevOmega_.fill(0.0f);
        currOmega_.fill(0.0f);
        currTau_.fill(0.0f);
        prevSynthPhase_.fill(0.0f);
        currSynthPhase_.fill(0.0f);
        dominantNativePhase_.fill(0.0f);
        dominantNativeInstFreq_.fill(0.0f);
        dominantNativeGroupDelay_.fill(0.0f);
        tilePurity_.fill(1.0f);
        dominantSourceIdx_.fill(0);
        specCartesianReal_.fill(0.0f);
        specCartesianImag_.fill(0.0f);
        ifftTimeBuffer_.fill(0.0f);
        wolaRingL_.fill(0.0f);
        wolaRingR_.fill(0.0f);
    }

    void setPhaseStaggerOffset(std::size_t offsetSamples) noexcept {
        hopSampleCounter_ = offsetSamples % kDefaultHopSize;
    }

    [[nodiscard]] std::size_t getHopSampleCounter() const noexcept {
        return hopSampleCounter_;
    }

    [[nodiscard]] uint64_t getHopFrameCounter() const noexcept {
        return hopFrameCounter_;
    }

    // --------------------------------------------------------------------------
    // STAGE 0 (Samples 0..127): SIMD Source-Filter + HPR Magnitude Recombination
    // --------------------------------------------------------------------------
    // Computes:
    // |S_out(t,k)| = E_rms(t)^(1-gamma) * (sum_i M_i^env * E_i)^gamma
    //              * (sum_j M_j^fine * (g_H F_{H,j} + g_P F_{P,j} + g_R F_{R,j}))
    // --------------------------------------------------------------------------
    void executeStage0_SourceFilterMix(
        std::span<const SpectralFrameSoA* const, kMaxSources> activeFrames,
        const MaskMatrix2D&                                   masks,
        const HopSynthesisParams&                             params) noexcept
    {
        activeMaskSnapshot_ = masks;
        currentParams_      = params;

        // Compute Sub-Bass Lock bin index: k_sub = floor(f_sub * M / f_s)
        const float binHz = static_cast<float>(params.sampleRate) / static_cast<float>(kDefaultFftSize);
        const std::size_t subLockLimitBin = (params.subLockSourceId >= 0 && params.subLockSourceId < static_cast<int>(kMaxSources))
            ? std::min<std::size_t>(kNumBins, static_cast<std::size_t>(std::max(0.0f, params.subLockCutoffHz) / binHz))
            : 0;
        const std::size_t lockedSrc = (params.subLockSourceId >= 0)
            ? static_cast<std::size_t>(params.subLockSourceId)
            : 0;

        // Compute weighted RMS reference E_rms(t) across active sources
        float weightedRms = 0.0f;
        float totalSlotWeight = 0.0f;
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            if (activeFrames[i] != nullptr && params.slots[i].weight > 0.0f) {
                weightedRms     += params.slots[i].weight * activeFrames[i]->rmsEnergy;
                totalSlotWeight += params.slots[i].weight;
            }
        }
        if (totalSlotWeight > kEps) {
            weightedRms /= totalSlotWeight;
        }
        const float gamma        = std::clamp(params.crossSynthGamma, 0.0f, 1.0f);
        const float rmsNormFactor = std::pow(std::max(weightedRms, kLogMagFloor), 1.0f - gamma);

        alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> envMix{};
        alignas(kCacheLineBytes) std::array<float, kNumBinsPadded> fineMix{};
        envMix.fill(0.0f);
        fineMix.fill(0.0f);
        tilePurity_.fill(0.0f);
        dominantSourceIdx_.fill(0);
        dominantNativePhase_.fill(0.0f);
        dominantNativeInstFreq_.fill(0.0f);
        dominantNativeGroupDelay_.fill(0.0f);

        // Accumulate across all N = 8 sources with SIMD-aligned loops over kNumBinsPadded
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const SpectralFrameSoA* frame = activeFrames[i];
            if (frame == nullptr) {
                continue;
            }

            const auto& slot = params.slots[i];
            if (slot.weight <= 0.0f || slot.gainLinear <= 0.0f) {
                continue;
            }

            const float gH    = slot.hGain() * slot.gainLinear;
            const float gP    = slot.pGain() * slot.gainLinear;
            const float gR    = slot.rGain() * slot.gainLinear;
            const float gainE = slot.gainLinear;

            const auto& mEnv  = masks.envMask[i];
            const auto& mFine = masks.fineMask[i];

            for (std::size_t k = 0; k < kNumBinsPadded; ++k) {
                const bool isSubLocked = (k < subLockLimitBin && activeFrames[lockedSrc] != nullptr);
                const float effEnvMask  = isSubLocked ? ((i == lockedSrc) ? 1.0f : 0.0f) : mEnv[k];
                const float effFineMask = isSubLocked ? ((i == lockedSrc) ? 1.0f : 0.0f) : mFine[k];

                envMix[k] += effEnvMask * (frame->env[k] * gainE);

                const float fCombined = gH * frame->fineH[k] + gP * frame->fineP[k] + gR * frame->fineR[k];
                fineMix[k] += effFineMask * fCombined;

                if (effFineMask > tilePurity_[k]) {
                    tilePurity_[k]               = effFineMask;
                    dominantSourceIdx_[k]        = static_cast<uint8_t>(i);
                    dominantNativePhase_[k]      = frame->phase[k];
                    dominantNativeInstFreq_[k]   = frame->instFreq[k];
                    dominantNativeGroupDelay_[k] = frame->groupDelay[k];
                }
            }
        }

        // Apply Source-Filter exponent gamma and combine: |S_out(t,k)| = E_rms^(1-gamma) * E_mix^gamma * F_mix
        framePeakMag_ = kLogMagFloor;
        for (std::size_t k = 0; k < kNumBins; ++k) {
            const float eTerm = (gamma >= 0.999f)
                ? envMix[k]
                : (rmsNormFactor * std::pow(std::max(envMix[k], kLogMagFloor), gamma));
            const float mag = std::max(0.0f, eTerm * fineMix[k]);
            outMag_[k] = mag;
            if (mag > framePeakMag_) {
                framePeakMag_ = mag;
            }
        }
        for (std::size_t k = kNumBins; k < kNumBinsPadded; ++k) {
            outMag_[k] = 0.0f;
        }
    }

    // --------------------------------------------------------------------------
    // STAGE 1 (Samples 128..255): Log-Magnitude L(t,k) & Cauchy-Riemann Gradients
    // --------------------------------------------------------------------------
    // Computes L(t,k) = ln(|S_out(t,k)| + eps) and fuses theoretical PGHI
    // gradients with dominant sample native derivatives via Phase Glue beta_glue.
    // --------------------------------------------------------------------------
    void executeStage1_LogMagGradients() noexcept {
        prevLogMag_ = currLogMag_;
        prevOmega_  = currOmega_;

        for (std::size_t k = 0; k < kNumBins; ++k) {
            currLogMag_[k] = std::log(std::max(outMag_[k], kLogMagFloor));
        }

        // Gaussian time-frequency aspect ratio kappa_TF = R * M / lambda^2
        // With standard Gaussian window parameter sigma_t ~ M / 6:
        constexpr float kKappaTF     = 6.0f;
        constexpr float kHalfKappaTF = 0.5f * kKappaTF;
        constexpr float kInvTwoKappa = 0.5f / kKappaTF;
        constexpr float kBaseAdv     = kTwoPi * static_cast<float>(kDefaultHopSize) / static_cast<float>(kDefaultFftSize);

        const float betaGlue = std::clamp(currentParams_.phaseGlueBeta, 0.0f, 1.0f);

        for (std::size_t k = 0; k < kNumBins; ++k) {
            const std::size_t kPrev = (k > 0) ? (k - 1) : 0;
            const std::size_t kNext = (k + 1 < kNumBins) ? (k + 1) : (kNumBins - 1);

            // Cauchy-Riemann Phase Gradient estimates (Eq. 1.4.1)
            const float dL_dk = currLogMag_[kNext] - currLogMag_[kPrev];
            const float dL_dt = currLogMag_[k] - prevLogMag_[k];

            const float omegaPghi = kBaseAdv * static_cast<float>(k) + kHalfKappaTF * dL_dk;
            const float tauPghi   = -kInvTwoKappa * dL_dt;

            // Adaptive Phase Glue blending factor: eta(t,k) = clamp(beta_glue * (2 - c_purity(t,k)), 0, 1)
            const float eta = std::clamp(betaGlue * (2.0f - tilePurity_[k]), 0.0f, 1.0f);

            currOmega_[k] = (1.0f - eta) * dominantNativeInstFreq_[k] + eta * omegaPghi;
            currTau_[k]   = (1.0f - eta) * dominantNativeGroupDelay_[k] + eta * tauPghi;
        }
    }

    // --------------------------------------------------------------------------
    // STAGE 2 (Samples 256..383): Static Binary Max-Heap Causal MS-PGHI
    // --------------------------------------------------------------------------
    // Integrates hybrid phase gradients over active bins (> -60 dB below peak)
    // using a pre-allocated static Max-Heap without dynamic memory allocation.
    // --------------------------------------------------------------------------
    void executeStage2_CausalPghiHeap() noexcept {
        prevSynthPhase_ = currSynthPhase_;

        const PhaseMode mode = currentParams_.phaseMode;
        if (mode == PhaseMode::RawSource) {
            for (std::size_t k = 0; k < kNumBins; ++k) {
                currSynthPhase_[k] = dominantNativePhase_[k];
            }
            return;
        }

        if (mode == PhaseMode::SpectralFreeze) {
            // Advance phase strictly by instantaneous frequency + slight decorrelation
            for (std::size_t k = 0; k < kNumBins; ++k) {
                currSynthPhase_[k] = wrapPhaseRadians(prevSynthPhase_[k] + dominantNativeInstFreq_[k]);
            }
            return;
        }

        if (mode == PhaseMode::IdentityPhaseLocked) {
            // Laroche & Dolson (1999) Identity Phase Locking around local spectral peaks
            for (std::size_t k = 0; k < kNumBins; ++k) {
                currSynthPhase_[k] = wrapPhaseRadians(prevSynthPhase_[k] + currOmega_[k]);
            }
            for (std::size_t k = 1; k + 1 < kNumBins; ++k) {
                if (outMag_[k] > outMag_[k - 1] && outMag_[k] >= outMag_[k + 1]) {
                    const float deltaPhi = currSynthPhase_[k] - dominantNativePhase_[k];
                    if (k > 1) {
                        currSynthPhase_[k - 1] = wrapPhaseRadians(dominantNativePhase_[k - 1] + deltaPhi);
                    }
                    if (k + 2 < kNumBins) {
                        currSynthPhase_[k + 1] = wrapPhaseRadians(dominantNativePhase_[k + 1] + deltaPhi);
                    }
                }
            }
            return;
        }

        // Default: PhaseMode::CausalMS_PGHI (Multi-Source Phase Gradient Heap Integration)
        const float magThreshold = framePeakMag_ * kPghiHeapTolRatio;
        pghiHeap_.clear();

        for (std::size_t k = 0; k < kNumBins; ++k) {
            if (outMag_[k] >= magThreshold) {
                binNeedsIntegration_[k] = 1;
                // Frame t-1 acts as causal anchor seed in the Max-Heap
                const float prevMag = std::exp(prevLogMag_[k]);
                pghiHeap_.push(prevMag, static_cast<uint16_t>(k), true);
            } else {
                binNeedsIntegration_[k] = 0;
                // Sub-threshold bins advance smoothly via instantaneous frequency
                currSynthPhase_[k] = wrapPhaseRadians(prevSynthPhase_[k] + currOmega_[k]);
            }
        }

        constexpr float kFreqStepScale = kPi * static_cast<float>(kDefaultHopSize) / static_cast<float>(kDefaultFftSize);

        while (!pghiHeap_.empty()) {
            const auto node = pghiHeap_.pop();
            const uint16_t k = node.binIndex;

            if (node.isPrevFrame != 0) {
                // Temporal propagation: (t-1, k) -> (t, k)
                if (binNeedsIntegration_[k] != 0) {
                    const float dPhiTime = 0.5f * (prevOmega_[k] + currOmega_[k]);
                    currSynthPhase_[k] = wrapPhaseRadians(prevSynthPhase_[k] + dPhiTime);
                    binNeedsIntegration_[k] = 0;
                    pghiHeap_.push(outMag_[k], k, false);
                }
            } else {
                // Frequency propagation: (t, k) -> (t, k + 1) and (t, k - 1)
                if (k + 1 < kNumBins && binNeedsIntegration_[k + 1] != 0) {
                    const float dPhiFreq = -kFreqStepScale * (currTau_[k] + currTau_[k + 1]);
                    currSynthPhase_[k + 1] = wrapPhaseRadians(currSynthPhase_[k] + dPhiFreq);
                    binNeedsIntegration_[k + 1] = 0;
                    pghiHeap_.push(outMag_[k + 1], static_cast<uint16_t>(k + 1), false);
                }
                if (k > 0 && binNeedsIntegration_[k - 1] != 0) {
                    const float dPhiFreq = kFreqStepScale * (currTau_[k] + currTau_[k - 1]);
                    currSynthPhase_[k - 1] = wrapPhaseRadians(currSynthPhase_[k] + dPhiFreq);
                    binNeedsIntegration_[k - 1] = 0;
                    pghiHeap_.push(outMag_[k - 1], static_cast<uint16_t>(k - 1), false);
                }
            }
        }
    }

    // --------------------------------------------------------------------------
    // STAGE 3 (Samples 384..511): Polar-to-Cartesian + IFFT + Circular WOLA
    // --------------------------------------------------------------------------
    void executeStage3_PolarIfftWola(float voiceStereoPan = 0.0f) noexcept {
        // 1. SIMD Polar-to-Cartesian conversion
        for (std::size_t k = 0; k < kNumBins; ++k) {
            const float mag = outMag_[k];
            const float phi = currSynthPhase_[k];
            specCartesianReal_[k] = mag * std::cos(phi);
            specCartesianImag_[k] = mag * std::sin(phi);
        }
        for (std::size_t k = kNumBins; k < kNumBinsPadded; ++k) {
            specCartesianReal_[k] = 0.0f;
            specCartesianImag_[k] = 0.0f;
        }

        // 2. Real-Complex IFFT (M = 4096)
        ifftEngine_.executeInverseReal(specCartesianReal_, specCartesianImag_, ifftTimeBuffer_);

        // 3. Equal-power stereo pan gains (combining voice pan + dominant slot pan)
        const float clampedPan = std::clamp(voiceStereoPan, -1.0f, 1.0f);
        const float panAngle   = (clampedPan + 1.0f) * (0.25f * kPi);
        const float gainL      = std::cos(panAngle);
        const float gainR      = std::sin(panAngle);

        // 4. Dual Synthesis Windowing + Circular WOLA Accumulation
        const std::size_t writeStart = wolaWriteAnchor_;
        for (std::size_t n = 0; n < kDefaultFftSize; ++n) {
            const float windowedSample = ifftTimeBuffer_[n] * synthesisWindow_[n];
            const std::size_t ringIdx  = (writeStart + n) & kWolaMask;
            wolaRingL_[ringIdx] += windowedSample * gainL;
            wolaRingR_[ringIdx] += windowedSample * gainR;
        }

        wolaWriteAnchor_ = (wolaWriteAnchor_ + kDefaultHopSize) & kWolaMask;
        ++hopFrameCounter_;
    }

    // --------------------------------------------------------------------------
    // Pack 6.65 KB GPU Telemetry Column Packet for the OpenGL/Metal Shader
    // --------------------------------------------------------------------------
    void buildGpuColumnPacket(GpuColumnPacket& outPacket) const noexcept {
        for (std::size_t b = 0; b < kGpuDisplayBands; ++b) {
            // Logarithmic mapping from 512 display bands to K = 2049 FFT bins
            const float norm     = static_cast<float>(b) / static_cast<float>(kGpuDisplayBands - 1);
            const float logRatio = ( std::pow(64.0f, norm) - 1.0f ) / 63.0f;
            const std::size_t k  = std::clamp<std::size_t>(
                static_cast<std::size_t>(logRatio * static_cast<float>(kNumBins - 1)),
                0,
                kNumBins - 1);

            const float magDb = 20.0f * std::log10(std::max(outMag_[k], 1.0e-5f));
            const float flux  = std::clamp(std::abs(currLogMag_[k] - prevLogMag_[k]) * 0.25f, 0.0f, 1.0f);

            outPacket.magDbAndTransience[b][0] = fp32ToFp16(magDb);
            outPacket.magDbAndTransience[b][1] = fp32ToFp16(flux);

            for (std::size_t i = 0; i < kMaxSources; ++i) {
                const float w = std::clamp(activeMaskSnapshot_.fineMask[i][k], 0.0f, 1.0f);
                outPacket.maskWeightsRgba[b][i] = static_cast<uint8_t>(std::lrintf(w * 255.0f));
            }
            outPacket.dominantSampleId[b] = dominantSourceIdx_[k];
        }
    }

    // --------------------------------------------------------------------------
    // Advance Pipeline by 1 Sample and Trigger Micro-Stages at 0, 128, 256, 384
    // --------------------------------------------------------------------------
    void processSingleSampleStep(
        std::span<const SpectralFrameSoA* const, kMaxSources> activeFrames,
        const MaskMatrix2D&                                   masks,
        const HopSynthesisParams&                             params,
        float                                                 voiceStereoPan,
        float&                                                outSampleL,
        float&                                                outSampleR,
        SpscRingBuffer<GpuColumnPacket>*                      gpuQueue = nullptr) noexcept
    {
        // Trigger the corresponding 128-sample micro-stage at the start of its window
        if (hopSampleCounter_ == 0) {
            executeStage0_SourceFilterMix(activeFrames, masks, params);
        } else if (hopSampleCounter_ == kMicroStageSamples) {
            executeStage1_LogMagGradients();
        } else if (hopSampleCounter_ == kMicroStageSamples * 2) {
            executeStage2_CausalPghiHeap();
        } else if (hopSampleCounter_ == kMicroStageSamples * 3) {
            executeStage3_PolarIfftWola(voiceStereoPan);
            if (gpuQueue != nullptr) {
                GpuColumnPacket pkt{};
                buildGpuColumnPacket(pkt);
                gpuQueue->tryPush(pkt);
            }
        }

        // Read & clear current sample from circular WOLA ring buffer
        const std::size_t readIdx = wolaReadPos_ & kWolaMask;
        outSampleL = wolaRingL_[readIdx];
        outSampleR = wolaRingR_[readIdx];
        wolaRingL_[readIdx] = 0.0f;
        wolaRingR_[readIdx] = 0.0f;

        wolaReadPos_      = (wolaReadPos_ + 1) & kWolaMask;
        hopSampleCounter_ = (hopSampleCounter_ + 1) & (kDefaultHopSize - 1);
    }

    [[nodiscard]] std::span<const float, kNumBinsPadded> getSynthesizedMagnitudes() const noexcept {
        return outMag_;
    }

    [[nodiscard]] std::span<const float, kNumBinsPadded> getSynthesizedPhases() const noexcept {
        return currSynthPhase_;
    }

private:
    void initializeGaussianSynthesisWindow() noexcept {
        // COLA-normalized Gaussian synthesis window for M = 4096, R = 512 (87.5% overlap)
        constexpr float kSigma = static_cast<float>(kDefaultFftSize) / 6.0f;
        constexpr float kCenter = 0.5f * static_cast<float>(kDefaultFftSize);

        for (std::size_t n = 0; n < kDefaultFftSize; ++n) {
            const float x = (static_cast<float>(n) - kCenter) / kSigma;
            synthesisWindow_[n] = std::exp(-0.5f * x * x);
        }

        // Normalize dual window for exact unity Overlap-Add reconstruction at hop R = 512
        for (std::size_t r = 0; r < kDefaultHopSize; ++r) {
            float sumSq = 0.0f;
            for (std::size_t m = r; m < kDefaultFftSize; m += kDefaultHopSize) {
                sumSq += synthesisWindow_[m];
            }
            const float norm = (sumSq > kEps) ? (1.0f / sumSq) : 1.0f;
            for (std::size_t m = r; m < kDefaultFftSize; m += kDefaultHopSize) {
                synthesisWindow_[m] *= norm;
            }
        }
    }

    std::size_t        hopSampleCounter_ = 0;
    uint64_t           hopFrameCounter_  = 0;
    std::size_t        wolaReadPos_      = 0;
    std::size_t        wolaWriteAnchor_  = 0;
    float              framePeakMag_     = kLogMagFloor;
    HopSynthesisParams currentParams_{};
    MaskMatrix2D       activeMaskSnapshot_{};

    // 64-byte aligned SoA working buffers
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   outMag_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   prevLogMag_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   currLogMag_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   prevOmega_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   currOmega_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   currTau_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   prevSynthPhase_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   currSynthPhase_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   dominantNativePhase_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   dominantNativeInstFreq_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   dominantNativeGroupDelay_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   tilePurity_{};
    alignas(kCacheLineBytes) std::array<uint8_t, kNumBinsPadded> dominantSourceIdx_{};
    alignas(kCacheLineBytes) std::array<uint8_t, kNumBinsPadded> binNeedsIntegration_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   specCartesianReal_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>   specCartesianImag_{};
    alignas(kCacheLineBytes) std::array<float, kDefaultFftSize>  ifftTimeBuffer_{};
    alignas(kCacheLineBytes) std::array<float, kDefaultFftSize>  synthesisWindow_{};
    alignas(kCacheLineBytes) std::array<float, kWolaBufferSize>  wolaRingL_{};
    alignas(kCacheLineBytes) std::array<float, kWolaBufferSize>  wolaRingR_{};

    StaticPghiMaxHeap<kNumBinsPadded> pghiHeap_{};
    RealTimeFastIfft4096              ifftEngine_{};
};

} // namespace engine
} // namespace spectile
