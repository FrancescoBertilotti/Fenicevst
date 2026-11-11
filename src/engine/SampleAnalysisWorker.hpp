#pragma once
// ==============================================================================
// SpecTile — src/engine/SampleAnalysisWorker.hpp
// Settore 3: Asynchronous Sample Pre-Analysis Worker Pool, RCU & Deferred GC
// ==============================================================================
// Performs non-causal spectral pre-analysis on background worker threads when
// user drags & drops up to N = 8 audio files:
//   1. Gaussian-windowed STFT (M = 4096, R = 512) + Instantaneous Frequency / GD
//   2. 3-Way HPR Decomposition (Directional Median Filters L_t=17, L_k=17 +
//      Soft Sigmoidal Wiener Masks with margin beta_HPR = 2.0)
//   3. Iterative Cepstral True Envelope (I_TE = 8 iterations, even DCT-I symmetry)
//      yielding E_i(t,k) and whitened fine structures F_{H,i}, F_{P,i}, F_{R,i}
//   4. 64-Band ERB Psychoacoustic Descriptors (Energy, Flatness, Flux, f0)
//   5. Lock-Free RCU Publication + DeferredSampleGarbageCollector (10 Hz GC)
// ==============================================================================

#include "StaggeredHopScheduler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace spectile::engine {

// ==============================================================================
// 1. DeferredSampleGarbageCollector & Lock-Free RealTimeRCU Slot Bank
// ==============================================================================
// Prevents `free()` from EVER executing on the real-time audio thread.
// Retired shared_ptrs are queued into a ring buffer and collected at 10 Hz
// on the MessageThread once `use_count() == 1`.
// ==============================================================================
class DeferredSampleGarbageCollector {
public:
    static constexpr std::size_t kMaxRetiredQueue = 128;

    void deferRelease(std::shared_ptr<const SpectralSampleData> retiredSample) noexcept {
        if (!retiredSample) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (retiredList_.size() >= kMaxRetiredQueue) {
            // Reclaim any already-unused entries first
            std::erase_if(retiredList_, [](const auto& ptr) {
                return !ptr || ptr.use_count() <= 1;
            });
        }
        retiredList_.push_back(std::move(retiredSample));
    }

    // Invoked periodically (e.g., 10 Hz juce::Timer on MessageThread)
    std::size_t collectUnusedOnMessageThread() {
        std::vector<std::shared_ptr<const SpectralSampleData>> toDestroyOutsideLock;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = retiredList_.begin();
            while (it != retiredList_.end()) {
                if (!(*it) || it->use_count() <= 1) {
                    toDestroyOutsideLock.push_back(std::move(*it));
                    it = retiredList_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        const std::size_t freedCount = toDestroyOutsideLock.size();
        toDestroyOutsideLock.clear(); // Actual deallocation occurs here on MessageThread
        return freedCount;
    }

    void clearAllBlocking() {
        std::lock_guard<std::mutex> lock(mutex_);
        retiredList_.clear();
    }

private:
    std::mutex                                             mutex_;
    std::vector<std::shared_ptr<const SpectralSampleData>> retiredList_;
};

// Lock-free RCU bank holding the 8 immutable SpectralSampleData slots
class SamplePoolRcuBank {
public:
    explicit SamplePoolRcuBank(DeferredSampleGarbageCollector& gc) noexcept
        : gc_(gc)
    {}

    // Called by background worker thread upon completing pre-analysis
    void publishSample(std::size_t slotIndex, std::shared_ptr<const SpectralSampleData> newSample) noexcept {
        if (slotIndex >= kMaxSources) {
            return;
        }
        std::shared_ptr<const SpectralSampleData> oldSample;
        {
            std::lock_guard<std::mutex> lock(slotMutexes_[slotIndex]);
            oldSample = std::move(slots_[slotIndex]);
            slots_[slotIndex] = std::move(newSample);
            slotGeneration_[slotIndex].fetch_add(1, std::memory_order_release);
        }
        if (oldSample) {
            gc_.deferRelease(std::move(oldSample));
        }
    }

    // Called by Audio Thread at start of processBlock — acquires snapshot without freeing
    void acquireSnapshotForAudioThread(
        std::array<std::shared_ptr<const SpectralSampleData>, kMaxSources>& outSnapshot) const noexcept
    {
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            if (std::unique_lock<std::mutex> lock(slotMutexes_[i], std::try_to_lock); lock.owns_lock()) {
                outSnapshot[i] = slots_[i];
            }
        }
    }

    [[nodiscard]] std::shared_ptr<const SpectralSampleData> getSlot(std::size_t slotIndex) const noexcept {
        if (slotIndex >= kMaxSources) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(slotMutexes_[slotIndex]);
        return slots_[slotIndex];
    }

private:
    DeferredSampleGarbageCollector&                                    gc_;
    mutable std::array<std::mutex, kMaxSources>                        slotMutexes_{};
    std::array<std::shared_ptr<const SpectralSampleData>, kMaxSources> slots_{};
    std::array<std::atomic<uint64_t>, kMaxSources>                     slotGeneration_{};
};

// ==============================================================================
// 2. Offline Forward/Inverse FFT & Pre-Analysis DSP Pipeline
// ==============================================================================
class OfflineSpectralAnalyzer {
public:
    static constexpr std::size_t kM    = kDefaultFftSize; // 4096
    static constexpr std::size_t kR    = kDefaultHopSize; // 512
    static constexpr std::size_t kLogM = 12;

    OfflineSpectralAnalyzer() {
        initializeTables();
    }

    // Executes the complete 5-stage offline pre-analysis pipeline on a mono buffer
    // while updating atomic progress in [0.0f, 1.0f] and checking cancelFlag.
    [[nodiscard]] std::shared_ptr<SpectralSampleData> analyzeMonoBuffer(
        std::size_t               slotIndex,
        const std::vector<float>& audioMono,
        double                    sampleRate,
        const std::string&        filePath,
        const std::string&        fileName,
        const std::string&        sha256Hash,
        std::atomic<float>&       progressOut,
        const std::atomic<bool>&  cancelFlag)
    {
        progressOut.store(0.02f, std::memory_order_relaxed);
        if (audioMono.empty() || cancelFlag.load(std::memory_order_relaxed)) {
            return nullptr;
        }

        auto result             = std::make_shared<SpectralSampleData>();
        result->slotIndex       = slotIndex;
        result->sampleRate      = sampleRate;
        result->fftSize         = kM;
        result->hopSize         = kR;
        result->numSamples      = audioMono.size();
        result->durationSeconds = static_cast<float>(audioMono.size()) / static_cast<float>(sampleRate);
        result->filePath        = filePath;
        result->fileName        = fileName;
        result->sha256Hash      = sha256Hash;
        result->rawAudioMono    = audioMono;

        const std::size_t numFrames = std::max<std::size_t>(
            1,
            (audioMono.size() + kR - 1) / kR);
        result->numFrames = numFrames;
        result->frames.resize(numFrames);

        // Temporary magnitude spectrogram |S_i(t,k)| for 2D HPR median filtering
        std::vector<std::array<float, kNumBinsPadded>> rawMag(numFrames);
        std::array<float, kNumBinsPadded> prevPhase{};
        prevPhase.fill(0.0f);

        std::array<float, kM> fftReal{};
        std::array<float, kM> fftImag{};

        // ----------------------------------------------------------------------
        // STAGE 1: Gaussian-Windowed STFT + Instantaneous Frequency & Group Delay
        // ----------------------------------------------------------------------
        const int halfM = static_cast<int>(kM / 2);
        const int totalSamples = static_cast<int>(audioMono.size());

        for (std::size_t t = 0; t < numFrames; ++t) {
            if (cancelFlag.load(std::memory_order_relaxed)) {
                return nullptr;
            }

            const int centerSample = static_cast<int>(t * kR);
            float frameEnergySum = 0.0f;

            for (std::size_t n = 0; n < kM; ++n) {
                const int srcIdx = centerSample + static_cast<int>(n) - halfM;
                const float s = (srcIdx >= 0 && srcIdx < totalSamples) ? audioMono[static_cast<std::size_t>(srcIdx)] : 0.0f;
                fftReal[n] = s * gaussianWindow_[n];
                fftImag[n] = 0.0f;
                frameEnergySum += s * s;
            }

            executeComplexFft(fftReal, fftImag, false);

            auto& frame = result->frames[t];
            frame.clear();
            frame.rmsEnergy = std::sqrt(frameEnergySum / static_cast<float>(kM));

            constexpr float kExpectedAdv = kTwoPi * static_cast<float>(kR) / static_cast<float>(kM);
            for (std::size_t k = 0; k < kNumBins; ++k) {
                const float re  = fftReal[k];
                const float im  = fftImag[k];
                const float mag = std::sqrt(re * re + im * im);
                const float phi = std::atan2(im, re);

                rawMag[t][k]   = mag;
                frame.phase[k] = phi;

                // Instantaneous Frequency omega_i^IF(t,k) via phase unwrapping
                const float expectedPhi = prevPhase[k] + kExpectedAdv * static_cast<float>(k);
                const float devPhi      = wrapPhaseRadians(phi - expectedPhi);
                frame.instFreq[k]       = kExpectedAdv * static_cast<float>(k) + devPhi;
                prevPhase[k]            = phi;
            }

            // Local Group Delay tau_i^GD(t,k) = -d(phi)/dk
            for (std::size_t k = 0; k < kNumBins; ++k) {
                const std::size_t kPrev = (k > 0) ? (k - 1) : 0;
                const std::size_t kNext = (k + 1 < kNumBins) ? (k + 1) : (kNumBins - 1);
                frame.groupDelay[k] = -0.5f * wrapPhaseRadians(frame.phase[kNext] - frame.phase[kPrev]);
            }

            if ((t & 7u) == 0) {
                progressOut.store(0.05f + 0.25f * (static_cast<float>(t) / static_cast<float>(numFrames)),
                                  std::memory_order_relaxed);
            }
        }

        // ----------------------------------------------------------------------
        // STAGE 2 & 3: 3-Way HPR Decomposition (Directional Median L_t=17, L_k=17)
        //              + Iterative Cepstral True Envelope (I_TE = 8)
        // ----------------------------------------------------------------------
        constexpr int   kHalfMedian = 8; // L_t = L_k = 17 => radius = 8
        constexpr float kBetaHpr    = 2.0f;
        constexpr float kGammaM     = 2.5f;

        std::array<float, 17> medianBuf{};
        std::array<float, kNumBinsPadded> hTilde{};
        std::array<float, kNumBinsPadded> pTilde{};
        std::array<float, kNumBinsPadded> hMag{};
        std::array<float, kNumBinsPadded> pMag{};
        std::array<float, kNumBinsPadded> rMag{};

        float globalF0Accum = 0.0f;
        float globalF0Weight = 0.0f;

        for (std::size_t t = 0; t < numFrames; ++t) {
            if (cancelFlag.load(std::memory_order_relaxed)) {
                return nullptr;
            }

            // 2a. Horizontal (Time) Median -> Harmonic estimate ~H_i(t,k)
            for (std::size_t k = 0; k < kNumBins; ++k) {
                std::size_t count = 0;
                for (int dt = -kHalfMedian; dt <= kHalfMedian; ++dt) {
                    const int tClamped = std::clamp(static_cast<int>(t) + dt, 0, static_cast<int>(numFrames) - 1);
                    medianBuf[count++] = rawMag[static_cast<std::size_t>(tClamped)][k];
                }
                std::nth_element(medianBuf.begin(), medianBuf.begin() + (count / 2), medianBuf.begin() + count);
                hTilde[k] = medianBuf[count / 2];
            }

            // 2b. Vertical (Frequency) Median -> Percussive estimate ~P_i(t,k)
            for (std::size_t k = 0; k < kNumBins; ++k) {
                std::size_t count = 0;
                for (int dk = -kHalfMedian; dk <= kHalfMedian; ++dk) {
                    const int kClamped = std::clamp(static_cast<int>(k) + dk, 0, static_cast<int>(kNumBins) - 1);
                    medianBuf[count++] = rawMag[t][static_cast<std::size_t>(kClamped)];
                }
                std::nth_element(medianBuf.begin(), medianBuf.begin() + (count / 2), medianBuf.begin() + count);
                pTilde[k] = medianBuf[count / 2];
            }

            // 2c. Soft Sigmoidal Wiener Masks (Eq. 1.3.1)
            for (std::size_t k = 0; k < kNumBins; ++k) {
                const float sVal = rawMag[t][k];
                const float hVal = hTilde[k];
                const float pVal = pTilde[k];

                const float rH   = hVal / (pVal + kEps);
                const float rP   = pVal / (hVal + kEps);
                const float sigH = 1.0f / (1.0f + std::exp(-kGammaM * (rH - kBetaHpr)));
                const float sigP = 1.0f / (1.0f + std::exp(-kGammaM * (rP - kBetaHpr)));

                const float wH   = (hVal * hVal) * sigH;
                const float wP   = (pVal * pVal) * sigP;
                const float wR   = (sVal * sVal) * (1.0f - sigH) * (1.0f - sigP);
                const float wSum = wH + wP + wR + kEps;

                hMag[k] = sVal * (wH / wSum);
                pMag[k] = sVal * (wP / wSum);
                rMag[k] = sVal * (wR / wSum);
            }

            // 3. Fundamental f0 estimation & Iterative Cepstral True Envelope (I_TE = 8)
            auto& frame = result->frames[t];
            frame.detectedF0Hz = estimateFrameF0Hz(hMag, sampleRate);
            if (frame.rmsEnergy > 1.0e-3f) {
                globalF0Accum  += frame.detectedF0Hz * frame.rmsEnergy;
                globalF0Weight += frame.rmsEnergy;
            }

            computeIterativeTrueEnvelope(rawMag[t], frame.detectedF0Hz, sampleRate, frame.env);

            // Factorize whitened fine structure F_{H,i}, F_{P,i}, F_{R,i} in [0, 1.05]
            for (std::size_t k = 0; k < kNumBins; ++k) {
                const float invEnv = 1.0f / std::max(frame.env[k], kLogMagFloor);
                frame.fineH[k] = std::clamp(hMag[k] * invEnv, 0.0f, 1.05f);
                frame.fineP[k] = std::clamp(pMag[k] * invEnv, 0.0f, 1.05f);
                frame.fineR[k] = std::clamp(rMag[k] * invEnv, 0.0f, 1.05f);
            }

            // 4. Compute 64-band ERB Descriptors (Energy, Flatness, Spectral Flux)
            const auto* prevMagPtr = (t > 0) ? &rawMag[t - 1] : nullptr;
            computeErbDescriptors(rawMag[t], prevMagPtr, frame);

            if ((t & 7u) == 0) {
                progressOut.store(0.30f + 0.65f * (static_cast<float>(t) / static_cast<float>(numFrames)),
                                  std::memory_order_relaxed);
            }
        }

        result->referenceF0Hz = (globalF0Weight > kEps)
            ? std::clamp(globalF0Accum / globalF0Weight, 30.0f, 2000.0f)
            : 220.0f;

        // 5. Populate compact 12-byte/bin storage (float16 + int16)
        result->compactStorage.resize(numFrames * kNumBins);
        for (std::size_t t = 0; t < numFrames; ++t) {
            const auto& f = result->frames[t];
            const std::size_t rowOffset = t * kNumBins;
            for (std::size_t k = 0; k < kNumBins; ++k) {
                PackedSpectralBin12B packed{};
                packed.logEnvFp16  = fp32ToFp16(std::log(std::max(f.env[k], kLogMagFloor)));
                packed.fineHFp16   = fp32ToFp16(f.fineH[k]);
                packed.finePFp16   = fp32ToFp16(f.fineP[k]);
                packed.fineRFp16   = fp32ToFp16(f.fineR[k]);
                packed.phaseQ15    = quantizePhaseInt16(f.phase[k]);
                packed.instFreqQ15 = quantizePhaseInt16(f.instFreq[k]);
                result->compactStorage[rowOffset + k] = packed;
            }
        }

        progressOut.store(1.0f, std::memory_order_release);
        return result;
    }

private:
    void initializeTables() noexcept {
        // Gaussian analysis window
        constexpr float kSigma  = static_cast<float>(kM) / 6.0f;
        constexpr float kCenter = 0.5f * static_cast<float>(kM);
        for (std::size_t n = 0; n < kM; ++n) {
            const float x = (static_cast<float>(n) - kCenter) / kSigma;
            gaussianWindow_[n] = std::exp(-0.5f * x * x);
        }

        // Bit-reversal and twiddle tables for M = 4096
        for (std::size_t i = 0; i < kM; ++i) {
            std::size_t rev = 0;
            std::size_t val = i;
            for (std::size_t b = 0; b < kLogM; ++b) {
                rev = (rev << 1) | (val & 1u);
                val >>= 1;
            }
            bitRev_[i] = static_cast<uint16_t>(rev);
        }
        for (std::size_t i = 0; i < kM / 2; ++i) {
            const float angle = kTwoPi * static_cast<float>(i) / static_cast<float>(kM);
            twiddleCos_[i] = std::cos(angle);
            twiddleSin_[i] = std::sin(angle);
        }

        // Precompute ERB band bin limits (Glasberg & Moore, 1990)
        const float maxErb = hzToErbCam(24000.0f);
        for (std::size_t b = 0; b < kNumErbBands; ++b) {
            const float erbLow  = maxErb * (static_cast<float>(b) / static_cast<float>(kNumErbBands));
            const float erbHigh = maxErb * (static_cast<float>(b + 1) / static_cast<float>(kNumErbBands));
            const float hzLow   = erbCamToHz(erbLow);
            const float hzHigh  = erbCamToHz(erbHigh);

            erbBinStart_[b] = std::clamp<uint16_t>(
                static_cast<uint16_t>(hzLow * static_cast<float>(kM) / 48000.0f),
                0,
                static_cast<uint16_t>(kNumBins - 1));
            erbBinEnd_[b] = std::clamp<uint16_t>(
                static_cast<uint16_t>(std::max<int>(erbBinStart_[b] + 1, static_cast<int>(hzHigh * static_cast<float>(kM) / 48000.0f))),
                1,
                static_cast<uint16_t>(kNumBins));
        }
    }

    [[nodiscard]] static float hzToErbCam(float fHz) noexcept {
        return 21.4f * std::log10(0.00437f * std::max(0.0f, fHz) + 1.0f);
    }

    [[nodiscard]] static float erbCamToHz(float cam) noexcept {
        return (std::pow(10.0f, cam / 21.4f) - 1.0f) / 0.00437f;
    }

    void executeComplexFft(
        std::array<float, kM>& real,
        std::array<float, kM>& imag,
        bool                   inverse) const noexcept
    {
        for (std::size_t i = 0; i < kM; ++i) {
            const uint16_t r = bitRev_[i];
            if (i < r) {
                std::swap(real[i], real[r]);
                std::swap(imag[i], imag[r]);
            }
        }

        const float sign = inverse ? 1.0f : -1.0f;
        for (std::size_t s = 1; s <= kLogM; ++s) {
            const std::size_t m      = std::size_t{1} << s;
            const std::size_t halfM  = m >> 1;
            const std::size_t stride = kM / m;

            for (std::size_t k = 0; k < kM; k += m) {
                for (std::size_t j = 0; j < halfM; ++j) {
                    const std::size_t twIdx = j * stride;
                    const float wRe = twiddleCos_[twIdx];
                    const float wIm = sign * twiddleSin_[twIdx];

                    const std::size_t uIdx = k + j;
                    const std::size_t vIdx = uIdx + halfM;

                    const float vRe = real[vIdx];
                    const float vIm = imag[vIdx];
                    const float tRe = wRe * vRe - wIm * vIm;
                    const float tIm = wRe * vIm + wIm * vRe;

                    real[vIdx] = real[uIdx] - tRe;
                    imag[vIdx] = imag[uIdx] - tIm;
                    real[uIdx] += tRe;
                    imag[uIdx] += tIm;
                }
            }
        }

        if (inverse) {
            constexpr float kInvM = 1.0f / static_cast<float>(kM);
            for (std::size_t i = 0; i < kM; ++i) {
                real[i] *= kInvM;
                imag[i] *= kInvM;
            }
        }
    }

    // Iterative Cepstral True Envelope (8 iterations, DCT-I even symmetry, Eq. 1.3.2)
    void computeIterativeTrueEnvelope(
        const std::array<float, kNumBinsPadded>& mag,
        float                                    f0Hz,
        double                                   sampleRate,
        std::array<float, kNumBinsPadded>&       outEnv) const noexcept
    {
        std::array<float, kM> a0{};
        std::array<float, kM> aCurr{};
        std::array<float, kM> cepReal{};
        std::array<float, kM> cepImag{};

        for (std::size_t k = 0; k < kNumBins; ++k) {
            const float logVal = std::log(std::max(mag[k], kLogMagFloor));
            a0[k]    = logVal;
            aCurr[k] = logVal;
        }
        // Even DCT-I symmetric extension: A(M - m) = A(m)
        for (std::size_t k = 1; k < kNumBins - 1; ++k) {
            a0[kM - k]    = a0[k];
            aCurr[kM - k] = aCurr[k];
        }

        const std::size_t qMax = std::clamp<std::size_t>(
            static_cast<std::size_t>(sampleRate / (2.2 * std::max(40.0f, f0Hz))),
            24,
            120);

        constexpr int   kNumIterations  = 8;
        constexpr float kOvershootDelta = 0.02f;

        for (int iter = 0; iter < kNumIterations; ++iter) {
            cepReal = aCurr;
            cepImag.fill(0.0f);
            executeComplexFft(cepReal, cepImag, false);

            // Apply Hann-like low-quefrency liftering window up to qMax
            for (std::size_t q = 1; q <= kM / 2; ++q) {
                float w = 0.0f;
                if (q <= qMax) {
                    w = 0.5f * (1.0f + std::cos(kPi * static_cast<float>(q) / static_cast<float>(qMax)));
                }
                cepReal[q]      *= w;
                cepImag[q]      *= w;
                cepReal[kM - q] *= w;
                cepImag[kM - q] *= w;
            }

            executeComplexFft(cepReal, cepImag, true);

            for (std::size_t m = 0; m < kM; ++m) {
                aCurr[m] = std::max(a0[m], cepReal[m] - kOvershootDelta);
            }
        }

        for (std::size_t k = 0; k < kNumBins; ++k) {
            outEnv[k] = std::exp(cepReal[k]);
        }
        for (std::size_t k = kNumBins; k < kNumBinsPadded; ++k) {
            outEnv[k] = kLogMagFloor;
        }
    }

    [[nodiscard]] static float estimateFrameF0Hz(
        const std::array<float, kNumBinsPadded>& harmonicMag,
        double                                   sampleRate) noexcept
    {
        const float binHz = static_cast<float>(sampleRate) / static_cast<float>(kM);
        const std::size_t minBin = std::max<std::size_t>(2, static_cast<std::size_t>(45.0f / binHz));
        const std::size_t maxBin = std::min<std::size_t>(kNumBins / 4, static_cast<std::size_t>(1200.0f / binHz));

        float bestScore = 0.0f;
        std::size_t bestBin = static_cast<std::size_t>(220.0f / binHz);

        // Harmonic Product / Sum salience across first 4 harmonics
        for (std::size_t k = minBin; k <= maxBin; ++k) {
            float score = harmonicMag[k]
                        + 0.50f * harmonicMag[std::min(kNumBins - 1, k * 2)]
                        + 0.33f * harmonicMag[std::min(kNumBins - 1, k * 3)]
                        + 0.25f * harmonicMag[std::min(kNumBins - 1, k * 4)];
            if (score > bestScore) {
                bestScore = score;
                bestBin   = k;
            }
        }
        return static_cast<float>(bestBin) * binHz;
    }

    void computeErbDescriptors(
        const std::array<float, kNumBinsPadded>& currMag,
        const std::array<float, kNumBinsPadded>* prevMag,
        SpectralFrameSoA&                        frame) const noexcept
    {
        for (std::size_t b = 0; b < kNumErbBands; ++b) {
            const uint16_t kStart = erbBinStart_[b];
            const uint16_t kEnd   = erbBinEnd_[b];
            const float    invCount = 1.0f / static_cast<float>(std::max<uint16_t>(1, kEnd - kStart));

            float arithMean = 0.0f;
            float logSum    = 0.0f;
            float energySum = 0.0f;
            float fluxSum   = 0.0f;

            for (uint16_t k = kStart; k < kEnd; ++k) {
                const float m  = std::max(currMag[k], kLogMagFloor);
                const float m2 = m * m;
                arithMean += m;
                logSum    += std::log(m);
                energySum += m2;

                if (prevMag != nullptr) {
                    const float diff = currMag[k] - (*prevMag)[k];
                    if (diff > 0.0f) {
                        fluxSum += diff * diff;
                    }
                }
            }

            arithMean *= invCount;
            const float geomMean = std::exp(logSum * invCount);

            frame.erbEnergy[b]   = 10.0f * std::log10(std::max(energySum * invCount, 1.0e-8f));
            frame.erbFlatness[b] = std::clamp(geomMean / (arithMean + kEps), 0.0f, 1.0f);
            frame.erbFlux[b]     = std::sqrt(fluxSum * invCount);
        }
    }

    std::array<float, kM>               gaussianWindow_{};
    std::array<uint16_t, kM>            bitRev_{};
    std::array<float, kM / 2>           twiddleCos_{};
    std::array<float, kM / 2>           twiddleSin_{};
    std::array<uint16_t, kNumErbBands>  erbBinStart_{};
    std::array<uint16_t, kNumErbBands>  erbBinEnd_{};
};

// ==============================================================================
// 3. SampleAnalysisWorkerPool — Multi-Threaded Background Pre-Analysis Engine
// ==============================================================================
class SampleAnalysisWorkerPool {
public:
    struct AnalysisJob {
        std::size_t        slotIndex  = 0;
        std::vector<float> audioMono;
        double             sampleRate = 48000.0;
        std::string        filePath;
        std::string        fileName;
        std::string        sha256Hash;
    };

    explicit SampleAnalysisWorkerPool(
        SamplePoolRcuBank& rcuBank,
        std::size_t        numThreads = 2)
        : rcuBank_(rcuBank)
    {
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            analysisProgress_[i].store(0.0f, std::memory_order_relaxed);
            slotCancelFlags_[i].store(false, std::memory_order_relaxed);
            isSlotAnalyzing_[i].store(false, std::memory_order_relaxed);
        }

        const std::size_t threadCount = std::clamp<std::size_t>(numThreads, 1, 4);
        workers_.reserve(threadCount);
        for (std::size_t t = 0; t < threadCount; ++t) {
            workers_.emplace_back([this]() { workerThreadLoop(); });
        }
    }

    ~SampleAnalysisWorkerPool() {
        shutdown();
    }

    SampleAnalysisWorkerPool(const SampleAnalysisWorkerPool&)            = delete;
    SampleAnalysisWorkerPool& operator=(const SampleAnalysisWorkerPool&) = delete;

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (shuttingDown_) {
                return;
            }
            shuttingDown_ = true;
            for (auto& flag : slotCancelFlags_) {
                flag.store(true, std::memory_order_relaxed);
            }
            jobQueue_.clear();
        }
        queueCv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Enqueues an asynchronous pre-analysis job for slotIndex in [0..7]
    void enqueueAnalysisJob(AnalysisJob job) {
        if (job.slotIndex >= kMaxSources || job.audioMono.empty()) {
            return;
        }
        const std::size_t slot = job.slotIndex;
        slotCancelFlags_[slot].store(true, std::memory_order_relaxed);
        analysisProgress_[slot].store(0.0f, std::memory_order_relaxed);
        isSlotAnalyzing_[slot].store(true, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            // Remove any pending queued job for the same slot
            std::erase_if(jobQueue_, [slot](const AnalysisJob& queued) {
                return queued.slotIndex == slot;
            });
            jobQueue_.push_back(std::move(job));
        }
        queueCv_.notify_one();
    }

    [[nodiscard]] float getSlotProgress(std::size_t slotIndex) const noexcept {
        if (slotIndex >= kMaxSources) {
            return 0.0f;
        }
        return analysisProgress_[slotIndex].load(std::memory_order_acquire);
    }

    [[nodiscard]] bool isSlotAnalyzing(std::size_t slotIndex) const noexcept {
        if (slotIndex >= kMaxSources) {
            return false;
        }
        return isSlotAnalyzing_[slotIndex].load(std::memory_order_acquire);
    }

private:
    void workerThreadLoop() {
        OfflineSpectralAnalyzer analyzer;

        while (true) {
            AnalysisJob currentJob;
            {
                std::unique_lock<std::mutex> lock(queueMutex_);
                queueCv_.wait(lock, [this]() {
                    return shuttingDown_ || !jobQueue_.empty();
                });

                if (shuttingDown_ && jobQueue_.empty()) {
                    return;
                }

                currentJob = std::move(jobQueue_.front());
                jobQueue_.pop_front();
                slotCancelFlags_[currentJob.slotIndex].store(false, std::memory_order_relaxed);
            }

            const std::size_t slot = currentJob.slotIndex;
            auto analyzedData = analyzer.analyzeMonoBuffer(
                slot,
                currentJob.audioMono,
                currentJob.sampleRate,
                currentJob.filePath,
                currentJob.fileName,
                currentJob.sha256Hash,
                analysisProgress_[slot],
                slotCancelFlags_[slot]);

            if (analyzedData && !slotCancelFlags_[slot].load(std::memory_order_relaxed)) {
                rcuBank_.publishSample(slot, std::move(analyzedData));
            }
            isSlotAnalyzing_[slot].store(false, std::memory_order_release);
        }
    }

    SamplePoolRcuBank&                            rcuBank_;
    std::mutex                                    queueMutex_;
    std::condition_variable                       queueCv_;
    std::deque<AnalysisJob>                       jobQueue_;
    bool                                          shuttingDown_ = false;
    std::vector<std::thread>                      workers_;
    std::array<std::atomic<float>, kMaxSources>   analysisProgress_{};
    std::array<std::atomic<bool>, kMaxSources>    slotCancelFlags_{};
    std::array<std::atomic<bool>, kMaxSources>    isSlotAnalyzing_{};
};

} // namespace spectile::engine
