#pragma once
// ==============================================================================
// SpecTile — src/engine/SpecTileVoice.hpp
// Settore 3: Dual-Mode Polyphonic & Phase-Staggered MPE Voice Engine
// ==============================================================================
// Implements:
//   - Mode A (Shared Spectral Bus, 16 Voices): Catmull-Rom cubic spectral
//     resampling of harmonic fine structure F_i(t, k / alpha_v) with decoupled
//     True Envelope formant preservation E_i(t, k / beta_v), feeding 1 shared
//     global StaggeredSpectralPipeline (1 IFFT total for all 16 voices).
//   - Mode B (True Per-Voice Chimeric ISTFT, 6 MPE Voices): Phase-Staggered
//     scheduling (offset_v = floor(v * R_a / 6) in {0, 85, 170, 256, 341, 426})
//     with per-note MPE expressions:
//       * Pitchbend X -> Continuous harmonic ratio alpha_v (+/- 48 st)
//       * Slide Y     -> Per-voice genetic branch morph / mask shift
//       * Pressure Z  -> Per-voice Source-Filter exponent gamma_v
// ==============================================================================

#include "StaggeredHopScheduler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace spectile::engine {

// ==============================================================================
// 1. Polyphony & Play Mode Enumerations
// ==============================================================================
enum class PolyphonyMode : uint8_t {
    SharedSpectralBus16 = 0, // Mode A: 16 voices summed in spectral domain -> 1 global ISTFT
    TrueMpePerVoice6    = 1  // Mode B: 6 independent MPE voices with staggered ISTFTs
};

enum class PlayMode : uint8_t {
    PitchedInstrument   = 0, // MIDI Polyphonic & MPE pitch-tracked resynthesis
    SpectralDroneLatch  = 1, // Infinite asynchronous time-stretched drone & orbital drift
    RhythmicLoopSynced  = 2, // DAW PPQ 16th-note tempo-quantized loop mosaicing
    LiveSidechainHybrid = 3  // Live audio input as Slot S1 modulating Sample Pool S2..S8
};

inline constexpr std::size_t kMaxSharedBusVoices = 16;
inline constexpr std::size_t kMaxMpeVoices       = 6;

// Phase-staggered offsets for V = 6 MPE voices across R_a = 512 samples:
// offset_v = floor(v * 512 / 6) = {0, 85, 170, 256, 341, 426}
inline constexpr std::array<std::size_t, kMaxMpeVoices> kMpeVoicePhaseOffsets = {
    0, 85, 170, 256, 341, 426
};

// ==============================================================================
// 2. Catmull-Rom Cubic Spectral Interpolation & Formant-Preserving Resampler
// ==============================================================================
struct CatmullRomResampler {
    // Evaluates 4-point Catmull-Rom cubic spline at fractional bin coordinate coord
    // with brickwall anti-aliasing above Nyquist (coord > K - 1 => 0.0f).
    [[nodiscard]] static inline float sampleCubic(
        std::span<const float, kNumBinsPadded> data,
        float                                  coord) noexcept
    {
        if (coord < 0.0f || coord >= static_cast<float>(kNumBins - 1)) {
            return 0.0f; // Anti-aliasing zeroing beyond Nyquist
        }

        const int   k1 = static_cast<int>(coord);
        const float t  = coord - static_cast<float>(k1);

        const int k0 = std::max(0, k1 - 1);
        const int k2 = std::min(static_cast<int>(kNumBins - 1), k1 + 1);
        const int k3 = std::min(static_cast<int>(kNumBins - 1), k1 + 2);

        const float y0 = data[static_cast<std::size_t>(k0)];
        const float y1 = data[static_cast<std::size_t>(k1)];
        const float y2 = data[static_cast<std::size_t>(k2)];
        const float y3 = data[static_cast<std::size_t>(k3)];

        // Standard Catmull-Rom spline polynomial in Horner form
        const float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
        const float a1 =  y0        - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float a2 = -0.5f * y0             + 0.5f * y2;
        const float a3 =  y1;

        const float interpolated = ((a0 * t + a1) * t + a2) * t + a3;
        return std::max(0.0f, interpolated);
    }

    // Resamples a source SpectralFrameSoA into a pitch-shifted & formant-preserved frame:
    //   - Harmonic fine structure F_H, F_R read at k_fine = k / alpha_v (Catmull-Rom)
    //   - Percussive fine structure F_P kept unshifted (or blended) to preserve transient punch
    //   - True Envelope E_i read at k_env = k / beta_v where beta_v = formantShift * alpha_v^keytrack
    //   - Instantaneous frequency scaled by alpha_v for Phase-Locked coherence
    static void resampleFrameSourceFilter(
        const SpectralFrameSoA& src,
        SpectralFrameSoA&       dst,
        float                   alphaPitchRatio,
        float                   betaFormantRatio,
        bool                    lockPercussivePitch = true) noexcept
    {
        const float invAlpha = 1.0f / std::clamp(alphaPitchRatio, 0.0625f, 16.0f);
        const float invBeta  = 1.0f / std::clamp(betaFormantRatio, 0.25f, 4.0f);

        for (std::size_t k = 0; k < kNumBins; ++k) {
            const float kf     = static_cast<float>(k);
            const float kFine  = kf * invAlpha;
            const float kEnv   = kf * invBeta;

            dst.env[k]   = std::max(kLogMagFloor, sampleCubic(src.env, kEnv));
            dst.fineH[k] = std::clamp(sampleCubic(src.fineH, kFine), 0.0f, 1.05f);
            dst.fineR[k] = std::clamp(sampleCubic(src.fineR, kFine), 0.0f, 1.05f);
            dst.fineP[k] = lockPercussivePitch
                ? src.fineP[k]
                : std::clamp(sampleCubic(src.fineP, kFine), 0.0f, 1.05f);

            const std::size_t nearestBin = std::min<std::size_t>(
                kNumBins - 1,
                static_cast<std::size_t>(std::max(0.0f, std::round(kFine))));
            dst.phase[k]      = src.phase[nearestBin];
            dst.instFreq[k]   = wrapPhaseRadians(src.instFreq[nearestBin] * alphaPitchRatio);
            dst.groupDelay[k] = src.groupDelay[nearestBin];
        }

        for (std::size_t k = kNumBins; k < kNumBinsPadded; ++k) {
            dst.env[k]        = kLogMagFloor;
            dst.fineH[k]      = 0.0f;
            dst.fineP[k]      = 0.0f;
            dst.fineR[k]      = 0.0f;
            dst.phase[k]      = 0.0f;
            dst.instFreq[k]   = 0.0f;
            dst.groupDelay[k] = 0.0f;
        }

        dst.erbEnergy    = src.erbEnergy;
        dst.erbFlatness  = src.erbFlatness;
        dst.erbFlux      = src.erbFlux;
        dst.rmsEnergy    = src.rmsEnergy;
        dst.detectedF0Hz = src.detectedF0Hz * alphaPitchRatio;
    }
};

// ==============================================================================
// 3. Per-Voice Fast Linear-in-dB / Exponential ADSR Envelope
// ==============================================================================
class VoiceAdsrEnvelope {
public:
    enum class Stage : uint8_t { Idle = 0, Attack, Decay, Sustain, Release };

    void setParameters(float attackSec, float decaySec, float sustainLevel, float releaseSec, double sampleRate) noexcept {
        const float sr = static_cast<float>(std::max(1000.0, sampleRate));
        attackStep_    = 1.0f / std::max(1.0f, attackSec * sr);
        decayCoeff_    = std::exp(-4.60517f / std::max(1.0f, decaySec * sr));
        sustainLevel_  = std::clamp(sustainLevel, 0.0f, 1.0f);
        releaseCoeff_  = std::exp(-4.60517f / std::max(1.0f, releaseSec * sr));
    }

    void noteOn() noexcept {
        stage_ = Stage::Attack;
    }

    void noteOff() noexcept {
        if (stage_ != Stage::Idle) {
            stage_ = Stage::Release;
        }
    }

    void reset() noexcept {
        stage_ = Stage::Idle;
        value_ = 0.0f;
    }

    [[nodiscard]] float getNextSample() noexcept {
        switch (stage_) {
            case Stage::Idle:
                value_ = 0.0f;
                break;
            case Stage::Attack:
                value_ += attackStep_;
                if (value_ >= 1.0f) {
                    value_ = 1.0f;
                    stage_ = Stage::Decay;
                }
                break;
            case Stage::Decay:
                value_ = sustainLevel_ + (value_ - sustainLevel_) * decayCoeff_;
                if (std::abs(value_ - sustainLevel_) < 1.0e-4f) {
                    value_ = sustainLevel_;
                    stage_ = Stage::Sustain;
                }
                break;
            case Stage::Sustain:
                value_ = sustainLevel_;
                break;
            case Stage::Release:
                value_ *= releaseCoeff_;
                if (value_ < 1.0e-4f) {
                    value_ = 0.0f;
                    stage_ = Stage::Idle;
                }
                break;
        }
        return value_;
    }

    [[nodiscard]] float getCurrentValue() const noexcept { return value_; }
    [[nodiscard]] bool  isActive() const noexcept { return stage_ != Stage::Idle; }
    [[nodiscard]] bool  isReleasing() const noexcept { return stage_ == Stage::Release; }

private:
    Stage stage_        = Stage::Idle;
    float value_        = 0.0f;
    float attackStep_   = 0.001f;
    float decayCoeff_   = 0.9995f;
    float sustainLevel_ = 0.85f;
    float releaseCoeff_ = 0.9990f;
};

// ==============================================================================
// 4. SpecTileVoice — Unified Voice State for Shared-Bus (Mode A) & MPE (Mode B)
// ==============================================================================
class SpecTileVoice {
public:
    SpecTileVoice(std::size_t voiceIndex = 0) noexcept
        : voiceIndex_(voiceIndex)
        , pipeline_(kMpeVoicePhaseOffsets[voiceIndex % kMaxMpeVoices])
    {
        reset();
    }

    void setVoiceIndex(std::size_t voiceIndex) noexcept {
        voiceIndex_ = voiceIndex;
        reset();
    }

    void prepare(double sampleRate) noexcept {
        sampleRate_ = sampleRate;
        adsr_.setParameters(0.012f, 0.18f, 0.88f, 0.35f, sampleRate);
        reset();
    }

    void reset() noexcept {
        active_                  = false;
        midiNote_                = -1;
        midiChannel_             = 1;
        velocity_                = 0.0f;
        noteOnTimestamp_         = 0;
        mpePitchbendSemitones_   = 0.0f;
        smoothedPitchbendSemi_   = 0.0f;
        mpeSlideY_               = 0.5f;
        smoothedSlideY_          = 0.5f;
        mpePressureZ_            = 0.0f;
        smoothedPressureZ_       = 0.0f;
        sourcePlayheadFrames_.fill(0.0f);
        adsr_.reset();
        pipeline_.reset(kMpeVoicePhaseOffsets[voiceIndex_ % kMaxMpeVoices]);
    }

    void startNote(
        int      midiNote,
        float    velocityNorm,
        int      midiChannel,
        uint64_t timestamp,
        float    initialPitchbendSemitones = 0.0f,
        float    initialSlideY             = 0.5f,
        float    initialPressureZ          = 0.0f) noexcept
    {
        active_                = true;
        midiNote_              = midiNote;
        midiChannel_           = midiChannel;
        velocity_              = std::clamp(velocityNorm, 0.0f, 1.0f);
        noteOnTimestamp_       = timestamp;
        mpePitchbendSemitones_ = initialPitchbendSemitones;
        smoothedPitchbendSemi_ = initialPitchbendSemitones;
        mpeSlideY_             = std::clamp(initialSlideY, 0.0f, 1.0f);
        smoothedSlideY_        = mpeSlideY_;
        mpePressureZ_          = std::clamp(initialPressureZ, 0.0f, 1.0f);
        smoothedPressureZ_     = mpePressureZ_;
        sourcePlayheadFrames_.fill(0.0f);
        adsr_.noteOn();
    }

    void stopNote(bool allowTailOff = true) noexcept {
        if (allowTailOff) {
            adsr_.noteOff();
        } else {
            adsr_.reset();
            active_   = false;
            midiNote_ = -1;
        }
    }

    // MPE Per-Note Expression Setters (X: Pitchbend, Y: Slide/Timbre, Z: Pressure)
    void setMpePitchbendSemitones(float semitones) noexcept {
        mpePitchbendSemitones_ = std::clamp(semitones, -48.0f, 48.0f);
    }

    void setMpeSlideY(float slideNorm) noexcept {
        mpeSlideY_ = std::clamp(slideNorm, 0.0f, 1.0f);
    }

    void setMpePressureZ(float pressureNorm) noexcept {
        mpePressureZ_ = std::clamp(pressureNorm, 0.0f, 1.0f);
    }

    void setAdsrParameters(float a, float d, float s, float r) noexcept {
        adsr_.setParameters(a, d, s, r, sampleRate_);
    }

    // Computes pitch ratio alpha_v and formant ratio beta_v for source i
    [[nodiscard]] static float midiNoteToHz(float noteNumber) noexcept {
        return 440.0f * std::pow(2.0f, (noteNumber - 69.0f) * (1.0f / 12.0f));
    }

    void computeSourcePitchAndFormantRatios(
        float  sourceReferenceF0Hz,
        float  slotPitchSemitones,
        float  formantShiftRatio,
        float  formantKeytrack,
        float& outAlphaPitch,
        float& outBetaFormant) const noexcept
    {
        const float effectiveMidiNote = static_cast<float>(midiNote_) + smoothedPitchbendSemi_ + slotPitchSemitones;
        const float noteHz            = midiNoteToHz(effectiveMidiNote);
        const float refF0             = std::clamp(sourceReferenceF0Hz, 20.0f, 4000.0f);

        outAlphaPitch  = std::clamp(noteHz / refF0, 0.125f, 8.0f);
        outBetaFormant = std::clamp(
            formantShiftRatio * std::pow(outAlphaPitch, std::clamp(formantKeytrack, 0.0f, 1.0f)),
            0.25f,
            4.0f);
    }

    // --------------------------------------------------------------------------
    // MODE A Helper: Accumulate this voice's resampled source frames into a
    // Shared Spectral Bus frame array (weighted by voice velocity & ADSR).
    // --------------------------------------------------------------------------
    void accumulateIntoSharedSpectralBus(
        std::span<const std::shared_ptr<const SpectralSampleData>, kMaxSources> samplePool,
        const HopSynthesisParams&                                               params,
        float                                                                   formantShiftRatio,
        float                                                                   formantKeytrack,
        float                                                                   timeStretchRatio,
        std::span<SpectralFrameSoA, kMaxSources>                                sharedBusAccum,
        std::array<bool, kMaxSources>&                                          slotHasSignal) noexcept
    {
        if (!active_) {
            return;
        }

        // Advance ADSR & MPE smoothers over 1 hop (512 samples)
        float envGain = 0.0f;
        for (std::size_t n = 0; n < kDefaultHopSize; ++n) {
            smoothMpeControllers();
            envGain = adsr_.getNextSample();
        }
        if (!adsr_.isActive()) {
            active_   = false;
            midiNote_ = -1;
            return;
        }

        const float voiceGain = envGain * (0.35f + 0.65f * velocity_) * (1.0f + 0.25f * smoothedPressureZ_);

        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const auto& sample = samplePool[i];
            if (!sample || !sample->isValid()) {
                continue;
            }

            const auto& srcFrame = sample->getFrameWrapped(
                static_cast<std::size_t>(sourcePlayheadFrames_[i]),
                params.slots[i].loopStartNorm,
                params.slots[i].loopEndNorm);

            float alphaPitch   = 1.0f;
            float betaFormant  = 1.0f;
            computeSourcePitchAndFormantRatios(
                sample->referenceF0Hz,
                params.slots[i].pitchSemitones,
                formantShiftRatio,
                formantKeytrack,
                alphaPitch,
                betaFormant);

            CatmullRomResampler::resampleFrameSourceFilter(
                srcFrame,
                voiceResampledFrames_[i],
                alphaPitch,
                betaFormant,
                true);

            auto& busFrame = sharedBusAccum[i];
            const auto& vFrame = voiceResampledFrames_[i];
            for (std::size_t k = 0; k < kNumBinsPadded; ++k) {
                busFrame.env[k]   += voiceGain * vFrame.env[k];
                busFrame.fineH[k] += voiceGain * vFrame.fineH[k];
                busFrame.fineP[k] += voiceGain * vFrame.fineP[k];
                busFrame.fineR[k] += voiceGain * vFrame.fineR[k];
                if (!slotHasSignal[i]) {
                    busFrame.phase[k]      = vFrame.phase[k];
                    busFrame.instFreq[k]   = vFrame.instFreq[k];
                    busFrame.groupDelay[k] = vFrame.groupDelay[k];
                }
            }
            busFrame.rmsEnergy += voiceGain * vFrame.rmsEnergy;
            slotHasSignal[i]    = true;

            sourcePlayheadFrames_[i] += std::clamp(timeStretchRatio, 0.05f, 4.0f);
        }
    }

    // --------------------------------------------------------------------------
    // MODE B: Render 1 Sample of True Per-Voice Chimeric ISTFT (Phase-Staggered)
    // --------------------------------------------------------------------------
    void processTrueMpeSample(
        std::span<const std::shared_ptr<const SpectralSampleData>, kMaxSources> samplePool,
        const MaskMatrix2D&                                                     baseMasks,
        const HopSynthesisParams&                                               baseParams,
        float                                                                   formantShiftRatio,
        float                                                                   formantKeytrack,
        float                                                                   timeStretchRatio,
        float&                                                                  accumOutL,
        float&                                                                  accumOutR,
        SpscRingBuffer<GpuColumnPacket>*                                        gpuQueue = nullptr) noexcept
    {
        if (!active_) {
            return;
        }

        smoothMpeControllers();
        const float envGain = adsr_.getNextSample();
        if (!adsr_.isActive()) {
            active_   = false;
            midiNote_ = -1;
            return;
        }

        // At the beginning of this voice's staggered hop (counter == 0),
        // resample each active sample slot with this voice's MPE Pitchbend X
        // and apply MPE Slide Y + Pressure Z to the synthesis parameters.
        if (pipeline_.getHopSampleCounter() == 0) {
            for (std::size_t i = 0; i < kMaxSources; ++i) {
                const auto& sample = samplePool[i];
                if (!sample || !sample->isValid()) {
                    activeFramePtrs_[i] = nullptr;
                    continue;
                }

                const auto& srcFrame = sample->getFrameWrapped(
                    static_cast<std::size_t>(sourcePlayheadFrames_[i]),
                    baseParams.slots[i].loopStartNorm,
                    baseParams.slots[i].loopEndNorm);

                float alphaPitch  = 1.0f;
                float betaFormant = 1.0f;
                computeSourcePitchAndFormantRatios(
                    sample->referenceF0Hz,
                    baseParams.slots[i].pitchSemitones,
                    formantShiftRatio,
                    formantKeytrack,
                    alphaPitch,
                    betaFormant);

                CatmullRomResampler::resampleFrameSourceFilter(
                    srcFrame,
                    voiceResampledFrames_[i],
                    alphaPitch,
                    betaFormant,
                    true);

                activeFramePtrs_[i] = &voiceResampledFrames_[i];
                sourcePlayheadFrames_[i] += std::clamp(timeStretchRatio, 0.05f, 4.0f);
            }

            // Apply MPE Slide Y (spectral tilt / branch morph shift across bins)
            applyMpeSlideYToMasks(baseMasks, voiceMasks_, smoothedSlideY_);
        }

        // Modulate Cross-Synthesis exponent gamma by MPE Pressure Z
        HopSynthesisParams voiceParams = baseParams;
        voiceParams.crossSynthGamma = std::clamp(
            baseParams.crossSynthGamma + 0.35f * (smoothedPressureZ_ - 0.5f),
            0.0f,
            1.0f);

        float rawL = 0.0f;
        float rawR = 0.0f;
        pipeline_.processSingleSampleStep(
            activeFramePtrs_,
            voiceMasks_,
            voiceParams,
            0.0f,
            rawL,
            rawR,
            gpuQueue);

        const float velAmp = envGain * (0.30f + 0.70f * velocity_) * (1.0f + 0.25f * smoothedPressureZ_);
        accumOutL += rawL * velAmp;
        accumOutR += rawR * velAmp;
    }

    [[nodiscard]] bool        isActive() const noexcept { return active_ && adsr_.isActive(); }
    [[nodiscard]] bool        isReleasing() const noexcept { return active_ && adsr_.isReleasing(); }
    [[nodiscard]] int         getMidiNote() const noexcept { return midiNote_; }
    [[nodiscard]] int         getMidiChannel() const noexcept { return midiChannel_; }
    [[nodiscard]] uint64_t    getNoteOnTimestamp() const noexcept { return noteOnTimestamp_; }
    [[nodiscard]] float       getCurrentEnvelopeLevel() const noexcept { return adsr_.getCurrentValue(); }
    [[nodiscard]] std::size_t getVoiceIndex() const noexcept { return voiceIndex_; }

private:
    void smoothMpeControllers() noexcept {
        constexpr float kSmoothAlpha = 0.008f;
        smoothedPitchbendSemi_ += kSmoothAlpha * (mpePitchbendSemitones_ - smoothedPitchbendSemi_);
        smoothedSlideY_        += kSmoothAlpha * (mpeSlideY_ - smoothedSlideY_);
        smoothedPressureZ_     += kSmoothAlpha * (mpePressureZ_ - smoothedPressureZ_);
    }

    static void applyMpeSlideYToMasks(
        const MaskMatrix2D& baseMasks,
        MaskMatrix2D&       outMasks,
        float               slideY) noexcept
    {
        // Slide Y shifts fine-structure allocation along the genetic rotation axis
        const float morphShift = (slideY - 0.5f) * 0.65f;
        if (std::abs(morphShift) < 1.0e-3f) {
            outMasks = baseMasks;
            return;
        }

        for (std::size_t k = 0; k < kNumBinsPadded; ++k) {
            float sumFine = 0.0f;
            for (std::size_t i = 0; i < kMaxSources; ++i) {
                const std::size_t nextSrc = (i + 1) % kMaxSources;
                const float       a       = std::abs(morphShift);
                const float       blended = (1.0f - a) * baseMasks.fineMask[i][k]
                                          + a * baseMasks.fineMask[nextSrc][k];
                outMasks.envMask[i][k]  = baseMasks.envMask[i][k];
                outMasks.fineMask[i][k] = blended;
                sumFine += blended;
            }
            const float invSum = (sumFine > kEps) ? (1.0f / sumFine) : (1.0f / static_cast<float>(kMaxSources));
            for (std::size_t i = 0; i < kMaxSources; ++i) {
                outMasks.fineMask[i][k] *= invSum;
            }
        }
    }

    std::size_t               voiceIndex_            = 0;
    double                    sampleRate_            = 48000.0;
    bool                      active_                = false;
    int                       midiNote_              = -1;
    int                       midiChannel_           = 1;
    float                     velocity_              = 0.0f;
    uint64_t                  noteOnTimestamp_       = 0;
    float                     mpePitchbendSemitones_ = 0.0f;
    float                     smoothedPitchbendSemi_ = 0.0f;
    float                     mpeSlideY_             = 0.5f;
    float                     smoothedSlideY_        = 0.5f;
    float                     mpePressureZ_          = 0.0f;
    float                     smoothedPressureZ_     = 0.0f;

    std::array<float, kMaxSources>                   sourcePlayheadFrames_{};
    std::array<SpectralFrameSoA, kMaxSources>        voiceResampledFrames_{};
    std::array<const SpectralFrameSoA*, kMaxSources> activeFramePtrs_{};
    MaskMatrix2D                                     voiceMasks_{};
    VoiceAdsrEnvelope                                adsr_{};
    StaggeredSpectralPipeline                        pipeline_;
};

// ==============================================================================
// 5. SpecTilePolyphonicVoiceManager — Coordinates Mode A (16v) & Mode B (6v MPE)
// ==============================================================================
class SpecTilePolyphonicVoiceManager {
public:
    SpecTilePolyphonicVoiceManager() noexcept {
        for (std::size_t v = 0; v < kMaxSharedBusVoices; ++v) {
            voices_[v].setVoiceIndex(v);
        }
    }

    void prepare(double sampleRate) noexcept {
        sampleRate_ = sampleRate;
        sharedBusPipeline_.reset(0);
        for (auto& voice : voices_) {
            voice.prepare(sampleRate);
        }
    }

    void resetAllVoices() noexcept {
        sharedBusPipeline_.reset(0);
        for (auto& voice : voices_) {
            voice.reset();
        }
    }

    void setPolyphonyMode(PolyphonyMode mode) noexcept {
        polyphonyMode_ = mode;
    }

    [[nodiscard]] PolyphonyMode getPolyphonyMode() const noexcept {
        return polyphonyMode_;
    }

    void setPlayMode(PlayMode mode) noexcept {
        playMode_ = mode;
    }

    [[nodiscard]] PlayMode getPlayMode() const noexcept {
        return playMode_;
    }

    void handleNoteOn(int midiChannel, int midiNote, float velocityNorm) noexcept {
        const std::size_t maxVoices = (polyphonyMode_ == PolyphonyMode::TrueMpePerVoice6)
            ? kMaxMpeVoices
            : kMaxSharedBusVoices;

        SpecTileVoice* targetVoice = findFreeOrStealableVoice(maxVoices, midiNote, midiChannel);
        if (targetVoice != nullptr) {
            targetVoice->startNote(midiNote, velocityNorm, midiChannel, ++eventCounter_);
        }
    }

    void handleNoteOff(int midiChannel, int midiNote, bool allowTailOff = true) noexcept {
        const std::size_t maxVoices = (polyphonyMode_ == PolyphonyMode::TrueMpePerVoice6)
            ? kMaxMpeVoices
            : kMaxSharedBusVoices;

        for (std::size_t v = 0; v < maxVoices; ++v) {
            if (voices_[v].isActive() && voices_[v].getMidiNote() == midiNote &&
                (polyphonyMode_ != PolyphonyMode::TrueMpePerVoice6 || voices_[v].getMidiChannel() == midiChannel))
            {
                voices_[v].stopNote(allowTailOff);
            }
        }
    }

    void handleMpePitchbend(int midiChannel, float semitones) noexcept {
        for (auto& voice : voices_) {
            if (voice.isActive() && (midiChannel == 1 || voice.getMidiChannel() == midiChannel)) {
                voice.setMpePitchbendSemitones(semitones);
            }
        }
    }

    void handleMpeSlideY(int midiChannel, float slideNorm) noexcept {
        for (auto& voice : voices_) {
            if (voice.isActive() && (midiChannel == 1 || voice.getMidiChannel() == midiChannel)) {
                voice.setMpeSlideY(slideNorm);
            }
        }
    }

    void handleMpePressureZ(int midiChannel, float pressureNorm) noexcept {
        for (auto& voice : voices_) {
            if (voice.isActive() && (midiChannel == 1 || voice.getMidiChannel() == midiChannel)) {
                voice.setMpePressureZ(pressureNorm);
            }
        }
    }

    // --------------------------------------------------------------------------
    // Render Audio Block (Dispatches Mode A Shared Bus or Mode B Staggered MPE)
    // --------------------------------------------------------------------------
    void renderBlock(
        float*                                                                  outL,
        float*                                                                  outR,
        int                                                                     numSamples,
        std::span<const std::shared_ptr<const SpectralSampleData>, kMaxSources> samplePool,
        const MaskMatrix2D&                                                     currentMasks,
        const HopSynthesisParams&                                               params,
        float                                                                   formantShiftRatio,
        float                                                                   formantKeytrack,
        float                                                                   timeStretchRatio,
        SpscRingBuffer<GpuColumnPacket>*                                        gpuQueue = nullptr) noexcept
    {
        std::fill_n(outL, numSamples, 0.0f);
        std::fill_n(outR, numSamples, 0.0f);

        // In SpectralDroneLatch mode, keep a drone voice alive if no MIDI note is held
        if (playMode_ == PlayMode::SpectralDroneLatch && getActiveVoiceCount() == 0) {
            voices_[0].startNote(60, 0.85f, 1, ++eventCounter_);
        }

        if (polyphonyMode_ == PolyphonyMode::TrueMpePerVoice6) {
            // Mode B: True Per-Voice Chimeric ISTFT with Phase-Staggered Scheduling
            for (int n = 0; n < numSamples; ++n) {
                float sampleL = 0.0f;
                float sampleR = 0.0f;
                bool  pushedGpuThisSample = false;

                for (std::size_t v = 0; v < kMaxMpeVoices; ++v) {
                    if (!voices_[v].isActive()) {
                        continue;
                    }
                    SpscRingBuffer<GpuColumnPacket>* vGpu = (!pushedGpuThisSample) ? gpuQueue : nullptr;
                    voices_[v].processTrueMpeSample(
                        samplePool,
                        currentMasks,
                        params,
                        formantShiftRatio,
                        formantKeytrack,
                        timeStretchRatio,
                        sampleL,
                        sampleR,
                        vGpu);
                    pushedGpuThisSample = true;
                }
                outL[n] = sampleL;
                outR[n] = sampleR;
            }
        } else {
            // Mode A: Shared Spectral Bus (16 Voices -> 1 Global Staggered ISTFT)
            for (int n = 0; n < numSamples; ++n) {
                if (sharedBusPipeline_.getHopSampleCounter() == 0) {
                    for (std::size_t i = 0; i < kMaxSources; ++i) {
                        sharedBusFrames_[i].clear();
                        sharedBusFramePtrs_[i] = nullptr;
                    }
                    std::array<bool, kMaxSources> slotHasSignal{};
                    slotHasSignal.fill(false);

                    for (std::size_t v = 0; v < kMaxSharedBusVoices; ++v) {
                        if (voices_[v].isActive()) {
                            voices_[v].accumulateIntoSharedSpectralBus(
                                samplePool,
                                params,
                                formantShiftRatio,
                                formantKeytrack,
                                timeStretchRatio,
                                sharedBusFrames_,
                                slotHasSignal);
                        }
                    }

                    for (std::size_t i = 0; i < kMaxSources; ++i) {
                        if (slotHasSignal[i]) {
                            // Clamp fine structures to [0, 1.05] after multi-voice sum
                            for (std::size_t k = 0; k < kNumBinsPadded; ++k) {
                                sharedBusFrames_[i].fineH[k] = std::min(sharedBusFrames_[i].fineH[k], 1.05f);
                                sharedBusFrames_[i].fineP[k] = std::min(sharedBusFrames_[i].fineP[k], 1.05f);
                                sharedBusFrames_[i].fineR[k] = std::min(sharedBusFrames_[i].fineR[k], 1.05f);
                            }
                            sharedBusFramePtrs_[i] = &sharedBusFrames_[i];
                        }
                    }
                }

                float sampleL = 0.0f;
                float sampleR = 0.0f;
                sharedBusPipeline_.processSingleSampleStep(
                    sharedBusFramePtrs_,
                    currentMasks,
                    params,
                    0.0f,
                    sampleL,
                    sampleR,
                    gpuQueue);
                outL[n] = sampleL;
                outR[n] = sampleR;
            }
        }
    }

    [[nodiscard]] std::size_t getActiveVoiceCount() const noexcept {
        std::size_t count = 0;
        for (const auto& v : voices_) {
            if (v.isActive()) {
                ++count;
            }
        }
        return count;
    }

private:
    [[nodiscard]] SpecTileVoice* findFreeOrStealableVoice(
        std::size_t maxVoices,
        int         midiNote,
        int         midiChannel) noexcept
    {
        // 1. Retrigger existing voice on same note + channel
        for (std::size_t v = 0; v < maxVoices; ++v) {
            if (voices_[v].isActive() &&
                voices_[v].getMidiNote() == midiNote &&
                voices_[v].getMidiChannel() == midiChannel)
            {
                return &voices_[v];
            }
        }

        // 2. Find an idle voice
        for (std::size_t v = 0; v < maxVoices; ++v) {
            if (!voices_[v].isActive()) {
                return &voices_[v];
            }
        }

        // 3. Steal quietest releasing voice
        SpecTileVoice* bestReleasing = nullptr;
        float          minRelLevel   = 2.0f;
        for (std::size_t v = 0; v < maxVoices; ++v) {
            if (voices_[v].isReleasing() && voices_[v].getCurrentEnvelopeLevel() < minRelLevel) {
                minRelLevel   = voices_[v].getCurrentEnvelopeLevel();
                bestReleasing = &voices_[v];
            }
        }
        if (bestReleasing != nullptr) {
            return bestReleasing;
        }

        // 4. Steal oldest voice
        SpecTileVoice* oldestVoice = &voices_[0];
        uint64_t       oldestTime  = voices_[0].getNoteOnTimestamp();
        for (std::size_t v = 1; v < maxVoices; ++v) {
            if (voices_[v].getNoteOnTimestamp() < oldestTime) {
                oldestTime  = voices_[v].getNoteOnTimestamp();
                oldestVoice = &voices_[v];
            }
        }
        return oldestVoice;
    }

    double                                           sampleRate_    = 48000.0;
    PolyphonyMode                                    polyphonyMode_ = PolyphonyMode::SharedSpectralBus16;
    PlayMode                                         playMode_      = PlayMode::PitchedInstrument;
    uint64_t                                         eventCounter_  = 0;
    std::array<SpecTileVoice, kMaxSharedBusVoices>   voices_{};
    StaggeredSpectralPipeline                        sharedBusPipeline_{0};
    std::array<SpectralFrameSoA, kMaxSources>        sharedBusFrames_{};
    std::array<const SpectralFrameSoA*, kMaxSources> sharedBusFramePtrs_{};
};

} // namespace spectile::engine
