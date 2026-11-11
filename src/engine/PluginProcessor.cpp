// ==============================================================================
// SpecTile — src/engine/PluginProcessor.cpp
// Settore 3: Implementation of SpecTileAudioProcessor, 24D Genome QR+SLERP,
//            O(B) Voronoi + Markov + Anti-Masking Rule Evaluator & Preset State
// ==============================================================================

#include "PluginProcessor.hpp"

#if SPECTILE_HAS_JUCE

#if __has_include("../ui/PluginEditor.hpp")
    #include "../ui/PluginEditor.hpp"
    #define SPECTILE_HAS_CUSTOM_EDITOR 1
#elif __has_include("ui/PluginEditor.hpp")
    #include "ui/PluginEditor.hpp"
    #define SPECTILE_HAS_CUSTOM_EDITOR 1
#else
    #define SPECTILE_HAS_CUSTOM_EDITOR 0
#endif

namespace spectile::engine {

namespace {

// ------------------------------------------------------------------------------
// 6 Calibrated Musical Archetypes in [0, 1]^24 (Section 2.2)
// Avoids the "Spectral Mud Paradox" by enforcing structured priors
// ------------------------------------------------------------------------------
constexpr std::array<std::array<float, kGenomeDim>, 6> kArchetypeCenters = {{
    // A1: Harmonic Interleave (High comb interleaving, tight temperature, strong Markov)
    {0.55f, 0.30f, 0.88f, 0.15f, 0.92f, 0.85f, 0.78f, 0.80f,
     0.25f, 0.20f, 0.90f, 0.45f, 0.70f, 0.65f, 0.30f, 0.82f,
     0.75f, 0.60f, 0.40f, 0.50f, 0.80f, 0.35f, 0.65f, 0.50f},

    // A2: Transient-Formant Graft (High onset split, high cross-synthesis, fast Markov reset)
    {0.40f, 0.45f, 0.82f, 0.18f, 0.88f, 0.75f, 0.95f, 0.35f,
     0.30f, 0.95f, 0.65f, 0.60f, 0.92f, 0.88f, 0.85f, 0.40f,
     0.30f, 0.80f, 0.75f, 0.60f, 0.45f, 0.70f, 0.55f, 0.65f},

    // A3: Voronoi Spectral Shatter (Strong 2D parabolic Voronoi tiles + dynamic drift)
    {0.25f, 0.75f, 0.78f, 0.12f, 0.65f, 0.60f, 0.85f, 0.95f,
     0.45f, 0.15f, 0.50f, 0.55f, 0.50f, 0.70f, 0.60f, 0.55f,
     0.85f, 0.40f, 0.70f, 0.80f, 0.60f, 0.50f, 0.75f, 0.40f},

    // A4: Complementarity Hole-Filler (Maximum anti-masking energy filling)
    {0.50f, 0.40f, 0.86f, 0.20f, 0.84f, 0.80f, 0.88f, 0.40f,
     0.35f, 0.30f, 0.70f, 0.75f, 0.96f, 0.80f, 0.50f, 0.65f,
     0.55f, 0.70f, 0.60f, 0.45f, 0.75f, 0.60f, 0.50f, 0.70f},

    // A5: Spectral Freeze Morph (High Perlin nebula fields + long horizontal Markov persistence)
    {0.80f, 0.55f, 0.95f, 0.32f, 0.96f, 0.92f, 0.20f, 0.45f,
     0.92f, 0.25f, 0.40f, 0.80f, 0.60f, 0.75f, 0.45f, 0.90f,
     0.65f, 0.50f, 0.85f, 0.35f, 0.70f, 0.80f, 0.40f, 0.85f},

    // A6: Formant Choir (Strong True Envelope cross-synthesis + broad ERB formant bands)
    {0.75f, 0.25f, 0.90f, 0.24f, 0.90f, 0.88f, 0.65f, 0.35f,
     0.60f, 0.55f, 0.80f, 0.85f, 0.72f, 0.95f, 0.78f, 0.85f,
     0.50f, 0.75f, 0.55f, 0.65f, 0.85f, 0.45f, 0.70f, 0.60f}
}};

[[nodiscard]] inline float logitTransform(float p) noexcept {
    const float clamped = std::clamp(p, 0.01f, 0.99f);
    return std::log(clamped / (1.0f - clamped));
}

[[nodiscard]] inline float sigmoidTransform(float z) noexcept {
    return 1.0f / (1.0f + std::exp(-std::clamp(z, -12.0f, 12.0f)));
}

// Fast deterministic 2D Simplex/Value noise for Family 1 organic spectral islands
[[nodiscard]] inline float fastSpectralNoise2D(float x, float y, uint32_t seed) noexcept {
    const int xi = static_cast<int>(std::floor(x));
    const int yi = static_cast<int>(std::floor(y));
    const float xf = x - static_cast<float>(xi);
    const float yf = y - static_cast<float>(yi);

    const float u = xf * xf * (3.0f - 2.0f * xf);
    const float v = yf * yf * (3.0f - 2.0f * yf);

    auto hashCorner = [seed](int cx, int cy) noexcept -> float {
        uint32_t h = static_cast<uint32_t>(cx) * 374761393u
                   + static_cast<uint32_t>(cy) * 668265263u
                   + seed * 1442695041u;
        h = (h ^ (h >> 13)) * 1274126177u;
        return static_cast<float>(h & 0x00FFFFFFu) * (2.0f / 16777215.0f) - 1.0f;
    };

    const float n00 = hashCorner(xi, yi);
    const float n10 = hashCorner(xi + 1, yi);
    const float n01 = hashCorner(xi, yi + 1);
    const float n11 = hashCorner(xi + 1, yi + 1);

    const float nx0 = n00 + u * (n10 - n00);
    const float nx1 = n01 + u * (n11 - n01);
    return nx0 + v * (nx1 - nx0);
}

} // namespace

// ==============================================================================
// 1. Constructor & Destructor
// ==============================================================================
SpecTileAudioProcessor::SpecTileAudioProcessor()
    : juce::AudioProcessor(
          BusesProperties()
              .withInput("Sidechain / Live Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Main Output", juce::AudioChannelSet::stereo(), true))
    , apvts_(*this, nullptr, "SpecTileState", SpecTileParameterLayout::createParameterLayout())
    , sampleRcuBank_(sampleGc_)
    , workerPool_(sampleRcuBank_, 2)
{
    SpecTileParameterLayout::bindAtomicPointers(apvts_, paramCache_);
    formatManager_.registerBasicFormats();

    initializeErbProjectionTables(48000.0);

    // Initialize uniform masks and Markov state in-place without stack temporaries
    maskTripleBuffer_.writeBuffer().setUniform(kMaxSources);
    maskTripleBuffer_.publishWrite();

    for (std::size_t i = 0; i < kMaxSources; ++i) {
        prevErbMasks_[i].fill(1.0f / static_cast<float>(kMaxSources));
    }

    // Initialize genome from Archetype 0 (Harmonic Interleave) & build 6 QR branches
    triggerGenomeMutation(0, 0x5350544Cu);

    // Start 10 Hz MessageThread timer for lock-free RCU garbage collection
    startTimerHz(10);
}

SpecTileAudioProcessor::~SpecTileAudioProcessor() {
    stopTimer();
    workerPool_.shutdown();
    for (auto& slotPtr : audioThreadSampleSnapshot_) {
        slotPtr.reset();
    }
    sampleGc_.clearAllBlocking();
}

// ==============================================================================
// 2. AudioProcessor Lifecycle & Bus Configuration
// ==============================================================================
void SpecTileAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    (void)samplesPerBlock;
    currentSampleRate_ = (sampleRate > 1000.0) ? sampleRate : 48000.0;
    initializeErbProjectionTables(currentSampleRate_);
    voiceManager_.prepare(currentSampleRate_);
    liveSidechainRing_.fill(0.0f);
    liveSidechainWritePos_ = 0;

    // Report STFT overlap latency (M - R = 4096 - 512 = 3584 samples)
    setLatencySamples(static_cast<int>(kDefaultFftSize - kDefaultHopSize));
}

void SpecTileAudioProcessor::releaseResources() {
    voiceManager_.resetAllVoices();
}

bool SpecTileAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& mainOut = layouts.getMainOutputChannelSet();
    if (mainOut != juce::AudioChannelSet::stereo()) {
        return false;
    }
    const auto& mainIn = layouts.getMainInputChannelSet();
    return mainIn.isDisabled()
        || mainIn == juce::AudioChannelSet::mono()
        || mainIn == juce::AudioChannelSet::stereo();
}

// ==============================================================================
// 3. Real-Time Audio Callback (Zero Allocations, Zero Locks, Denormal-Safe)
// ==============================================================================
void SpecTileAudioProcessor::processBlock(
    juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer&         midiMessages)
{
    juce::ScopedNoDenormals noDenormals;

    const int numSamples    = buffer.getNumSamples();
    const int numOutChans   = getTotalNumOutputChannels();
    const int numInChans    = getTotalNumInputChannels();

    if (numSamples <= 0 || numOutChans < 2) {
        return;
    }

    // 1. Capture live sidechain input if present (for PlayMode::LiveSidechainHybrid)
    if (numInChans > 0) {
        const float* inL = buffer.getReadPointer(0);
        const float* inR = (numInChans > 1) ? buffer.getReadPointer(1) : inL;
        for (int n = 0; n < numSamples; ++n) {
            liveSidechainRing_[liveSidechainWritePos_ & (kDefaultFftSize - 1)] = 0.5f * (inL[n] + inR[n]);
            ++liveSidechainWritePos_;
        }
    }

    // 2. Read wait-free atomic parameters (O(1))
    const HopSynthesisParams hopParams = paramCache_.readHopSynthesisParams(currentSampleRate_);

    const int polyModeInt = static_cast<int>(std::lrintf(
        AtomicParameterCache::loadOrDefault(paramCache_.polyphonyMode, 0.0f)));
    voiceManager_.setPolyphonyMode(
        (polyModeInt == 1) ? PolyphonyMode::TrueMpePerVoice6 : PolyphonyMode::SharedSpectralBus16);

    const int playModeInt = std::clamp(
        static_cast<int>(std::lrintf(AtomicParameterCache::loadOrDefault(paramCache_.playMode, 0.0f))),
        0, 3);
    voiceManager_.setPlayMode(static_cast<PlayMode>(playModeInt));

    // Evaluate continuous QR+SLERP branch morph if user scrubbed along a branch
    const float branchRho   = AtomicParameterCache::loadOrDefault(paramCache_.branchMorph, 0.0f);
    const float branchAngle = AtomicParameterCache::loadOrDefault(paramCache_.branchAngle, 0.0f);
    if (branchRho > 1.0e-4f) {
        evaluateBranchMorph(branchRho, branchAngle);
    }

    // 3. Acquire lock-free RCU snapshot of the 8 sample slots
    sampleRcuBank_.acquireSnapshotForAudioThread(audioThreadSampleSnapshot_);

    // 4. Parse MIDI & MPE Expressions (NoteOn/Off, Pitchbend X, CC74 Slide Y, Pressure Z)
    for (const auto metadata : midiMessages) {
        const auto msg = metadata.getMessage();
        const int  ch  = msg.getChannel();

        if (msg.isNoteOn()) {
            voiceManager_.handleNoteOn(ch, msg.getNoteNumber(), msg.getFloatVelocity());
        } else if (msg.isNoteOff()) {
            voiceManager_.handleNoteOff(ch, msg.getNoteNumber(), true);
        } else if (msg.isAllNotesOff() || msg.isAllSoundOff()) {
            voiceManager_.resetAllVoices();
        } else if (msg.isPitchWheel()) {
            const float normWheel = (static_cast<float>(msg.getPitchWheelValue()) - 8192.0f) / 8192.0f;
            const float pitchbendRange = (voiceManager_.getPolyphonyMode() == PolyphonyMode::TrueMpePerVoice6)
                ? 48.0f
                : 2.0f;
            voiceManager_.handleMpePitchbend(ch, normWheel * pitchbendRange);
        } else if (msg.isController() && msg.getControllerNumber() == 74) {
            // MPE Timbre / Slide Y (CC 74)
            const float slideNorm = static_cast<float>(msg.getControllerValue()) / 127.0f;
            voiceManager_.handleMpeSlideY(ch, slideNorm);
        } else if (msg.isChannelPressure()) {
            // MPE Pressure Z (Channel Aftertouch)
            const float pressNorm = static_cast<float>(msg.getChannelPressureValue()) / 127.0f;
            voiceManager_.handleMpePressureZ(ch, pressNorm);
        }
    }

    // 5. Evaluate Generative Rule Families on B=64 ERB bands & upsample to K=2049
    evaluateRealTimeRuleMasks(audioThreadSampleSnapshot_, hopParams, globalHopCounter_++);
    const MaskMatrix2D& currentMasks = maskTripleBuffer_.readLatest();

    // 6. Render Polyphonic / MPE Audio Block via Staggered Spectral Pipeline
    const float formantShift    = AtomicParameterCache::loadOrDefault(paramCache_.formantShift, 1.0f);
    const float formantKeytrack = AtomicParameterCache::loadOrDefault(paramCache_.formantKeytrack, 0.15f);
    const float timeStretch     = AtomicParameterCache::loadOrDefault(paramCache_.timeStretch, 1.0f);

    float* outL = buffer.getWritePointer(0);
    float* outR = buffer.getWritePointer(1);

    voiceManager_.renderBlock(
        outL,
        outR,
        numSamples,
        audioThreadSampleSnapshot_,
        currentMasks,
        hopParams,
        formantShift,
        formantKeytrack,
        timeStretch,
        &gpuTelemetryQueue_);

    // Clear any extra output channels
    for (int ch = 2; ch < numOutChans; ++ch) {
        buffer.clear(ch, 0, numSamples);
    }

    // Apply master output gain
    const float masterDb   = AtomicParameterCache::loadOrDefault(paramCache_.masterGainDb, 0.0f);
    const float masterGain = std::pow(10.0f, masterDb * 0.05f);
    if (std::abs(masterGain - 1.0f) > 1.0e-4f) {
        buffer.applyGain(0, numSamples, masterGain);
    }
}

// ==============================================================================
// 4. Psychoacoustic ERB -> FFT Raised-Cosine Projection & Rule Evaluation
// ==============================================================================
void SpecTileAudioProcessor::initializeErbProjectionTables(double sampleRate) noexcept {
    // Glasberg & Moore (1990): E_ERB(f) = 21.4 * log10(0.00437 * f + 1)
    const float nyquistHz = static_cast<float>(sampleRate) * 0.5f;
    const float maxErbCam = 21.4f * std::log10(0.00437f * nyquistHz + 1.0f);
    const float binHz     = static_cast<float>(sampleRate) / static_cast<float>(kDefaultFftSize);

    for (std::size_t k = 0; k < kNumBinsPadded; ++k) {
        const float fHz    = static_cast<float>(std::min(k, kNumBins - 1)) * binHz;
        const float erbCam = 21.4f * std::log10(0.00437f * fHz + 1.0f);
        const float xi     = std::clamp(
            (erbCam / std::max(maxErbCam, kEps)) * static_cast<float>(kNumErbBands - 1),
            0.0f,
            static_cast<float>(kNumErbBands - 1));

        const uint16_t b0    = static_cast<uint16_t>(std::floor(xi));
        const uint16_t b1    = std::min<uint16_t>(b0 + 1, static_cast<uint16_t>(kNumErbBands - 1));
        const float    alpha = xi - static_cast<float>(b0);

        erbLowerBandIdx_[k]    = b0;
        erbUpperBandIdx_[k]    = b1;
        // Raised-cosine C^1 weight w_up(k) = 0.5 * (1 - cos(pi * alpha)) (Eq. 1.2)
        erbRaisedCosWeight_[k] = 0.5f * (1.0f - std::cos(kPi * alpha));
    }
}

void SpecTileAudioProcessor::evaluateRealTimeRuleMasks(
    const std::array<std::shared_ptr<const SpectralSampleData>, kMaxSources>& sampleSnapshot,
    const HopSynthesisParams&                                                 hopParams,
    uint64_t                                                                  hopFrameIdx) noexcept
{
    // Read Macro & Rule parameters
    const float macroTileSize    = AtomicParameterCache::loadOrDefault(paramCache_.macroTileSize, 0.45f);
    const float macroOrderMutate = AtomicParameterCache::loadOrDefault(paramCache_.macroOrderMutate, 0.35f);
    const float macroPhaseGlue   = AtomicParameterCache::loadOrDefault(paramCache_.macroPhaseGlue, 0.85f);

    const float tauTemp       = std::clamp(AtomicParameterCache::loadOrDefault(paramCache_.softmaxTemp, 0.22f), 0.03f, 1.50f);
    const float sparsityGate  = std::clamp(AtomicParameterCache::loadOrDefault(paramCache_.sparsityGate, 0.35f), 0.0f, 0.95f);
    const float markovStayT   = std::clamp(AtomicParameterCache::loadOrDefault(paramCache_.markovStayTime, 0.82f), 0.0f, 0.99f);
    const float markovStayF   = std::clamp(AtomicParameterCache::loadOrDefault(paramCache_.markovStayFreq, 0.75f), 0.0f, 0.99f);
    const float transReset    = std::clamp(AtomicParameterCache::loadOrDefault(paramCache_.transientReset, 0.90f), 0.0f, 1.0f);
    const float wVoronoi      = AtomicParameterCache::loadOrDefault(paramCache_.voronoiWeight, 0.60f);
    const float wPerlin       = AtomicParameterCache::loadOrDefault(paramCache_.perlinWeight, 0.40f);
    const float wComb         = AtomicParameterCache::loadOrDefault(paramCache_.combWeight, 0.30f);
    const float lambdaAnti    = AtomicParameterCache::loadOrDefault(paramCache_.antiMasking, 0.55f);
    const float chiCross      = AtomicParameterCache::loadOrDefault(paramCache_.crossSynthPerm, 0.50f);

    const float tSec    = static_cast<float>(hopFrameIdx * kDefaultHopSize) / static_cast<float>(currentSampleRate_);
    const float freqScl = 0.04f + 0.35f * (1.0f - macroTileSize);
    const float timeScl = 0.35f + 2.50f * (1.0f - macroTileSize);

    // 1. Evaluate Logits L_i(t, b) across N = 8 sources and B = 64 ERB bands (Section 2.1)
    std::array<std::array<float, kNumErbBands>, kMaxSources> logits{};
    std::array<std::array<float, kNumErbBands>, kMaxSources> erbMasksFine{};
    std::array<std::array<float, kNumErbBands>, kMaxSources> erbMasksEnv{};

    bool anySampleLoaded = false;
    for (std::size_t i = 0; i < kMaxSources; ++i) {
        if (sampleSnapshot[i] && sampleSnapshot[i]->isValid()) {
            anySampleLoaded = true;
            break;
        }
    }

    for (std::size_t b = 0; b < kNumErbBands; ++b) {
        const float bNorm = static_cast<float>(b) / static_cast<float>(kNumErbBands - 1);

        // Compute aggregate energy across loaded samples for Anti-Masking Hole-Filler
        float totalBandEnergy = 0.0f;
        float maxBandFlux     = 0.0f;
        std::array<float, kMaxSources> srcEnergyNorm{};
        std::array<float, kMaxSources> srcFlatness{};
        std::array<float, kMaxSources> srcFlux{};

        for (std::size_t i = 0; i < kMaxSources; ++i) {
            if (sampleSnapshot[i] && sampleSnapshot[i]->isValid()) {
                const auto& frame = sampleSnapshot[i]->getFrameWrapped(
                    static_cast<std::size_t>(hopFrameIdx),
                    hopParams.slots[i].loopStartNorm,
                    hopParams.slots[i].loopEndNorm);
                const float eLin = std::pow(10.0f, std::clamp(frame.erbEnergy[b], -80.0f, 20.0f) * 0.05f);
                srcEnergyNorm[i] = eLin;
                srcFlatness[i]   = frame.erbFlatness[b];
                srcFlux[i]       = frame.erbFlux[b];
                totalBandEnergy += eLin;
                maxBandFlux      = std::max(maxBandFlux, frame.erbFlux[b]);
            } else {
                srcEnergyNorm[i] = 0.5f;
                srcFlatness[i]   = 0.25f;
                srcFlux[i]       = 0.0f;
                totalBandEnergy += 0.5f;
            }
        }

        // Onset sigmoid Omega_onset(t, b) for Transient Markov Reset & Onset/Continuant split
        const float omegaOnset = 1.0f / (1.0f + std::exp(-6.0f * (maxBandFlux - 0.18f)));

        float logitMean = 0.0f;
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const float slotW = (!anySampleLoaded || (sampleSnapshot[i] && sampleSnapshot[i]->isValid()))
                ? hopParams.slots[i].weight
                : 0.0f;

            if (slotW <= 1.0e-5f) {
                logits[i][b] = -20.0f;
                continue;
            }

            // Family 1: Geometric Tiling (1D Parabolic Voronoi + 2D Perlin + von Mises Comb)
            const float seedCenterB = std::fmod(
                activeGenome_[i % kGenomeDim] * static_cast<float>(kNumErbBands)
                + 9.0f * std::sin(0.7f * tSec * timeScl + static_cast<float>(i) * 1.1f),
                static_cast<float>(kNumErbBands));
            const float dB = (static_cast<float>(b) - seedCenterB) * freqScl;
            const float phiVoronoi = -(dB * dB);

            const float phiPerlin = fastSpectralNoise2D(
                tSec * timeScl + static_cast<float>(i) * 3.7f,
                static_cast<float>(b) * freqScl,
                static_cast<uint32_t>(i + 1));

            const float combPhase = kTwoPi * (static_cast<float>(b) - static_cast<float>(i)) / static_cast<float>(kMaxSources);
            const float phiComb   = std::exp(1.8f * std::cos(combPhase)) - 1.0f;

            const float phiGeo = wVoronoi * phiVoronoi + wPerlin * phiPerlin + wComb * phiComb;

            // Family 2: Descriptor-Driven Competition & Anti-Masking Hole-Filler (Eq. 2.1.2)
            const float otherEnergy = std::max(0.0f, totalBandEnergy - srcEnergyNorm[i]) / static_cast<float>(kMaxSources - 1);
            const float phiComp     = srcEnergyNorm[i] * (1.0f - lambdaAnti * otherEnergy)
                                    - lambdaAnti * otherEnergy * srcEnergyNorm[i];
            const float phiTonal    = (1.0f - srcFlatness[i]) * (1.0f - omegaOnset) + srcFlux[i] * omegaOnset;
            const float phiDesc     = 0.55f * phiComp + 0.45f * phiTonal;

            // Family 3: Causal 2D Markov Continuity with Transient Markov Reset (Eq. 2.1.3)
            const float bPrevMask = (b > 0) ? erbMasksFine[i][b - 1] : prevErbMasks_[i][b];
            const float bUpMask   = (b + 1 < kNumErbBands) ? prevErbMasks_[i][b + 1] : prevErbMasks_[i][b];
            const float markovMem = (1.0f - transReset * omegaOnset)
                                  * (markovStayT * prevErbMasks_[i][b] + 0.5f * markovStayF * (bPrevMask + bUpMask));

            const float alphaGeo  = 0.35f + 0.65f * macroOrderMutate;
            const float alphaDesc = 0.35f + 0.65f * (1.0f - macroOrderMutate);

            const float rawLogit = std::log(slotW + kEps)
                                 + alphaGeo * phiGeo
                                 + alphaDesc * phiDesc
                                 + 1.25f * markovMem
                                 + 0.25f * (activeGenome_[(i * 3 + (b & 7u)) % kGenomeDim] - 0.5f) * (1.0f - bNorm);

            logits[i][b] = rawLogit;
            logitMean   += rawLogit;
        }
        logitMean /= static_cast<float>(kMaxSources);

        // Temperature-scaled Softmax + Spectral Sparsity Gating
        float maxLogit = -100.0f;
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            logits[i][b] -= logitMean;
            maxLogit = std::max(maxLogit, logits[i][b]);
        }

        float sumFine = 0.0f;
        const float gateThresh = sparsityGate * (0.65f / static_cast<float>(kMaxSources));
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            float p = std::exp(std::clamp((logits[i][b] - maxLogit) / tauTemp, -20.0f, 0.0f));
            if (p < gateThresh) {
                p = 0.0f;
            }
            erbMasksFine[i][b] = p;
            sumFine += p;
        }

        // Enforce exact Partition of Unity sum_i M_i(t, b) = 1
        if (sumFine > kEps) {
            const float invSum = 1.0f / sumFine;
            for (std::size_t i = 0; i < kMaxSources; ++i) {
                erbMasksFine[i][b] *= invSum;
            }
        } else {
            erbMasksFine[0][b] = 1.0f;
        }

        // Family 4: Source-Filter Cross-Synthesis Circular Permutation Pi_Delta
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const std::size_t shiftedIdx = (i + 1) % kMaxSources;
            erbMasksEnv[i][b] = (1.0f - chiCross) * erbMasksFine[i][b]
                              + chiCross * erbMasksFine[shiftedIdx][b];
            prevErbMasks_[i][b] = erbMasksFine[i][b];
        }
    }

    // 2. Upsample B = 64 ERB bands -> K = 2049 FFT bins via Raised-Cosine (Eq. 1.2)
    //    and apply 2D Spectral Feathering while preserving sum_i M_i(t, k) == 1.0f
    MaskMatrix2D& outMasks = maskTripleBuffer_.writeBuffer();
    outMasks.frameSequenceId = hopFrameIdx;

    for (std::size_t k = 0; k < kNumBinsPadded; ++k) {
        const uint16_t b0  = erbLowerBandIdx_[k];
        const uint16_t b1  = erbUpperBandIdx_[k];
        const float    wUp = erbRaisedCosWeight_[k];
        const float    wLo = 1.0f - wUp;

        for (std::size_t i = 0; i < kMaxSources; ++i) {
            outMasks.fineMask[i][k] = wLo * erbMasksFine[i][b0] + wUp * erbMasksFine[i][b1];
            outMasks.envMask[i][k]  = wLo * erbMasksEnv[i][b0]  + wUp * erbMasksEnv[i][b1];
        }
    }

    // Apply separable 3-tap Spectral Feathering kernel controlled by Macro Phase Glue
    if (macroPhaseGlue > 0.05f) {
        const float featherAlpha = 0.25f * macroPhaseGlue;
        const float centerWeight = 1.0f - 2.0f * featherAlpha;
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            auto& fRow = outMasks.fineMask[i];
            auto& eRow = outMasks.envMask[i];
            float prevF = fRow[0];
            float prevE = eRow[0];
            for (std::size_t k = 1; k + 1 < kNumBins; ++k) {
                const float currF = fRow[k];
                const float currE = eRow[k];
                fRow[k] = centerWeight * currF + featherAlpha * (prevF + fRow[k + 1]);
                eRow[k] = centerWeight * currE + featherAlpha * (prevE + eRow[k + 1]);
                prevF = currF;
                prevE = currE;
            }
        }
    }

    maskTripleBuffer_.publishWrite();
}

// ==============================================================================
// 5. 24D Spectral Genome & 6 QR-Orthonormalized Evolutionary Branches (Sec 2.2)
// ==============================================================================
void SpecTileAudioProcessor::triggerGenomeMutation(int archetypeIndex, uint32_t randomSeed) noexcept {
    if (randomSeed != 0) {
        prngState_ = randomSeed;
    } else {
        prngState_ = prngState_ * 1664525u + 1013904223u;
    }

    const int archIdx = (archetypeIndex >= 0 && archetypeIndex < 6)
        ? archetypeIndex
        : static_cast<int>(prngState_ % 6u);

    const auto& center = kArchetypeCenters[static_cast<std::size_t>(archIdx)];
    for (std::size_t d = 0; d < kGenomeDim; ++d) {
        prngState_ = prngState_ * 1664525u + 1013904223u;
        const float u = static_cast<float>(prngState_ & 0x00FFFFFFu) / 16777215.0f;
        const float perturbation = (u - 0.5f) * 0.14f;
        baseGenome_[d]   = std::clamp(center[d] + perturbation, 0.02f, 0.98f);
        baseLogitsZ0_[d] = logitTransform(baseGenome_[d]);
        activeGenome_[d] = baseGenome_[d];
    }

    buildOrthonormalEvolutionaryBranches();
}

void SpecTileAudioProcessor::buildOrthonormalEvolutionaryBranches() noexcept {
    // Construct 6 semantically distinct tangent basis vectors in R^24 and
    // orthonormalize via Modified Gram-Schmidt QR Factorization: U^T * U = I_6
    for (std::size_t b = 0; b < kNumBranches; ++b) {
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            const float angle = kTwoPi * static_cast<float>((b + 1) * (d + 1)) / static_cast<float>(kGenomeDim);
            qrBranchVectors_[b][d] = std::cos(angle + static_cast<float>(b) * 0.52f);
        }

        // Gram-Schmidt orthogonalization against previous branches 0..b-1
        for (std::size_t prev = 0; prev < b; ++prev) {
            float dot = 0.0f;
            for (std::size_t d = 0; d < kGenomeDim; ++d) {
                dot += qrBranchVectors_[b][d] * qrBranchVectors_[prev][d];
            }
            for (std::size_t d = 0; d < kGenomeDim; ++d) {
                qrBranchVectors_[b][d] -= dot * qrBranchVectors_[prev][d];
            }
        }

        // L2 Normalization
        float normSq = 0.0f;
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            normSq += qrBranchVectors_[b][d] * qrBranchVectors_[b][d];
        }
        const float invNorm = (normSq > kEps) ? (1.0f / std::sqrt(normSq)) : 1.0f;
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            qrBranchVectors_[b][d] *= invNorm;
        }
    }
}

void SpecTileAudioProcessor::evaluateBranchMorph(float rhoRadius, float thetaRadians) noexcept {
    // Logit-Riemannian SLERP interpolation (Eq. 2.2.1):
    // g(rho, theta) = sigma(z_0 + rho * R_branch * u(theta))
    const float rho = std::clamp(rhoRadius, 0.0f, 1.0f);
    const float wrappedTheta = std::fmod(std::max(0.0f, thetaRadians), kTwoPi);
    const float sector       = (wrappedTheta / kTwoPi) * static_cast<float>(kNumBranches);
    const std::size_t b0     = static_cast<std::size_t>(sector) % kNumBranches;
    const std::size_t b1     = (b0 + 1) % kNumBranches;
    const float frac         = sector - std::floor(sector);

    // Spherical interpolation between adjacent orthogonal branch vectors u_{b0}, u_{b1}
    const float cosOmega = std::cos(0.5f * kPi * frac);
    const float sinOmega = std::sin(0.5f * kPi * frac);
    constexpr float kBranchRadius = 3.2f;

    for (std::size_t d = 0; d < kGenomeDim; ++d) {
        const float uDir = cosOmega * qrBranchVectors_[b0][d] + sinOmega * qrBranchVectors_[b1][d];
        const float z    = baseLogitsZ0_[d] + rho * kBranchRadius * uDir;
        activeGenome_[d] = sigmoidTransform(z);
    }
}

// ==============================================================================
// 6. Asynchronous Sample Loading & 10 Hz RCU Garbage Collection Timer
// ==============================================================================
void SpecTileAudioProcessor::timerCallback() {
    // Reclaim retired sample buffers safely on the MessageThread when use_count() == 1
    sampleGc_.collectUnusedOnMessageThread();
}

std::string SpecTileAudioProcessor::computeFileSha256Hex(const juce::File& file) {
    if (!file.existsAsFile()) {
        return {};
    }
    juce::SHA256 hasher(file);
    return hasher.toHexString().toStdString();
}

bool SpecTileAudioProcessor::loadSampleFileIntoSlot(std::size_t slotIndex, const juce::File& audioFile) {
    if (slotIndex >= kMaxSources || !audioFile.existsAsFile()) {
        return false;
    }

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager_.createReaderFor(audioFile));
    if (!reader || reader->lengthInSamples <= 0) {
        return false;
    }

    // Cap maximum sample length to 30 seconds to respect RAM budget
    const int maxSamples = static_cast<int>( std::min<int64_t>(
        reader->lengthInSamples,
        static_cast<int64_t>(reader->sampleRate * 30.0)) );

    juce::AudioBuffer<float> tempBuffer(static_cast<int>(reader->numChannels), maxSamples);
    reader->read(&tempBuffer, 0, maxSamples, 0, true, true);

    std::vector<float> monoSamples(static_cast<std::size_t>(maxSamples), 0.0f);
    const int numCh = tempBuffer.getNumChannels();
    const float chScale = 1.0f / static_cast<float>(std::max(1, numCh));
    for (int ch = 0; ch < numCh; ++ch) {
        const float* src = tempBuffer.getReadPointer(ch);
        for (int n = 0; n < maxSamples; ++n) {
            monoSamples[static_cast<std::size_t>(n)] += src[n] * chScale;
        }
    }

    const std::string sha256 = computeFileSha256Hex(audioFile);
    loadRawMonoBufferIntoSlot(
        slotIndex,
        monoSamples,
        reader->sampleRate,
        audioFile.getFileName().toStdString(),
        audioFile.getFullPathName().toStdString(),
        sha256);
    return true;
}

void SpecTileAudioProcessor::loadRawMonoBufferIntoSlot(
    std::size_t               slotIndex,
    const std::vector<float>& audioMono,
    double                    sampleRate,
    const std::string&        displayName,
    const std::string&        filePath,
    const std::string&        sha256Hash)
{
    if (slotIndex >= kMaxSources || audioMono.empty()) {
        return;
    }

    SampleAnalysisWorkerPool::AnalysisJob job{};
    job.slotIndex  = slotIndex;
    job.audioMono  = audioMono;
    job.sampleRate = sampleRate;
    job.fileName   = displayName;
    job.filePath   = filePath;
    job.sha256Hash = sha256Hash;

    workerPool_.enqueueAnalysisJob(std::move(job));
}

void SpecTileAudioProcessor::addSampleSearchDirectory(const juce::File& directory) {
    if (directory.isDirectory()) {
        userSampleSearchDirs_.push_back(directory);
    }
}

// ==============================================================================
// 7. State & Preset Serialization (XML + JSON + Triple Sample Fallback)
// ==============================================================================
void SpecTileAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto stateTree = apvts_.copyState();
    std::unique_ptr<juce::XmlElement> rootXml(stateTree.createXml());
    if (!rootXml) {
        rootXml = std::make_unique<juce::XmlElement>("SpecTileState");
    }

    // 1. Serialize 24D Spectral Genome g_0
    auto* genomeXml = rootXml->createNewChildElement("SpectralGenome24D");
    for (std::size_t d = 0; d < kGenomeDim; ++d) {
        genomeXml->setAttribute("g" + juce::String(static_cast<int>(d)), static_cast<double>(baseGenome_[d]));
    }

    // 2. Serialize 8 Sample Slots with Triple Fallback (relativePath, sha256Hash, flacBase64)
    auto* slotsXml = rootXml->createNewChildElement("SampleSlots");
    const bool embedFlac = embedSamplesInPreset_.load();
    const juce::File baseDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);

    for (std::size_t i = 0; i < kMaxSources; ++i) {
        const auto slotData = sampleRcuBank_.getSlot(i);
        if (!slotData || !slotData->isValid()) {
            continue;
        }

        auto* slotElem = slotsXml->createNewChildElement("Slot");
        slotElem->setAttribute("index", static_cast<int>(i));
        slotElem->setAttribute("fileName", juce::String(slotData->fileName));
        slotElem->setAttribute("absolutePath", juce::String(slotData->filePath));

        if (!slotData->filePath.empty()) {
            const juce::File f(slotData->filePath);
            slotElem->setAttribute("relativePath", f.getRelativePathFrom(baseDir));
        }
        slotElem->setAttribute("sha256Hash", juce::String(slotData->sha256Hash));

        if (embedFlac && !slotData->rawAudioMono.empty()) {
            const std::string flacB64 = SpecTileParameterLayout::encodeMonoAudioToFlacBase64(
                slotData->rawAudioMono,
                slotData->sampleRate);
            slotElem->setAttribute("flacBase64", juce::String(flacB64));
        }
    }

    copyXmlToBinary(*rootXml, destData);
}

void SpecTileAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    std::unique_ptr<juce::XmlElement> rootXml(getXmlFromBinary(data, sizeInBytes));
    if (!rootXml) {
        return;
    }

    // 1. Restore APVTS parameter tree
    if (rootXml->hasTagName(apvts_.state.getType())) {
        apvts_.replaceState(juce::ValueTree::fromXml(*rootXml));
    }

    // 2. Restore 24D Spectral Genome g_0
    if (const auto* genomeXml = rootXml->getChildByName("SpectralGenome24D")) {
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            const double val = genomeXml->getDoubleAttribute(
                "g" + juce::String(static_cast<int>(d)),
                static_cast<double>(baseGenome_[d]));
            baseGenome_[d]   = std::clamp(static_cast<float>(val), 0.02f, 0.98f);
            baseLogitsZ0_[d] = logitTransform(baseGenome_[d]);
            activeGenome_[d] = baseGenome_[d];
        }
        buildOrthonormalEvolutionaryBranches();
    }

    // 3. Restore 8 Sample Slots via Triple Fallback
    if (const auto* slotsXml = rootXml->getChildByName("SampleSlots")) {
        const juce::File baseDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
        for (const auto* slotElem : slotsXml->getChildWithTagNameIterator("Slot")) {
            SampleSlotPresetState st{};
            st.slotIndex    = static_cast<std::size_t>(std::clamp(slotElem->getIntAttribute("index", 0), 0, 7));
            st.relativePath = slotElem->getStringAttribute("relativePath").toStdString();
            st.absolutePath = slotElem->getStringAttribute("absolutePath").toStdString();
            st.sha256Hash   = slotElem->getStringAttribute("sha256Hash").toStdString();
            st.flacBase64   = slotElem->getStringAttribute("flacBase64").toStdString();

            resolveAndLoadSampleWithTripleFallback(st, baseDir);
        }
    }
}

bool SpecTileAudioProcessor::resolveAndLoadSampleWithTripleFallback(
    const SampleSlotPresetState& slotState,
    const juce::File&            presetBaseDir)
{
    // Fallback 1: Try relative path from preset/project directory, then absolute path
    if (!slotState.relativePath.empty()) {
        const juce::File relCandidate = presetBaseDir.getChildFile(juce::String(slotState.relativePath));
        if (relCandidate.existsAsFile()) {
            if (slotState.sha256Hash.empty() || computeFileSha256Hex(relCandidate) == slotState.sha256Hash) {
                if (loadSampleFileIntoSlot(slotState.slotIndex, relCandidate)) {
                    return true;
                }
            }
        }
    }

    if (!slotState.absolutePath.empty()) {
        const juce::File absCandidate(slotState.absolutePath);
        if (absCandidate.existsAsFile()) {
            if (slotState.sha256Hash.empty() || computeFileSha256Hex(absCandidate) == slotState.sha256Hash) {
                if (loadSampleFileIntoSlot(slotState.slotIndex, absCandidate)) {
                    return true;
                }
            }
        }
    }

    // Fallback 2: Search user sample directories by cryptographic SHA-256 content hash
    if (!slotState.sha256Hash.empty()) {
        for (const auto& searchDir : userSampleSearchDirs_) {
            if (!searchDir.isDirectory()) {
                continue;
            }
            for (const auto& entry : juce::RangedDirectoryIterator(
                     searchDir, true, "*.wav;*.flac;*.aif;*.aiff;*.ogg", juce::File::findFiles))
            {
                const juce::File candidateFile = entry.getFile();
                if (computeFileSha256Hex(candidateFile) == slotState.sha256Hash) {
                    if (loadSampleFileIntoSlot(slotState.slotIndex, candidateFile)) {
                        return true;
                    }
                }
            }
        }
    }

    // Fallback 3: Decode lossless embedded FLAC Base64 directly from preset state
    if (!slotState.flacBase64.empty()) {
        std::vector<float> decodedMono;
        double             decodedRate = currentSampleRate_;
        if (SpecTileParameterLayout::decodeFlacBase64ToMonoAudio(slotState.flacBase64, decodedMono, decodedRate)) {
            loadRawMonoBufferIntoSlot(
                slotState.slotIndex,
                decodedMono,
                decodedRate,
                "Embedded_Slot_" + std::to_string(slotState.slotIndex + 1) + ".flac",
                slotState.absolutePath,
                slotState.sha256Hash);
            return true;
        }
    }

    return false;
}

std::string SpecTileAudioProcessor::exportPresetToJsonString(bool embedFlacBase64) const {
    auto rootObj = std::make_unique<juce::DynamicObject>();
    rootObj->setProperty("plugin", "SpecTile");
    rootObj->setProperty("version", "1.0.0");

    // Export Genome
    juce::Array<juce::var> genomeArr;
    for (float gVal : baseGenome_) {
        genomeArr.add(static_cast<double>(gVal));
    }
    rootObj->setProperty("genome24D", genomeArr);

    // Export Sample Slots
    juce::Array<juce::var> slotsArr;
    const juce::File baseDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    for (std::size_t i = 0; i < kMaxSources; ++i) {
        const auto slotData = sampleRcuBank_.getSlot(i);
        if (!slotData || !slotData->isValid()) {
            continue;
        }
        auto slotObj = std::make_unique<juce::DynamicObject>();
        slotObj->setProperty("index", static_cast<int>(i));
        slotObj->setProperty("absolutePath", juce::String(slotData->filePath));
        if (!slotData->filePath.empty()) {
            slotObj->setProperty("relativePath", juce::File(slotData->filePath).getRelativePathFrom(baseDir));
        }
        slotObj->setProperty("sha256Hash", juce::String(slotData->sha256Hash));
        if (embedFlacBase64 && !slotData->rawAudioMono.empty()) {
            slotObj->setProperty(
                "flacBase64",
                juce::String(SpecTileParameterLayout::encodeMonoAudioToFlacBase64(
                    slotData->rawAudioMono, slotData->sampleRate)));
        }
        slotsArr.add(juce::var(slotObj.release()));
    }
    rootObj->setProperty("sampleSlots", slotsArr);

    return juce::JSON::toString(juce::var(rootObj.release()), false).toStdString();
}

bool SpecTileAudioProcessor::importPresetFromJsonString(const std::string& jsonText) {
    const juce::var parsed = juce::JSON::parse(juce::String(jsonText));
    const auto* rootObj = parsed.getDynamicObject();
    if (rootObj == nullptr) {
        return false;
    }

    if (const auto* genomeArr = rootObj->getProperty("genome24D").getArray()) {
        const std::size_t count = std::min<std::size_t>(kGenomeDim, static_cast<std::size_t>(genomeArr->size()));
        for (std::size_t d = 0; d < count; ++d) {
            baseGenome_[d]   = std::clamp(static_cast<float>(static_cast<double>((*genomeArr)[static_cast<int>(d)])), 0.02f, 0.98f);
            baseLogitsZ0_[d] = logitTransform(baseGenome_[d]);
            activeGenome_[d] = baseGenome_[d];
        }
        buildOrthonormalEvolutionaryBranches();
    }

    if (const auto* slotsArr = rootObj->getProperty("sampleSlots").getArray()) {
        const juce::File baseDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
        for (const auto& item : *slotsArr) {
            if (const auto* slotObj = item.getDynamicObject()) {
                SampleSlotPresetState st{};
                st.slotIndex    = static_cast<std::size_t>(std::clamp(static_cast<int>(slotObj->getProperty("index")), 0, 7));
                st.relativePath = slotObj->getProperty("relativePath").toString().toStdString();
                st.absolutePath = slotObj->getProperty("absolutePath").toString().toStdString();
                st.sha256Hash   = slotObj->getProperty("sha256Hash").toString().toStdString();
                st.flacBase64   = slotObj->getProperty("flacBase64").toString().toStdString();
                resolveAndLoadSampleWithTripleFallback(st, baseDir);
            }
        }
    }
    return true;
}

// ==============================================================================
// 8. Editor Factory & JUCE Plugin Entry Point
// ==============================================================================
juce::AudioProcessorEditor* SpecTileAudioProcessor::createEditor() {
#if SPECTILE_HAS_CUSTOM_EDITOR
    return new spectile::ui::SpecTileAudioProcessorEditor(*this);
#else
    return new juce::GenericAudioProcessorEditor(*this);
#endif
}

} // namespace spectile::engine

// Global JUCE factory function
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new spectile::engine::SpecTileAudioProcessor();
}

#endif // SPECTILE_HAS_JUCE
