#pragma once
// ==============================================================================
// SpecTile — src/engine/PluginProcessor.hpp
// Settore 3: Main JUCE 8 / CLAP Real-Time AudioProcessor, RCU Garbage Collector,
//            24D Genome QR+SLERP Mutator, ERB->FFT Rule Mask Generator & APVTS
// ==============================================================================

#include "StaggeredHopScheduler.hpp"
#include "SpecTileVoice.hpp"
#include "SampleAnalysisWorker.hpp"
#include "ParameterLayout.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if SPECTILE_HAS_JUCE

namespace spectile::engine {

// ==============================================================================
// SpecTileAudioProcessor — Real-Time Multi-Sample Spectral Mosaicing Synth
// ==============================================================================
class SpecTileAudioProcessor final : public juce::AudioProcessor,
                                     private juce::Timer
{
public:
    SpecTileAudioProcessor();
    ~SpecTileAudioProcessor() override;

    SpecTileAudioProcessor(const SpecTileAudioProcessor&)            = delete;
    SpecTileAudioProcessor& operator=(const SpecTileAudioProcessor&) = delete;

    // --------------------------------------------------------------------------
    // JUCE AudioProcessor Lifecycle & Real-Time Audio Callback
    // --------------------------------------------------------------------------
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    // --------------------------------------------------------------------------
    // Editor & Metadata
    // --------------------------------------------------------------------------
    juce::AudioProcessorEditor* createEditor() override;
    bool                        hasEditor() const override { return true; }

    const juce::String getName() const override { return "SpecTile"; }
    bool               acceptsMidi() const override { return true; }
    bool               producesMidi() const override { return false; }
    bool               isMidiEffect() const override { return false; }
    double             getTailLengthSeconds() const override { return 0.5; }

    // --------------------------------------------------------------------------
    // Programs / Presets
    // --------------------------------------------------------------------------
    int                getNumPrograms() override { return 1; }
    int                getCurrentProgram() override { return 0; }
    void               setCurrentProgram(int index) override { (void)index; }
    const juce::String getProgramName(int index) override { (void)index; return "Default Chimera"; }
    void               changeProgramName(int index, const juce::String& newName) override { (void)index; (void)newName; }

    // --------------------------------------------------------------------------
    // State & Preset Serialization (XML + JSON + Triple Sample Fallback)
    // --------------------------------------------------------------------------
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    [[nodiscard]] std::string exportPresetToJsonString(bool embedFlacBase64 = true) const;
    bool                      importPresetFromJsonString(const std::string& jsonText);

    // --------------------------------------------------------------------------
    // Asynchronous Sample Slot Loading & Inspection API (Slots 0..7)
    // --------------------------------------------------------------------------
    bool loadSampleFileIntoSlot(std::size_t slotIndex, const juce::File& audioFile);
    void loadRawMonoBufferIntoSlot(
        std::size_t               slotIndex,
        const std::vector<float>& audioMono,
        double                    sampleRate,
        const std::string&        displayName,
        const std::string&        filePath   = {},
        const std::string&        sha256Hash = {});

    [[nodiscard]] float getSlotAnalysisProgress(std::size_t slotIndex) const noexcept {
        return workerPool_.getSlotProgress(slotIndex);
    }

    [[nodiscard]] bool isSlotAnalyzing(std::size_t slotIndex) const noexcept {
        return workerPool_.isSlotAnalyzing(slotIndex);
    }

    [[nodiscard]] std::shared_ptr<const SpectralSampleData> getAnalyzedSlotData(std::size_t slotIndex) const noexcept {
        return sampleRcuBank_.getSlot(slotIndex);
    }

    void addSampleSearchDirectory(const juce::File& directory);
    void setEmbedFlacInState(bool shouldEmbed) noexcept { embedSamplesInPreset_.store(shouldEmbed); }
    [[nodiscard]] bool getEmbedFlacInState() const noexcept { return embedSamplesInPreset_.load(); }

    // --------------------------------------------------------------------------
    // "The Chimera Button" — 24D Spectral Genome & 6 QR-Orthonormal Branches
    // --------------------------------------------------------------------------
    void triggerGenomeMutation(int archetypeIndex = -1, uint32_t randomSeed = 0) noexcept;
    void evaluateBranchMorph(float rhoRadius, float thetaRadians) noexcept;

    [[nodiscard]] std::array<float, kGenomeDim> getCurrentGenome() const noexcept {
        return activeGenome_;
    }

    // --------------------------------------------------------------------------
    // Wait-Free SPSC Telemetry Queue for 60/120 FPS Polychromatic GPU Shader
    // --------------------------------------------------------------------------
    bool popNextGpuColumnPacket(GpuColumnPacket& outPacket) noexcept {
        return gpuTelemetryQueue_.tryPop(outPacket);
    }

    [[nodiscard]] SpscRingBuffer<GpuColumnPacket, 64>& getGpuTelemetryQueue() noexcept {
        return gpuTelemetryQueue_;
    }

    [[nodiscard]] juce::AudioProcessorValueTreeState& getApvts() noexcept { return apvts_; }
    [[nodiscard]] const juce::AudioProcessorValueTreeState& getApvts() const noexcept { return apvts_; }
    [[nodiscard]] const AtomicParameterCache& getAtomicParamCache() const noexcept { return paramCache_; }

private:
    // 10 Hz MessageThread Timer callback for Lock-Free RCU Deferred Garbage Collection
    void timerCallback() override;

    // Precomputes psychoacoustic ERB->FFT raised-cosine upsampling tables (O(K))
    void initializeErbProjectionTables(double sampleRate) noexcept;

    // Builds 6 Gram-Schmidt QR-orthonormalized evolutionary branch vectors in R^24
    void buildOrthonormalEvolutionaryBranches() noexcept;

    // Evaluates the 4 Rule Families on B=64 ERB bands and upsamples + feathers to K=2049
    void evaluateRealTimeRuleMasks(
        const std::array<std::shared_ptr<const SpectralSampleData>, kMaxSources>& sampleSnapshot,
        const HopSynthesisParams&                                                 hopParams,
        uint64_t                                                                  hopFrameIdx) noexcept;

    // Resolves a sample slot during preset recall via Triple Fallback:
    // (1) Relative/Absolute Path -> (2) SHA-256 Search -> (3) Embedded FLAC Base64
    bool resolveAndLoadSampleWithTripleFallback(
        const SampleSlotPresetState& slotState,
        const juce::File&            presetBaseDir);

    [[nodiscard]] static std::string computeFileSha256Hex(const juce::File& file);

    // --------------------------------------------------------------------------
    // Core Subsystems & Lock-Free Concurrency Primitives
    // --------------------------------------------------------------------------
    juce::AudioProcessorValueTreeState  apvts_;
    AtomicParameterCache                paramCache_{};
    juce::AudioFormatManager            formatManager_;

    DeferredSampleGarbageCollector      sampleGc_;
    SamplePoolRcuBank                   sampleRcuBank_;
    SampleAnalysisWorkerPool            workerPool_;

    LockFreeTripleBuffer<MaskMatrix2D>  maskTripleBuffer_;
    SpscRingBuffer<GpuColumnPacket, 64> gpuTelemetryQueue_;
    SpecTilePolyphonicVoiceManager      voiceManager_;

    // Pre-allocated audio thread snapshot of the 8 sample slots (never freed on audio thread)
    std::array<std::shared_ptr<const SpectralSampleData>, kMaxSources> audioThreadSampleSnapshot_{};

    // Psychoacoustic ERB -> FFT Raised-Cosine Upsampling Tables (Eq. 1.2)
    alignas(kCacheLineBytes) std::array<uint16_t, kNumBinsPadded> erbLowerBandIdx_{};
    alignas(kCacheLineBytes) std::array<uint16_t, kNumBinsPadded> erbUpperBandIdx_{};
    alignas(kCacheLineBytes) std::array<float, kNumBinsPadded>    erbRaisedCosWeight_{};

    // Causal 2D Markov Persistence State across B = 64 ERB bands
    alignas(kCacheLineBytes) std::array<std::array<float, kNumErbBands>, kMaxSources> prevErbMasks_{};

    // 24D Spectral Genome g_0 in (0,1)^24, Tangent Logits z_0 in R^24 & 6 QR Orthonormal Branches
    std::array<float, kGenomeDim>                               baseGenome_{};
    std::array<float, kGenomeDim>                               baseLogitsZ0_{};
    std::array<float, kGenomeDim>                               activeGenome_{};
    std::array<std::array<float, kGenomeDim>, kNumBranches>     qrBranchVectors_{};
    uint32_t                                                    prngState_ = 0x5350544Cu;

    // Live Sidechain Input Ring Buffer for PlayMode::LiveSidechainHybrid (Slot S1)
    alignas(kCacheLineBytes) std::array<float, kDefaultFftSize> liveSidechainRing_{};
    std::size_t                                                 liveSidechainWritePos_ = 0;

    double                    currentSampleRate_     = 48000.0;
    uint64_t                  globalHopCounter_      = 0;
    std::atomic<bool>         embedSamplesInPreset_{true};
    std::vector<juce::File>   userSampleSearchDirs_;
};

} // namespace spectile::engine

#endif // SPECTILE_HAS_JUCE
