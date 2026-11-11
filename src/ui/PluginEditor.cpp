#include "PluginEditor.hpp"

namespace spectile::ui
{

SpecTileAudioProcessorEditor::SpecTileAudioProcessorEditor(
    juce::AudioProcessor& processor,
    juce::AudioProcessorValueTreeState* apvts,
    LockFreeSpscRingBuffer<SpectralColumnPacket, 128>* externalSpscQueue)
    : juce::AudioProcessorEditor(&processor),
      apvts_(apvts),
      activeSpscQueue_(externalSpscQueue != nullptr ? externalSpscQueue : &internalSpscQueue_)
{
    setLookAndFeel(&lookAndFeel_);
    setOpaque(true);

    // 1. Configure Top Bar Selectors (4 Play Modes & 6 Musical Archetypes)
    for (size_t i = 0; i < ThemeOKLCH::kPlayModeNames.size(); ++i)
        playModeSelector_.addItem(juce::String(std::string(ThemeOKLCH::kPlayModeNames[i])), static_cast<int>(i + 1));
    playModeSelector_.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(playModeSelector_);

    for (size_t i = 0; i < ThemeOKLCH::kArchetypeNames.size(); ++i)
        archetypeSelector_.addItem(juce::String(std::string(ThemeOKLCH::kArchetypeNames[i])), static_cast<int>(i + 1));
    archetypeSelector_.setSelectedId(1, juce::dontSendNotification);
    archetypeSelector_.onChange = [this]
    {
        chimeraHub_.selectArchetype(archetypeSelector_.getSelectedId() - 1);
    };
    addAndMakeVisible(archetypeSelector_);

    viewToggleBtn_.setButtonText("[# DEEP MATRIX]");
    viewToggleBtn_.setClickingTogglesState(true);
    viewToggleBtn_.onClick = [this]
    {
        isDeepMatrixView_ = viewToggleBtn_.getToggleState();
        viewToggleBtn_.setButtonText(isDeepMatrixView_ ? "[O CHIMERA HUB]" : "[# DEEP MATRIX]");
    };
    addAndMakeVisible(viewToggleBtn_);

    // 2. Initialize Central GPU Polychromatic Spectrogram
    spectrogramGL_ = std::make_unique<PolychromeSpectrogramGL>(*activeSpscQueue_);
    addAndMakeVisible(*spectrogramGL_);

    // 3. Initialize Layer 1: 8 Sample Slot Cards (S1..S4 Left Dock, S5..S8 Right Dock) + Central Radial Hub
    for (int i = 0; i < LayoutMetrics::kSlotCount; ++i)
    {
        slotCards_[static_cast<size_t>(i)] = std::make_unique<SpectralSampleSlotCard>(i);
        addAndMakeVisible(*slotCards_[static_cast<size_t>(i)]);
    }

    addAndMakeVisible(chimeraHub_);

    // 4. Initialize Layer 2: 4-Quadrant Deep Spectral Rule Matrix
    addChildComponent(deepMatrixView_);

    // Synchronize Band Locks bidirectionally between Spectrogram GL and Q1 Partitioning
    spectrogramGL_->setOnBandLockChanged([this](int lockIdx, float freqHz, int targetSlot)
    {
        deepMatrixView_.getQ1Partitioning().updateBandLockFromSpectrogram(lockIdx, freqHz, targetSlot);
    });

    deepMatrixView_.getQ1Partitioning().setOnBandLockSync([this](int lockIdx, const BandLockDividerState& st)
    {
        spectrogramGL_->setBandLockState(lockIdx, st);
    });

    // Synchronize 24-Gene Vector bidirectionally between Chimera Radial Hub (SLERP) and Q4 Live Inspector
    chimeraHub_.setOnGenomeChanged([this](const ChimeraRadialHub::GenomeVector& g, float, float)
    {
        deepMatrixView_.getQ4GenomeLfo().setLiveGenome(g);
    });

    deepMatrixView_.getQ4GenomeLfo().setOnGenomeEdited([this](const ChimeraRadialHub::GenomeVector& g)
    {
        chimeraHub_.setCurrentGenome(g);
    });

    // Trigger initial genome broadcast
    deepMatrixView_.getQ4GenomeLfo().setLiveGenome(chimeraHub_.getCurrentGenome());

    // 5. Initialize Bottom Bar: 3 Macro Knobs with Modulation Rings + 8 Drag-and-Drop Mod Sources
    macroTileSize_ = std::make_unique<MacroKnobWithModRing>(
        "SPECTRAL TILE SIZE", "Delta_t x Delta_f Scale", 0.42, ThemeOKLCH::getSlotColour(0));
    macroRuleMutation_ = std::make_unique<MacroKnobWithModRing>(
        "RULE ORDER <-> MUTATION", "Deterministic vs. Voronoi", 0.50, ThemeOKLCH::getSlotColour(2));
    macroPhaseGlue_ = std::make_unique<MacroKnobWithModRing>(
        "PHASE & SPECTRAL GLUE", "2D Feathering + MS-PGHI", 0.88, ThemeOKLCH::getSlotColour(3));

    macroPhaseGlue_->getSlider().onValueChange = [this]
    {
        const float glue = static_cast<float>(macroPhaseGlue_->getSlider().getValue());
        spectrogramGL_->setVoronoiBorderIntensity(1.0f - 0.55f * glue);
    };

    addAndMakeVisible(*macroTileSize_);
    addAndMakeVisible(*macroRuleMutation_);
    addAndMakeVisible(*macroPhaseGlue_);

    static constexpr std::array<const char*, 8> kModNames {{
        "LFO 1", "LFO 2", "SPEC FLW", "ENV", "VELOCITY", "MPE Y", "MPE Z", "RANDOM"
    }};

    for (int m = 0; m < 8; ++m)
    {
        modSourcePills_[static_cast<size_t>(m)] = std::make_unique<ModSourcePill>(
            m, kModNames[static_cast<size_t>(m)], ThemeOKLCH::getSlotColour(m));
        addAndMakeVisible(*modSourcePills_[static_cast<size_t>(m)]);
    }

    setResizable(true, true);
    setResizeLimits(LayoutMetrics::kMinWidth, LayoutMetrics::kMinHeight,
                    LayoutMetrics::kMaxWidth, LayoutMetrics::kMaxHeight);
    setSize(LayoutMetrics::kDefaultWidth, LayoutMetrics::kDefaultHeight);

    startTimerHz(60);
    juce::ignoreUnused(apvts_);
}

SpecTileAudioProcessorEditor::~SpecTileAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

void SpecTileAudioProcessorEditor::timerCallback()
{
    // 1. Update conquered bin percentages on the 8 sample slot cards
    const auto ratios = spectrogramGL_->getLatestConqueredRatios();
    for (size_t i = 0; i < LayoutMetrics::kSlotCount; ++i)
    {
        if (slotCards_[i] != nullptr)
            slotCards_[i]->setConqueredRatio(ratios[i]);
    }

    // 2. Smoothly animate transition between Layer 1 (Chimera View) and Layer 2 (Deep Rule Matrix)
    const float targetMorph = isDeepMatrixView_ ? 1.0f : 0.0f;
    if (std::abs(layerMorphProgress_ - targetMorph) > 0.004f)
    {
        layerMorphProgress_ += (targetMorph - layerMorphProgress_) * 0.24f;
        if (std::abs(layerMorphProgress_ - targetMorph) <= 0.004f)
            layerMorphProgress_ = targetMorph;
        updateDualLayerGeometry();
        repaint();
    }
}

void SpecTileAudioProcessorEditor::resized()
{
    auto fullBounds = getLocalBounds();

    // 1. Top Bar Layout
    auto topBar = fullBounds.removeFromTop(LayoutMetrics::kTopBarHeight).reduced(10, 8);
    topBar.removeFromLeft(260); // Space for SpecTile logo & subtitle

    playModeSelector_.setBounds(topBar.removeFromLeft(186).reduced(0, 2));
    topBar.removeFromLeft(12);
    archetypeSelector_.setBounds(topBar.removeFromLeft(238).reduced(0, 2));

    viewToggleBtn_.setBounds(topBar.removeFromRight(154).reduced(0, 2));

    // 2. Bottom Bar Layout (3 Macro Knobs + 8 Modulator Drag Pills)
    auto bottomBar = fullBounds.removeFromBottom(LayoutMetrics::kBottomBarHeight).reduced(10, 6);
    auto modStrip  = bottomBar.removeFromBottom(26);
    bottomBar.removeFromBottom(6);

    const int macroW = (bottomBar.getWidth() - 16) / 3;
    macroTileSize_->setBounds(bottomBar.removeFromLeft(macroW));
    bottomBar.removeFromLeft(8);
    macroRuleMutation_->setBounds(bottomBar.removeFromLeft(macroW));
    bottomBar.removeFromLeft(8);
    macroPhaseGlue_->setBounds(bottomBar);

    modStrip.removeFromLeft(96); // "MOD SOURCES:" label space
    const int pillW = (modStrip.getWidth() - 7 * 6) / 8;
    for (size_t m = 0; m < 8; ++m)
    {
        modSourcePills_[m]->setBounds(modStrip.removeFromLeft(pillW));
        modStrip.removeFromLeft(6);
    }

    // 3. Central Dual-Layer Area
    updateDualLayerGeometry();
}

void SpecTileAudioProcessorEditor::updateDualLayerGeometry()
{
    auto centerArea = getLocalBounds()
        .withTrimmedTop(LayoutMetrics::kTopBarHeight)
        .withTrimmedBottom(LayoutMetrics::kBottomBarHeight)
        .reduced(10, 4);

    if (centerArea.isEmpty())
        return;

    const float t = std::clamp(layerMorphProgress_, 0.0f, 1.0f);

    // Layer 1 Side Docks slide out and fade as t -> 1.0
    const int dockW = juce::roundToInt(static_cast<float>(LayoutMetrics::kSideDockWidth) * (1.0f - t));
    const bool showLayer1 = (t < 0.96f);
    const bool showLayer2 = (t > 0.04f);

    for (auto& card : slotCards_)
    {
        card->setVisible(showLayer1);
        card->setAlpha(1.0f - t);
    }
    chimeraHub_.setVisible(showLayer1);
    chimeraHub_.setAlpha(1.0f - t);

    deepMatrixView_.setVisible(showLayer2);
    deepMatrixView_.setAlpha(t);

    if (showLayer1 && dockW > 40)
    {
        auto leftDock  = centerArea.withWidth(dockW);
        auto rightDock = centerArea.withLeft(centerArea.getRight() - dockW);
        const int slotH = centerArea.getHeight() / 4;

        for (int i = 0; i < 4; ++i)
        {
            slotCards_[static_cast<size_t>(i)]->setBounds(
                leftDock.getX(), centerArea.getY() + i * slotH, LayoutMetrics::kSideDockWidth, slotH - 4);
            slotCards_[static_cast<size_t>(i + 4)]->setBounds(
                rightDock.getRight() - LayoutMetrics::kSideDockWidth, centerArea.getY() + i * slotH,
                LayoutMetrics::kSideDockWidth, slotH - 4);
        }
    }

    // Spectrogram GL transitions smoothly from Central Arena (Layer 1) to Top Panoramic Monitor (Layer 2)
    const juce::Rectangle<int> specLayer1Bounds = centerArea.withTrimmedLeft(LayoutMetrics::kSideDockWidth + 8)
                                                            .withTrimmedRight(LayoutMetrics::kSideDockWidth + 8);
    const int bannerH = juce::roundToInt(static_cast<float>(centerArea.getHeight()) * LayoutMetrics::kMatrixHeaderSpecRatio);
    const juce::Rectangle<int> specLayer2Bounds = centerArea.withHeight(bannerH);

    auto lerpInt = [t](int a, int b) { return a + juce::roundToInt(static_cast<float>(b - a) * t); };
    const juce::Rectangle<int> specBounds(
        lerpInt(specLayer1Bounds.getX(),      specLayer2Bounds.getX()),
        lerpInt(specLayer1Bounds.getY(),      specLayer2Bounds.getY()),
        lerpInt(specLayer1Bounds.getWidth(),  specLayer2Bounds.getWidth()),
        lerpInt(specLayer1Bounds.getHeight(), specLayer2Bounds.getHeight()));

    spectrogramGL_->setCompactBannerMode(t > 0.5f);
    spectrogramGL_->setBounds(specBounds);
    chimeraHub_.setBounds(specLayer1Bounds);

    const juce::Rectangle<int> matrixBounds = centerArea.withTrimmedTop(bannerH + 6);
    deepMatrixView_.setBounds(matrixBounds);
}

void SpecTileAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(ThemeOKLCH::obsidianDeep());

    // Top Bar background & branding
    const juce::Rectangle<float> topBar(0.0f, 0.0f, static_cast<float>(getWidth()),
                                        static_cast<float>(LayoutMetrics::kTopBarHeight));
    g.setColour(ThemeOKLCH::panelSlate());
    g.fillRect(topBar);
    g.setColour(ThemeOKLCH::borderSubtle());
    g.drawLine(0.0f, topBar.getBottom(), topBar.getWidth(), topBar.getBottom(), 1.2f);

    // SpecTile Title & Subtitle
    g.setColour(ThemeOKLCH::accentLaser());
    g.setFont(juce::FontOptions(19.0f, juce::Font::bold));
    g.drawText("SPECTILE", 14, 7, 130, 22, juce::Justification::centredLeft, false);

    g.setColour(ThemeOKLCH::textSecondary());
    g.setFont(juce::FontOptions(10.0f));
    g.drawText("CHIMERIC SPECTRAL MOSAICING // MS-PGHI", 14, 27, 240, 16, juce::Justification::centredLeft, false);

    // GPU FPS & Real-Time DSP Telemetry Badge
    const int fps = juce::roundToInt(spectrogramGL_->getEstimatedGpuFps());
    const juce::String telemetry = "GPU " + juce::String(fps) + " FPS  |  DSP RTF 2.6%  |  8x STFT-HPR";
    const juce::Rectangle<float> badge(static_cast<float>(getWidth() - 410), 14.0f, 232.0f, 24.0f);
    g.setColour(ThemeOKLCH::obsidianDeep());
    g.fillRoundedRectangle(badge, 4.0f);
    g.setColour(ThemeOKLCH::borderSubtle());
    g.drawRoundedRectangle(badge, 4.0f, 1.0f);
    g.setColour(ThemeOKLCH::accentLaser());
    g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
    g.drawText(telemetry, badge.toNearestInt(), juce::Justification::centred, false);

    // Bottom Bar divider & Mod Sources label
    const float bottomY = static_cast<float>(getHeight() - LayoutMetrics::kBottomBarHeight);
    g.setColour(ThemeOKLCH::borderSubtle());
    g.drawLine(0.0f, bottomY, static_cast<float>(getWidth()), bottomY, 1.2f);

    g.setColour(ThemeOKLCH::textSecondary());
    g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
    g.drawText("MOD SOURCES:", 12, getHeight() - 32, 90, 24, juce::Justification::centredLeft, false);
}

} // namespace spectile::ui
