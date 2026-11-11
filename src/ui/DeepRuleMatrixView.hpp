#pragma once

#include "ThemeOKLCH.hpp"
#include "PolychromeSpectrogramGL.hpp"
#include "ChimeraRadialHub.hpp"
#include <juce_gui_basics/juce_gui_basics.h>
#include <array>
#include <cmath>
#include <functional>

namespace spectile::ui
{

// =============================================================================
// QUADRANT 1: SPECTRAL PARTITIONING & FREQUENCY BAND LOCKS
// =============================================================================

class SpectralPartitioningQuadrant final : public juce::Component
{
public:
    using BandLockSyncCallback = std::function<void(int lockIdx, const BandLockDividerState& state)>;

    SpectralPartitioningQuadrant()
    {
        fftSizeBox_.addItem("1024 (Low Latency)", 1);
        fftSizeBox_.addItem("2048 (Balanced)",    2);
        fftSizeBox_.addItem("4096 (Default Hi-Res)", 3);
        fftSizeBox_.addItem("8192 (Ultra Pad)",   4);
        fftSizeBox_.setSelectedId(3, juce::dontSendNotification);
        addAndMakeVisible(fftSizeBox_);

        scaleBox_.addItem("ERB (64 Psychoacoustic Bands)", 1);
        scaleBox_.addItem("Bark (24 Critical Bands)",      2);
        scaleBox_.addItem("Harmonic Overtone Grid",        3);
        scaleBox_.addItem("Linear FFT Bins (2049)",        4);
        scaleBox_.setSelectedId(1, juce::dontSendNotification);
        addAndMakeVisible(scaleBox_);

        static constexpr std::array<float, 4> kDefaultFreqs { 145.0f, 650.0f, 2800.0f, 8500.0f };
        static constexpr std::array<int, 4>   kDefaultSlots { 0, 1, 3, 6 };

        for (size_t i = 0; i < LayoutMetrics::kBandLockCount; ++i)
        {
            auto& row = lockRows_[i];
            row.enableBtn.setButtonText("LOCK " + juce::String(static_cast<int>(i + 1)));
            row.enableBtn.setClickingTogglesState(true);
            row.enableBtn.setToggleState(i == 0, juce::dontSendNotification);
            addAndMakeVisible(row.enableBtn);

            row.freqSlider.setSliderStyle(juce::Slider::LinearHorizontal);
            row.freqSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 64, 18);
            row.freqSlider.setRange(20.0, 20000.0, 1.0);
            row.freqSlider.setSkewFactorFromMidPoint(1000.0);
            row.freqSlider.setTextValueSuffix(" Hz");
            row.freqSlider.setValue(kDefaultFreqs[i], juce::dontSendNotification);
            addAndMakeVisible(row.freqSlider);

            for (int s = 0; s < LayoutMetrics::kSlotCount; ++s)
            {
                const auto& slotInfo = ThemeOKLCH::kSlotPalette[static_cast<size_t>(s)];
                row.slotBox.addItem(juce::String(std::string(slotInfo.code)) + " - " + juce::String(std::string(slotInfo.name)), s + 1);
            }
            row.slotBox.setSelectedId(kDefaultSlots[i] + 1, juce::dontSendNotification);
            addAndMakeVisible(row.slotBox);

            const int lockIndex = static_cast<int>(i);
            auto emitSync = [this, lockIndex]
            {
                if (onBandLockSync_)
                {
                    const auto& r = lockRows_[static_cast<size_t>(lockIndex)];
                    BandLockDividerState st;
                    st.enabled    = r.enableBtn.getToggleState();
                    st.freqHz     = static_cast<float>(r.freqSlider.getValue());
                    st.targetSlot = std::clamp(r.slotBox.getSelectedId() - 1, 0, 7);
                    onBandLockSync_(lockIndex, st);
                }
            };

            row.enableBtn.onClick        = emitSync;
            row.freqSlider.onValueChange = emitSync;
            row.slotBox.onChange         = emitSync;
        }
    }

    void updateBandLockFromSpectrogram(int lockIdx, float freqHz, int targetSlot)
    {
        if (lockIdx >= 0 && lockIdx < LayoutMetrics::kBandLockCount)
        {
            auto& r = lockRows_[static_cast<size_t>(lockIdx)];
            r.enableBtn.setToggleState(true, juce::dontSendNotification);
            r.freqSlider.setValue(freqHz, juce::dontSendNotification);
            r.slotBox.setSelectedId(targetSlot + 1, juce::dontSendNotification);
        }
    }

    void setOnBandLockSync(BandLockSyncCallback cb) { onBandLockSync_ = std::move(cb); }

    void resized() override
    {
        auto b = getLocalBounds().reduced(10);
        b.removeFromTop(22); // Quadrant header

        auto topCombos = b.removeFromTop(24);
        const int halfW = topCombos.getWidth() / 2 - 4;
        fftSizeBox_.setBounds(topCombos.removeFromLeft(halfW).withTrimmedLeft(52));
        topCombos.removeFromLeft(8);
        scaleBox_.setBounds(topCombos.withTrimmedLeft(48));

        b.removeFromTop(8);
        const int rowH = std::max(22, b.getHeight() / 4);
        for (size_t i = 0; i < LayoutMetrics::kBandLockCount; ++i)
        {
            auto rBounds = b.removeFromTop(rowH).reduced(0, 2);
            lockRows_[i].enableBtn.setBounds(rBounds.removeFromLeft(64));
            rBounds.removeFromLeft(6);
            lockRows_[i].slotBox.setBounds(rBounds.removeFromRight(120));
            rBounds.removeFromRight(6);
            lockRows_[i].freqSlider.setBounds(rBounds);
        }
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();
        g.setColour(ThemeOKLCH::panelSlate());
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);
        g.setColour(ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds.reduced(0.5f), LayoutMetrics::kCornerRadius, 1.0f);

        g.setColour(ThemeOKLCH::accentLaser());
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText("Q1 // SPECTRAL PARTITIONING & FREQUENCY BAND LOCKS", 12, 8, getWidth() - 24, 18,
                   juce::Justification::centredLeft, false);

        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(11.0f));
        g.drawText("N_FFT:", 12, 32, 48, 22, juce::Justification::centredLeft, false);
        g.drawText("Scale:", getWidth() / 2 + 6, 32, 46, 22, juce::Justification::centredLeft, false);
    }

private:
    struct BandLockRowWidgets
    {
        juce::TextButton enableBtn;
        juce::Slider     freqSlider;
        juce::ComboBox   slotBox;
    };

    juce::ComboBox fftSizeBox_;
    juce::ComboBox scaleBox_;
    std::array<BandLockRowWidgets, LayoutMetrics::kBandLockCount> lockRows_;
    BandLockSyncCallback onBandLockSync_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectralPartitioningQuadrant)
};

// =============================================================================
// QUADRANT 2: 5x8 RULE & FEATURE ALLOCATION MATRIX + MARKOV 2D PERSISTENCE
// =============================================================================

class RuleAllocationMatrixQuadrant final : public juce::Component
{
public:
    static constexpr int kNumRules = 5;
    static constexpr int kNumSlots = LayoutMetrics::kSlotCount;

    static constexpr std::array<std::string_view, kNumRules> kRuleLabels {{
        "Max Flux (Onset)",
        "Pure Harmonics",
        "Hole-Filler (Anti-Mask)",
        "Harmonic Comb O/E",
        "Voronoi Attraction"
    }};

    RuleAllocationMatrixQuadrant()
    {
        // Initialize default rule matrix weights W_{r, s} in [0, 1]
        for (int r = 0; r < kNumRules; ++r)
        {
            for (int s = 0; s < kNumSlots; ++s)
            {
                const float v = 0.5f + 0.42f * std::sin(static_cast<float>(r * 3 + s) * 0.95f);
                weights_[static_cast<size_t>(r)][static_cast<size_t>(s)] = std::clamp(v, 0.05f, 0.95f);
            }
        }

        configureRotary(tauSlider_,       0.03, 1.50, 0.28, "Temp tau");
        configureRotary(sparsitySlider_,  0.00, 1.00, 0.35, "Sparsity");
        configureRotary(markovStaySlider_,0.00, 0.99, 0.82, "Markov P_t");
        configureRotary(markovResetSlider_,0.00, 1.00, 0.90, "Onset Reset");
    }

    void resized() override
    {
        auto b = getLocalBounds().reduced(10);
        b.removeFromTop(22);

        auto bottomKnobs = b.removeFromBottom(58);
        const int knobW = bottomKnobs.getWidth() / 4;
        tauSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        sparsitySlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        markovStaySlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        markovResetSlider_.setBounds(bottomKnobs.withTrimmedTop(12));

        b.removeFromBottom(4);
        gridArea_ = b.withTrimmedLeft(128).withTrimmedTop(16);
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();
        g.setColour(ThemeOKLCH::panelSlate());
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);
        g.setColour(ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds.reduced(0.5f), LayoutMetrics::kCornerRadius, 1.0f);

        g.setColour(ThemeOKLCH::accentLaser());
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText("Q2 // 5x8 FEATURE & RULE ALLOCATION MATRIX", 12, 8, getWidth() - 24, 18,
                   juce::Justification::centredLeft, false);

        if (!gridArea_.isEmpty())
        {
            const float cellW = static_cast<float>(gridArea_.getWidth())  / static_cast<float>(kNumSlots);
            const float cellH = static_cast<float>(gridArea_.getHeight()) / static_cast<float>(kNumRules);

            // Column headers S1..S8
            g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
            for (int s = 0; s < kNumSlots; ++s)
            {
                const float cx = static_cast<float>(gridArea_.getX()) + static_cast<float>(s) * cellW;
                g.setColour(ThemeOKLCH::getSlotColour(s));
                g.drawText("S" + juce::String(s + 1), static_cast<int>(cx), gridArea_.getY() - 15,
                           static_cast<int>(cellW), 14, juce::Justification::centred, false);
            }

            // Row labels & 5x8 cells
            g.setFont(juce::FontOptions(10.5f));
            for (int r = 0; r < kNumRules; ++r)
            {
                const float cy = static_cast<float>(gridArea_.getY()) + static_cast<float>(r) * cellH;
                g.setColour(ThemeOKLCH::textSecondary());
                g.drawText(juce::String(std::string(kRuleLabels[static_cast<size_t>(r)])),
                           12, static_cast<int>(cy), gridArea_.getX() - 16, static_cast<int>(cellH),
                           juce::Justification::centredLeft, false);

                for (int s = 0; s < kNumSlots; ++s)
                {
                    const float cx = static_cast<float>(gridArea_.getX()) + static_cast<float>(s) * cellW;
                    const juce::Rectangle<float> cell(cx + 1.5f, cy + 1.5f, cellW - 3.0f, cellH - 3.0f);

                    g.setColour(ThemeOKLCH::obsidianDeep());
                    g.fillRoundedRectangle(cell, 3.0f);

                    const float w = weights_[static_cast<size_t>(r)][static_cast<size_t>(s)];
                    const juce::Colour slotCol = ThemeOKLCH::getSlotColour(s);
                    const float fillH = cell.getHeight() * w;
                    const juce::Rectangle<float> fillRect(cell.getX(), cell.getBottom() - fillH, cell.getWidth(), fillH);

                    g.setColour(slotCol.withAlpha(0.72f));
                    g.fillRoundedRectangle(fillRect, 2.5f);

                    g.setColour(ThemeOKLCH::borderSubtle());
                    g.drawRoundedRectangle(cell, 3.0f, 0.8f);
                }
            }
        }

        // Bottom knob labels
        static constexpr std::array<const char*, 4> kKnobTitles { "Softmax Tau", "Sparsity Gate", "Markov Stay", "Onset Reset" };
        const int knobW = (getWidth() - 20) / 4;
        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
        for (int i = 0; i < 4; ++i)
        {
            g.drawText(kKnobTitles[static_cast<size_t>(i)], 10 + i * knobW, getHeight() - 68, knobW, 13,
                       juce::Justification::centred, false);
        }
    }

    void mouseDown(const juce::MouseEvent& e) override { updateCellFromMouse(e.position); }
    void mouseDrag(const juce::MouseEvent& e) override { updateCellFromMouse(e.position); }

private:
    void configureRotary(juce::Slider& s, double minV, double maxV, double initV, const juce::String& name)
    {
        s.setName(name);
        s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        s.setRange(minV, maxV, 0.01);
        s.setValue(initV, juce::dontSendNotification);
        addAndMakeVisible(s);
    }

    void updateCellFromMouse(juce::Point<float> pt)
    {
        if (!gridArea_.toFloat().contains(pt))
            return;

        const float cellW = static_cast<float>(gridArea_.getWidth())  / static_cast<float>(kNumSlots);
        const float cellH = static_cast<float>(gridArea_.getHeight()) / static_cast<float>(kNumRules);

        const int s = std::clamp(static_cast<int>((pt.x - static_cast<float>(gridArea_.getX())) / cellW), 0, kNumSlots - 1);
        const int r = std::clamp(static_cast<int>((pt.y - static_cast<float>(gridArea_.getY())) / cellH), 0, kNumRules - 1);
        const float cellTop = static_cast<float>(gridArea_.getY()) + static_cast<float>(r) * cellH;
        const float normVal = std::clamp(1.0f - (pt.y - cellTop) / cellH, 0.0f, 1.0f);

        weights_[static_cast<size_t>(r)][static_cast<size_t>(s)] = normVal;
        repaint();
    }

    std::array<std::array<float, kNumSlots>, kNumRules> weights_ {};
    juce::Rectangle<int> gridArea_;
    juce::Slider tauSlider_, sparsitySlider_, markovStaySlider_, markovResetSlider_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RuleAllocationMatrixQuadrant)
};

// =============================================================================
// QUADRANT 3: SOURCE-FILTER CROSS-SYNTHESIS (E_i x F_j) & MS-PGHI PHASE SURGERY
// =============================================================================

class SourceFilterPhaseQuadrant final : public juce::Component
{
public:
    SourceFilterPhaseQuadrant()
    {
        for (size_t i = 0; i < LayoutMetrics::kSlotCount; ++i)
        {
            envWeights_[i]  = (i < 3) ? 0.85f : 0.25f;
            fineWeights_[i] = (i >= 2 && i < 6) ? 0.85f : 0.30f;
        }

        phaseModeBox_.addItem("Causal MS-PGHI + Phase Glue (Default)", 1);
        phaseModeBox_.addItem("Identity Phase-Locked (Laroche-Dolson)", 2);
        phaseModeBox_.addItem("Raw Source Phase Stitching",             3);
        phaseModeBox_.addItem("Stochastic Spectral Freeze Blur",        4);
        phaseModeBox_.setSelectedId(1, juce::dontSendNotification);
        addAndMakeVisible(phaseModeBox_);

        configureRotary(crossMixSlider_,    0.0,  1.0,  0.65);
        configureRotary(gammaExpSlider_,    0.2,  1.5,  0.85);
        configureRotary(formantShiftSlider_,-12.0, 12.0, 0.0);
        configureRotary(phaseGlueSlider_,   0.0,  1.0,  0.88);
    }

    void resized() override
    {
        auto b = getLocalBounds().reduced(10);
        b.removeFromTop(22);

        auto modeRow = b.removeFromTop(22);
        phaseModeBox_.setBounds(modeRow.withTrimmedLeft(84));

        auto bottomKnobs = b.removeFromBottom(58);
        const int knobW = bottomKnobs.getWidth() / 4;
        crossMixSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        gammaExpSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        formantShiftSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        phaseGlueSlider_.setBounds(bottomKnobs.withTrimmedTop(12));

        b.removeFromTop(6);
        b.removeFromBottom(4);
        const int halfW = b.getWidth() / 2 - 6;
        envBarsArea_  = b.removeFromLeft(halfW).withTrimmedTop(16);
        b.removeFromLeft(12);
        fineBarsArea_ = b.withTrimmedTop(16);
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();
        g.setColour(ThemeOKLCH::panelSlate());
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);
        g.setColour(ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds.reduced(0.5f), LayoutMetrics::kCornerRadius, 1.0f);

        g.setColour(ThemeOKLCH::accentLaser());
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText("Q3 // SOURCE-FILTER CROSS-SYNTHESIS (E_i x F_j) & MS-PGHI PHASE", 12, 8, getWidth() - 24, 18,
                   juce::Justification::centredLeft, false);

        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(11.0f));
        g.drawText("Phase Engine:", 12, 32, 80, 22, juce::Justification::centredLeft, false);

        drawWeightBars(g, envBarsArea_,  envWeights_,  "Formant Envelope E_i(t,k)");
        drawWeightBars(g, fineBarsArea_, fineWeights_, "Fine Harmonic/Noise F_j(t,k)");

        static constexpr std::array<const char*, 4> kTitles { "Cross-Synth X", "Env Gamma", "Formant Shift", "MS-PGHI Glue" };
        const int knobW = (getWidth() - 20) / 4;
        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
        for (int i = 0; i < 4; ++i)
        {
            g.drawText(kTitles[static_cast<size_t>(i)], 10 + i * knobW, getHeight() - 68, knobW, 13,
                       juce::Justification::centred, false);
        }
    }

    void mouseDown(const juce::MouseEvent& e) override { handleBarDrag(e.position); }
    void mouseDrag(const juce::MouseEvent& e) override { handleBarDrag(e.position); }

private:
    void configureRotary(juce::Slider& s, double minV, double maxV, double initV)
    {
        s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        s.setRange(minV, maxV, 0.01);
        s.setValue(initV, juce::dontSendNotification);
        addAndMakeVisible(s);
    }

    static void drawWeightBars(juce::Graphics& g,
                               const juce::Rectangle<int>& area,
                               const std::array<float, LayoutMetrics::kSlotCount>& weights,
                               const juce::String& title)
    {
        if (area.isEmpty())
            return;

        g.setColour(ThemeOKLCH::textPrimary());
        g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
        g.drawText(title, area.getX(), area.getY() - 16, area.getWidth(), 14, juce::Justification::centredLeft, false);

        const float barW = static_cast<float>(area.getWidth()) / static_cast<float>(LayoutMetrics::kSlotCount);
        for (int i = 0; i < LayoutMetrics::kSlotCount; ++i)
        {
            const float bx = static_cast<float>(area.getX()) + static_cast<float>(i) * barW;
            const juce::Rectangle<float> colBox(bx + 2.0f, static_cast<float>(area.getY()), barW - 4.0f, static_cast<float>(area.getHeight()));

            g.setColour(ThemeOKLCH::obsidianDeep());
            g.fillRoundedRectangle(colBox, 3.0f);

            const float h = colBox.getHeight() * std::clamp(weights[static_cast<size_t>(i)], 0.0f, 1.0f);
            const juce::Rectangle<float> fillBox(colBox.getX(), colBox.getBottom() - h, colBox.getWidth(), h);
            g.setColour(ThemeOKLCH::getSlotColour(i, 0.82f));
            g.fillRoundedRectangle(fillBox, 3.0f);

            g.setColour(ThemeOKLCH::borderSubtle());
            g.drawRoundedRectangle(colBox, 3.0f, 1.0f);
        }
    }

    void handleBarDrag(juce::Point<float> pt)
    {
        auto updateArray = [pt, this](const juce::Rectangle<int>& area, std::array<float, LayoutMetrics::kSlotCount>& arr)
        {
            if (area.toFloat().contains(pt))
            {
                const float barW = static_cast<float>(area.getWidth()) / static_cast<float>(LayoutMetrics::kSlotCount);
                const int idx = std::clamp(static_cast<int>((pt.x - static_cast<float>(area.getX())) / barW), 0, 7);
                const float val = std::clamp(1.0f - (pt.y - static_cast<float>(area.getY())) / static_cast<float>(area.getHeight()), 0.0f, 1.0f);
                arr[static_cast<size_t>(idx)] = val;
                repaint();
            }
        };

        updateArray(envBarsArea_,  envWeights_);
        updateArray(fineBarsArea_, fineWeights_);
    }

    std::array<float, LayoutMetrics::kSlotCount> envWeights_  {};
    std::array<float, LayoutMetrics::kSlotCount> fineWeights_ {};
    juce::Rectangle<int> envBarsArea_;
    juce::Rectangle<int> fineBarsArea_;
    juce::ComboBox phaseModeBox_;
    juce::Slider crossMixSlider_, gammaExpSlider_, formantShiftSlider_, phaseGlueSlider_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SourceFilterPhaseQuadrant)
};

// =============================================================================
// QUADRANT 4: 24-GENE LIVE BAR INSPECTOR, SPECTRAL LFOS & PARTIAL QUANTIZER
// =============================================================================

class GenomeLfoQuantizerQuadrant final : public juce::Component
{
public:
    using GenomeEditCallback = std::function<void(const ChimeraRadialHub::GenomeVector& editedGenome)>;

    GenomeLfoQuantizerQuadrant()
    {
        for (size_t d = 0; d < LayoutMetrics::kGenomeDimension; ++d)
            genome_[d] = 0.5f + 0.3f * std::sin(static_cast<float>(d) * 0.55f);

        quantScaleBox_.addItem("Off (Continuous Frequency)", 1);
        quantScaleBox_.addItem("Chromatic 12-TET",           2);
        quantScaleBox_.addItem("Just Intonation Harmonic",   3);
        quantScaleBox_.addItem("Minor Pentatonic",           4);
        quantScaleBox_.addItem("Spectral Overtone Series",   5);
        quantScaleBox_.setSelectedId(5, juce::dontSendNotification);
        addAndMakeVisible(quantScaleBox_);

        configureRotary(lfo1RateSlider_,   0.02, 16.0, 0.45);
        configureRotary(lfo1DepthSlider_,  0.00, 1.00, 0.38);
        configureRotary(lfo2DriftSlider_,  0.01, 8.00, 0.22);
        configureRotary(quantAmtSlider_,   0.00, 1.00, 0.00);
    }

    void setLiveGenome(const ChimeraRadialHub::GenomeVector& g) noexcept
    {
        genome_ = g;
        repaint();
    }

    void setOnGenomeEdited(GenomeEditCallback cb) { onGenomeEdited_ = std::move(cb); }

    void resized() override
    {
        auto b = getLocalBounds().reduced(10);
        b.removeFromTop(22);

        auto scaleRow = b.removeFromTop(22);
        quantScaleBox_.setBounds(scaleRow.withTrimmedLeft(108));

        auto bottomKnobs = b.removeFromBottom(58);
        const int knobW = bottomKnobs.getWidth() / 4;
        lfo1RateSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        lfo1DepthSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        lfo2DriftSlider_.setBounds(bottomKnobs.removeFromLeft(knobW).withTrimmedTop(12));
        quantAmtSlider_.setBounds(bottomKnobs.withTrimmedTop(12));

        b.removeFromTop(18);
        b.removeFromBottom(4);
        genomeBarsArea_ = b;
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();
        g.setColour(ThemeOKLCH::panelSlate());
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);
        g.setColour(ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds.reduced(0.5f), LayoutMetrics::kCornerRadius, 1.0f);

        g.setColour(ThemeOKLCH::accentLaser());
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText("Q4 // 24-GENE LIVE INSPECTOR, SPECTRAL LFOS & PARTIAL QUANTIZER", 12, 8, getWidth() - 24, 18,
                   juce::Justification::centredLeft, false);

        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(11.0f));
        g.drawText("Partial Quantizer:", 12, 32, 102, 22, juce::Justification::centredLeft, false);

        if (!genomeBarsArea_.isEmpty())
        {
            g.setColour(ThemeOKLCH::textPrimary());
            g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
            g.drawText("Spectral Genome Vector g in [0, 1]^24 (Live SLERP / Scrubbable):",
                       genomeBarsArea_.getX(), genomeBarsArea_.getY() - 15, genomeBarsArea_.getWidth(), 14,
                       juce::Justification::centredLeft, false);

            const float barW = static_cast<float>(genomeBarsArea_.getWidth()) / static_cast<float>(LayoutMetrics::kGenomeDimension);
            for (int d = 0; d < LayoutMetrics::kGenomeDimension; ++d)
            {
                const float bx = static_cast<float>(genomeBarsArea_.getX()) + static_cast<float>(d) * barW;
                const juce::Rectangle<float> track(bx + 1.2f, static_cast<float>(genomeBarsArea_.getY()),
                                                   barW - 2.4f, static_cast<float>(genomeBarsArea_.getHeight()));

                g.setColour(ThemeOKLCH::obsidianDeep());
                g.fillRoundedRectangle(track, 2.0f);

                const float val = std::clamp(genome_[static_cast<size_t>(d)], 0.0f, 1.0f);
                const float fillH = track.getHeight() * val;
                const juce::Rectangle<float> fill(track.getX(), track.getBottom() - fillH, track.getWidth(), fillH);

                const juce::Colour geneCol = ThemeOKLCH::getSlotColour(d % LayoutMetrics::kSlotCount, 0.85f);
                g.setColour(geneCol);
                g.fillRoundedRectangle(fill, 2.0f);

                g.setColour(ThemeOKLCH::borderSubtle());
                g.drawRoundedRectangle(track, 2.0f, 0.8f);
            }
        }

        static constexpr std::array<const char*, 4> kTitles { "Spec LFO1 Hz", "LFO1 Depth", "Spec LFO2 Drift", "Partial Quant" };
        const int knobW = (getWidth() - 20) / 4;
        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
        for (int i = 0; i < 4; ++i)
        {
            g.drawText(kTitles[static_cast<size_t>(i)], 10 + i * knobW, getHeight() - 68, knobW, 13,
                       juce::Justification::centred, false);
        }
    }

    void mouseDown(const juce::MouseEvent& e) override { scrubGeneBar(e.position); }
    void mouseDrag(const juce::MouseEvent& e) override { scrubGeneBar(e.position); }

private:
    void configureRotary(juce::Slider& s, double minV, double maxV, double initV)
    {
        s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        s.setRange(minV, maxV, 0.01);
        s.setValue(initV, juce::dontSendNotification);
        addAndMakeVisible(s);
    }

    void scrubGeneBar(juce::Point<float> pt)
    {
        if (!genomeBarsArea_.toFloat().contains(pt))
            return;

        const float barW = static_cast<float>(genomeBarsArea_.getWidth()) / static_cast<float>(LayoutMetrics::kGenomeDimension);
        const int idx = std::clamp(static_cast<int>((pt.x - static_cast<float>(genomeBarsArea_.getX())) / barW),
                                   0, LayoutMetrics::kGenomeDimension - 1);
        const float val = std::clamp(1.0f - (pt.y - static_cast<float>(genomeBarsArea_.getY()))
                                     / static_cast<float>(genomeBarsArea_.getHeight()), 0.0f, 1.0f);
        genome_[static_cast<size_t>(idx)] = val;
        if (onGenomeEdited_)
            onGenomeEdited_(genome_);
        repaint();
    }

    ChimeraRadialHub::GenomeVector genome_ {};
    juce::Rectangle<int> genomeBarsArea_;
    juce::ComboBox quantScaleBox_;
    juce::Slider lfo1RateSlider_, lfo1DepthSlider_, lfo2DriftSlider_, quantAmtSlider_;
    GenomeEditCallback onGenomeEdited_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GenomeLfoQuantizerQuadrant)
};

// =============================================================================
// COMPLETE LAYER 2 CONTAINER: 4-QUADRANT DEEP SPECTRAL RULE MATRIX
// =============================================================================

class DeepRuleMatrixView final : public juce::Component
{
public:
    DeepRuleMatrixView()
    {
        addAndMakeVisible(q1Partitioning_);
        addAndMakeVisible(q2RuleMatrix_);
        addAndMakeVisible(q3SourceFilter_);
        addAndMakeVisible(q4GenomeLfo_);
    }

    SpectralPartitioningQuadrant& getQ1Partitioning() noexcept { return q1Partitioning_; }
    RuleAllocationMatrixQuadrant& getQ2RuleMatrix()   noexcept { return q2RuleMatrix_; }
    SourceFilterPhaseQuadrant&    getQ3SourceFilter() noexcept { return q3SourceFilter_; }
    GenomeLfoQuantizerQuadrant&   getQ4GenomeLfo()    noexcept { return q4GenomeLfo_; }

    void resized() override
    {
        auto b = getLocalBounds();
        const int halfW = b.getWidth() / 2;
        const int halfH = b.getHeight() / 2;

        auto topRow = b.removeFromTop(halfH);
        q1Partitioning_.setBounds(topRow.removeFromLeft(halfW).reduced(4));
        q2RuleMatrix_.setBounds(topRow.reduced(4));

        q3SourceFilter_.setBounds(b.removeFromLeft(halfW).reduced(4));
        q4GenomeLfo_.setBounds(b.reduced(4));
    }

private:
    SpectralPartitioningQuadrant q1Partitioning_;
    RuleAllocationMatrixQuadrant q2RuleMatrix_;
    SourceFilterPhaseQuadrant    q3SourceFilter_;
    GenomeLfoQuantizerQuadrant   q4GenomeLfo_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeepRuleMatrixView)
};

} // namespace spectile::ui
