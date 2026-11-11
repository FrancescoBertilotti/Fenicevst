#pragma once

#include "ThemeOKLCH.hpp"
#include "PolychromeSpectrogramGL.hpp"
#include "ChimeraRadialHub.hpp"
#include "DeepRuleMatrixView.hpp"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <array>
#include <memory>

namespace spectile::ui
{

// =============================================================================
// 1. DRAG-AND-DROP MODULATION SOURCE PILL & MACRO KNOB WITH MODULATION RINGS
// =============================================================================

class ModSourcePill final : public juce::Component
{
public:
    ModSourcePill(int sourceIndex, const juce::String& label, juce::Colour accent)
        : sourceIndex_(sourceIndex), label_(label), accent_(accent)
    {
        setMouseCursor(juce::MouseCursor::DraggingHandCursor);
    }

    void paint(juce::Graphics& g) override
    {
        const auto b = getLocalBounds().toFloat().reduced(1.5f);
        g.setColour(isHovered_ ? ThemeOKLCH::elevatedCard().brighter(0.18f) : ThemeOKLCH::elevatedCard());
        g.fillRoundedRectangle(b, 4.0f);

        g.setColour(accent_.withAlpha(isHovered_ ? 1.0f : 0.78f));
        g.drawRoundedRectangle(b, 4.0f, isHovered_ ? 1.6f : 1.1f);

        g.setColour(ThemeOKLCH::textPrimary());
        g.setFont(juce::FontOptions(10.5f, juce::Font::bold));
        g.drawText(label_, getLocalBounds(), juce::Justification::centred, false);
    }

    void mouseEnter(const juce::MouseEvent&) override { isHovered_ = true;  repaint(); }
    void mouseExit(const juce::MouseEvent&) override  { isHovered_ = false; repaint(); }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (e.getDistanceFromDragStart() > 3)
        {
            if (auto* container = juce::DragAndDropContainer::findParentDragContainerFor(this))
            {
                if (!container->isDragAndDropActive())
                {
                    container->startDragging("MOD_SRC_" + juce::String(sourceIndex_), this);
                }
            }
        }
    }

private:
    int sourceIndex_ { 0 };
    juce::String label_;
    juce::Colour accent_;
    bool isHovered_ { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ModSourcePill)
};

class MacroKnobWithModRing final : public juce::Component,
                                   public juce::DragAndDropTarget
{
public:
    MacroKnobWithModRing(const juce::String& title,
                         const juce::String& subtitle,
                         double defaultVal,
                         juce::Colour ringAccent)
        : title_(title), subtitle_(subtitle), accent_(ringAccent)
    {
        slider_.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        slider_.setRange(0.0, 1.0, 0.001);
        slider_.setValue(defaultVal, juce::dontSendNotification);
        slider_.setColour(juce::Slider::rotarySliderFillColourId, accent_);
        addAndMakeVisible(slider_);
    }

    juce::Slider& getSlider() noexcept { return slider_; }

    void setModulationAssignment(int modSourceIndex, float depth)
    {
        if (modSourceIndex >= 0 && modSourceIndex < 8)
        {
            modDepths_[static_cast<size_t>(modSourceIndex)] = std::clamp(depth, -1.0f, 1.0f);
            repaint();
        }
    }

    // DragAndDropTarget implementation
    bool isInterestedInDragSource(const SourceDetails& dragSourceDetails) override
    {
        return dragSourceDetails.description.toString().startsWith("MOD_SRC_");
    }

    void itemDragEnter(const SourceDetails&) override { isDragTargetHover_ = true;  repaint(); }
    void itemDragExit(const SourceDetails&) override  { isDragTargetHover_ = false; repaint(); }

    void itemDropped(const SourceDetails& dragSourceDetails) override
    {
        isDragTargetHover_ = false;
        const juce::String desc = dragSourceDetails.description.toString();
        const int srcIdx = desc.fromFirstOccurrenceOf("MOD_SRC_", false, false).getIntValue();
        if (srcIdx >= 0 && srcIdx < 8)
        {
            const float current = modDepths_[static_cast<size_t>(srcIdx)];
            modDepths_[static_cast<size_t>(srcIdx)] = (current > 0.05f) ? 0.0f : 0.35f;
        }
        repaint();
    }

    void resized() override
    {
        auto b = getLocalBounds();
        b.removeFromRight(b.getWidth() - 78);
        slider_.setBounds(b.reduced(4));
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat().reduced(1.5f);
        g.setColour(isDragTargetHover_ ? ThemeOKLCH::elevatedCard().brighter(0.15f) : ThemeOKLCH::panelSlate());
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);

        g.setColour(isDragTargetHover_ ? accent_ : ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds, LayoutMetrics::kCornerRadius, isDragTargetHover_ ? 1.6f : 1.0f);

        // Draw outer modulation rings around the rotary slider
        const auto sliderBounds = slider_.getBounds().toFloat();
        const auto centre = sliderBounds.getCentre();
        const float outerR = std::min(sliderBounds.getWidth(), sliderBounds.getHeight()) * 0.48f;

        const float startAngle = slider_.getRotaryParameters().startAngleRadians;
        const float endAngle   = slider_.getRotaryParameters().endAngleRadians;
        const float valNorm    = static_cast<float>(slider_.valueToProportionOfLength(slider_.getValue()));
        const float valAngle   = startAngle + valNorm * (endAngle - startAngle);

        int activeRing = 0;
        for (int m = 0; m < 8; ++m)
        {
            const float depth = modDepths_[static_cast<size_t>(m)];
            if (std::abs(depth) < 0.01f)
                continue;

            const float ringR = outerR + 2.0f + static_cast<float>(activeRing) * 2.5f;
            const float modEndAngle = std::clamp(valAngle + depth * (endAngle - startAngle), startAngle, endAngle);

            juce::Path modArc;
            modArc.addCentredArc(centre.x, centre.y, ringR, ringR, 0.0f,
                                 std::min(valAngle, modEndAngle), std::max(valAngle, modEndAngle), true);
            g.setColour(ThemeOKLCH::getSlotColour(m, 0.90f));
            g.strokePath(modArc, juce::PathStrokeType(2.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
            ++activeRing;
        }

        // Title, subtitle & live percentage readout
        const int textX = 80;
        const int textW = getWidth() - textX - 8;

        g.setColour(ThemeOKLCH::textPrimary());
        g.setFont(juce::FontOptions(12.5f, juce::Font::bold));
        g.drawText(title_, textX, 12, textW, 16, juce::Justification::centredLeft, false);

        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(10.5f));
        g.drawText(subtitle_, textX, 30, textW, 14, juce::Justification::centredLeft, false);

        g.setColour(accent_);
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        const int pct = juce::roundToInt(slider_.getValue() * 100.0);
        g.drawText(juce::String(pct) + " %", textX, 48, textW, 16, juce::Justification::centredLeft, false);
    }

private:
    juce::String title_;
    juce::String subtitle_;
    juce::Colour accent_;
    juce::Slider slider_;
    std::array<float, 8> modDepths_ { 0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    bool isDragTargetHover_ { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MacroKnobWithModRing)
};

// =============================================================================
// 2. MAIN DUAL-LAYER PLUGIN EDITOR (SpecTileAudioProcessorEditor)
// =============================================================================

class SpecTileAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                           public juce::DragAndDropContainer,
                                           private juce::Timer
{
public:
    explicit SpecTileAudioProcessorEditor(
        juce::AudioProcessor& processor,
        juce::AudioProcessorValueTreeState* apvts = nullptr,
        LockFreeSpscRingBuffer<SpectralColumnPacket, 128>* externalSpscQueue = nullptr);

    ~SpecTileAudioProcessorEditor() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // External accessors for processor / test harnesses
    PolychromeSpectrogramGL& getSpectrogramGL() noexcept { return *spectrogramGL_; }
    ChimeraRadialHub&        getChimeraHub()    noexcept { return chimeraHub_; }
    DeepRuleMatrixView&      getDeepMatrix()    noexcept { return deepMatrixView_; }

    void pushSpectralColumnPacket(const SpectralColumnPacket& pkt) noexcept
    {
        activeSpscQueue_->tryPush(pkt);
    }

private:
    void timerCallback() override;
    void updateDualLayerGeometry();

    ObsidianSpectralLookAndFeel lookAndFeel_;
    juce::AudioProcessorValueTreeState* apvts_ { nullptr };

    // Internal fallback SPSC queue if processor does not pass an external queue pointer
    LockFreeSpscRingBuffer<SpectralColumnPacket, 128> internalSpscQueue_;
    LockFreeSpscRingBuffer<SpectralColumnPacket, 128>* activeSpscQueue_ { nullptr };

    // Top Bar Controls
    juce::ComboBox playModeSelector_;
    juce::ComboBox archetypeSelector_;
    juce::TextButton viewToggleBtn_;

    // Central GPU Spectrogram + Layer 1 (Chimera Radial Hub + 8 Sample Slots) + Layer 2 (Deep Rule Matrix)
    std::unique_ptr<PolychromeSpectrogramGL> spectrogramGL_;
    ChimeraRadialHub chimeraHub_;
    DeepRuleMatrixView deepMatrixView_;
    std::array<std::unique_ptr<SpectralSampleSlotCard>, LayoutMetrics::kSlotCount> slotCards_;

    // Bottom Bar: 3 Macro Knobs + 8 Drag-and-Drop Modulation Sources
    std::unique_ptr<MacroKnobWithModRing> macroTileSize_;
    std::unique_ptr<MacroKnobWithModRing> macroRuleMutation_;
    std::unique_ptr<MacroKnobWithModRing> macroPhaseGlue_;
    std::array<std::unique_ptr<ModSourcePill>, 8> modSourcePills_;

    // Animated transition state between Layer 1 (0.0f) and Layer 2 (1.0f)
    bool isDeepMatrixView_ { false };
    float layerMorphProgress_ { 0.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpecTileAudioProcessorEditor)
};

} // namespace spectile::ui
