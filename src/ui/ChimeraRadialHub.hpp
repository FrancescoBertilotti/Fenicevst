#pragma once

#include "ThemeOKLCH.hpp"
#include <juce_gui_basics/juce_gui_basics.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <string>

namespace spectile::ui
{

// =============================================================================
// 1. SPECTRAL SAMPLE SLOT STATE & SIDE-DOCK CARD (S1 .. S8)
// =============================================================================

struct SampleSlotUIState
{
    int slotIndex { 0 };
    juce::String sampleName { "Empty Slot (Drop Audio)" };
    bool loaded { false };
    bool filterH { true };
    bool filterP { true };
    bool filterR { true };
    bool locked  { false };
    float gainDb { 0.0f };
    float pitchSemis { 0.0f };
    float loopStartNorm { 0.0f };
    float loopEndNorm   { 1.0f };
    float conqueredRatio { 0.0f };
};

class SpectralSampleSlotCard final : public juce::Component,
                                     public juce::FileDragAndDropTarget
{
public:
    using SlotStateCallback = std::function<void(int slotIndex, const SampleSlotUIState& state)>;
    using FileDropCallback  = std::function<void(int slotIndex, const juce::File& audioFile)>;

    explicit SpectralSampleSlotCard(int slotIndex)
    {
        state_.slotIndex = std::clamp(slotIndex, 0, LayoutMetrics::kSlotCount - 1);
        state_.loaded    = true;

        static constexpr std::array<const char*, 8> kDefaultSampleNames {{
            "Reese_Sub_Neuro.wav",
            "Tuvan_Throat_Formant.wav",
            "Anvil_Transient_Hit.wav",
            "Cello_Sul_Ponticello.wav",
            "Bowed_Crotales_High.wav",
            "Fire_Crackle_Foley.wav",
            "Arctic_Wind_Noise.wav",
            "Glass_Harmonica_C4.wav"
        }};
        state_.sampleName = kDefaultSampleNames[static_cast<size_t>(state_.slotIndex)];

        const juce::Colour slotCol = ThemeOKLCH::getSlotColour(state_.slotIndex);

        auto configureToggle = [this, slotCol](juce::TextButton& btn, const juce::String& text, bool initial)
        {
            btn.setButtonText(text);
            btn.setClickingTogglesState(true);
            btn.setToggleState(initial, juce::dontSendNotification);
            btn.setColour(juce::TextButton::buttonOnColourId, slotCol.withAlpha(0.32f));
            btn.setColour(juce::TextButton::textColourOnId, ThemeOKLCH::textPrimary());
            addAndMakeVisible(btn);
        };

        configureToggle(btnH_, "H", true);
        configureToggle(btnP_, "P", true);
        configureToggle(btnR_, "R", true);
        configureToggle(btnLock_, "LOCK", false);
        btnLock_.setColour(juce::TextButton::buttonOnColourId, ThemeOKLCH::accentLockGold().withAlpha(0.35f));

        btnH_.onClick = [this] { state_.filterH = btnH_.getToggleState(); notifyStateChange(); };
        btnP_.onClick = [this] { state_.filterP = btnP_.getToggleState(); notifyStateChange(); };
        btnR_.onClick = [this] { state_.filterR = btnR_.getToggleState(); notifyStateChange(); };
        btnLock_.onClick = [this] { state_.locked = btnLock_.getToggleState(); notifyStateChange(); repaint(); };

        generateMiniSpectrogramThumbnail();
    }

    void setConqueredRatio(float ratio) noexcept
    {
        const float clamped = std::clamp(ratio, 0.0f, 1.0f);
        if (std::abs(clamped - state_.conqueredRatio) > 0.002f)
        {
            state_.conqueredRatio = clamped;
            repaint();
        }
    }

    const SampleSlotUIState& getState() const noexcept { return state_; }

    void setState(const SampleSlotUIState& newState)
    {
        state_ = newState;
        btnH_.setToggleState(state_.filterH, juce::dontSendNotification);
        btnP_.setToggleState(state_.filterP, juce::dontSendNotification);
        btnR_.setToggleState(state_.filterR, juce::dontSendNotification);
        btnLock_.setToggleState(state_.locked, juce::dontSendNotification);
        repaint();
    }

    void setOnStateChanged(SlotStateCallback cb) { onStateChanged_ = std::move(cb); }
    void setOnFileDropped(FileDropCallback cb)   { onFileDropped_  = std::move(cb); }

    // FileDragAndDropTarget implementation
    bool isInterestedInFileDrag(const juce::StringArray& files) override
    {
        for (const auto& f : files)
        {
            if (f.endsWithIgnoreCase(".wav") || f.endsWithIgnoreCase(".flac")
                || f.endsWithIgnoreCase(".aiff") || f.endsWithIgnoreCase(".ogg"))
                return true;
        }
        return false;
    }

    void fileDragEnter(const juce::StringArray&, int, int) override { isFileDragHover_ = true; repaint(); }
    void fileDragExit(const juce::StringArray&) override            { isFileDragHover_ = false; repaint(); }

    void filesDropped(const juce::StringArray& files, int, int) override
    {
        isFileDragHover_ = false;
        if (!files.isEmpty())
        {
            const juce::File file(files[0]);
            state_.sampleName = file.getFileName();
            state_.loaded = true;
            generateMiniSpectrogramThumbnail();
            if (onFileDropped_)
                onFileDropped_(state_.slotIndex, file);
            notifyStateChange();
        }
        repaint();
    }

    void resized() override
    {
        auto b = getLocalBounds().reduced(6);
        auto topRow = b.removeFromTop(20);
        btnLock_.setBounds(topRow.removeFromRight(42));
        topRow.removeFromRight(3);
        btnR_.setBounds(topRow.removeFromRight(22));
        topRow.removeFromRight(2);
        btnP_.setBounds(topRow.removeFromRight(22));
        topRow.removeFromRight(2);
        btnH_.setBounds(topRow.removeFromRight(22));

        b.removeFromTop(4);
        b.removeFromBottom(18); // Conquered bar + Pitch/Gain readout
        miniSpecBounds_ = b.toFloat();
    }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat().reduced(1.0f);
        const juce::Colour slotCol = ThemeOKLCH::getSlotColour(state_.slotIndex);

        // Card background
        g.setColour(isFileDragHover_ ? ThemeOKLCH::elevatedCard().brighter(0.15f) : ThemeOKLCH::panelSlate());
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);

        // Left vertical OKLCH color stripe
        const juce::Rectangle<float> colorBar(bounds.getX(), bounds.getY(), 4.5f, bounds.getHeight());
        g.setColour(slotCol);
        g.fillRoundedRectangle(colorBar, 2.0f);

        // Header label: "S1  SampleName.wav"
        const auto& info = ThemeOKLCH::kSlotPalette[static_cast<size_t>(state_.slotIndex)];
        g.setColour(slotCol);
        g.setFont(juce::FontOptions(12.0f, juce::Font::bold));
        g.drawText(juce::String(std::string(info.code)), 10, 6, 24, 18, juce::Justification::centredLeft, false);

        g.setColour(ThemeOKLCH::textPrimary());
        g.setFont(juce::FontOptions(11.5f));
        g.drawText(state_.sampleName, 34, 6, getWidth() - 160, 18, juce::Justification::centredLeft, true);

        // Mini-spectrogram preview with loop handles
        if (!miniSpecBounds_.isEmpty())
        {
            g.setColour(ThemeOKLCH::obsidianDeep());
            g.fillRoundedRectangle(miniSpecBounds_, 3.5f);
            g.drawImage(miniSpecImage_, miniSpecBounds_, juce::RectanglePlacement::stretchToFit);

            // Dim outside loop region
            const float lx0 = miniSpecBounds_.getX() + state_.loopStartNorm * miniSpecBounds_.getWidth();
            const float lx1 = miniSpecBounds_.getX() + state_.loopEndNorm   * miniSpecBounds_.getWidth();

            g.setColour(juce::Colours::black.withAlpha(0.55f));
            g.fillRect(miniSpecBounds_.getX(), miniSpecBounds_.getY(), lx0 - miniSpecBounds_.getX(), miniSpecBounds_.getHeight());
            g.fillRect(lx1, miniSpecBounds_.getY(), miniSpecBounds_.getRight() - lx1, miniSpecBounds_.getHeight());

            // Loop bracket lines
            g.setColour(slotCol.withAlpha(0.9f));
            g.drawVerticalLine(juce::roundToInt(lx0), miniSpecBounds_.getY(), miniSpecBounds_.getBottom());
            g.drawVerticalLine(juce::roundToInt(lx1), miniSpecBounds_.getY(), miniSpecBounds_.getBottom());

            g.setColour(ThemeOKLCH::borderSubtle());
            g.drawRoundedRectangle(miniSpecBounds_, 3.5f, 1.0f);
        }

        // Bottom row: Real-time Conquered Bins % Bar + Gain/Pitch pill
        const float barY = bounds.getBottom() - 15.0f;
        const juce::Rectangle<float> barTrack(10.0f, barY + 3.0f, bounds.getWidth() - 108.0f, 7.0f);
        g.setColour(ThemeOKLCH::obsidianDeep());
        g.fillRoundedRectangle(barTrack, 3.0f);

        const float fillW = barTrack.getWidth() * std::clamp(state_.conqueredRatio, 0.0f, 1.0f);
        if (fillW > 1.0f)
        {
            g.setColour(slotCol);
            g.fillRoundedRectangle(barTrack.withWidth(fillW), 3.0f);
        }

        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
        const int pct = juce::roundToInt(state_.conqueredRatio * 100.0f);
        const juce::String metaText = juce::String(pct) + "% | "
            + (state_.pitchSemis >= 0.0f ? "+" : "") + juce::String(state_.pitchSemis, 1) + "st";
        g.drawText(metaText, static_cast<int>(barTrack.getRight()) + 4, static_cast<int>(barY) - 1, 90, 14,
                   juce::Justification::centredRight, false);

        // Card border (highlighted in gold if locked)
        g.setColour(state_.locked ? ThemeOKLCH::accentLockGold() : ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds, LayoutMetrics::kCornerRadius, state_.locked ? 1.5f : 1.0f);
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        if (miniSpecBounds_.contains(e.position))
        {
            const float normX = std::clamp((e.position.x - miniSpecBounds_.getX()) / miniSpecBounds_.getWidth(), 0.0f, 1.0f);
            draggingLoopStart_ = (std::abs(normX - state_.loopStartNorm) < std::abs(normX - state_.loopEndNorm));
            mouseDrag(e);
        }
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (!miniSpecBounds_.isEmpty() && e.mods.isLeftButtonDown())
        {
            const float normX = std::clamp((e.position.x - miniSpecBounds_.getX()) / miniSpecBounds_.getWidth(), 0.0f, 1.0f);
            if (draggingLoopStart_)
                state_.loopStartNorm = std::min(normX, state_.loopEndNorm - 0.05f);
            else
                state_.loopEndNorm = std::max(normX, state_.loopStartNorm + 0.05f);
            notifyStateChange();
            repaint();
        }
    }

private:
    void notifyStateChange()
    {
        if (onStateChanged_)
            onStateChanged_(state_.slotIndex, state_);
    }

    void generateMiniSpectrogramThumbnail()
    {
        constexpr int w = 96;
        constexpr int h = 32;
        miniSpecImage_ = juce::Image(juce::Image::ARGB, w, h, true);
        const juce::Colour col = ThemeOKLCH::getSlotColour(state_.slotIndex);
        const float seed = static_cast<float>(state_.slotIndex + 1) * 1.37f;

        for (int x = 0; x < w; ++x)
        {
            const float tx = static_cast<float>(x) / static_cast<float>(w);
            const float env = std::exp(-2.2f * tx) * (0.6f + 0.4f * std::sin(tx * 18.0f + seed));
            for (int y = 0; y < h; ++y)
            {
                const float fy = 1.0f - static_cast<float>(y) / static_cast<float>(h);
                const float ridge = 0.5f + 0.5f * std::cos(fy * 22.0f + seed * 2.0f + tx * 5.0f);
                const float val = std::clamp(env * ridge * std::exp(-1.8f * (fy - 0.35f) * (fy - 0.35f)), 0.0f, 1.0f);
                miniSpecImage_.setPixelAt(x, y, col.withAlpha(val * 0.88f));
            }
        }
    }

    SampleSlotUIState state_;
    juce::TextButton btnH_, btnP_, btnR_, btnLock_;
    juce::Rectangle<float> miniSpecBounds_;
    juce::Image miniSpecImage_;
    bool isFileDragHover_ { false };
    bool draggingLoopStart_ { true };

    SlotStateCallback onStateChanged_;
    FileDropCallback  onFileDropped_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectralSampleSlotCard)
};

// =============================================================================
// 2. CENTRAL RADIAL EVOLUTIONARY HUB (6 QR-ORTHOGONAL BRANCHES + SLERP)
// =============================================================================

class ChimeraRadialHub final : public juce::Component,
                               private juce::Timer
{
public:
    static constexpr size_t kGenomeDim  = static_cast<size_t>(LayoutMetrics::kGenomeDimension);
    static constexpr size_t kNumBranches = static_cast<size_t>(LayoutMetrics::kRadialBranches);

    using GenomeVector = std::array<float, kGenomeDim>;
    using GenomeChangedCallback = std::function<void(const GenomeVector& genome, float rho, float thetaRad)>;

    ChimeraRadialHub()
    {
        setInterceptsMouseClicks(true, true);
        initializeArchetypeSeeds();
        selectArchetype(0);
        startTimerHz(60);
    }

    ~ChimeraRadialHub() override
    {
        stopTimer();
    }

    void selectArchetype(int archetypeIndex)
    {
        activeArchetype_ = std::clamp(archetypeIndex, 0, 5);
        centerGenome_    = archetypeSeeds_[static_cast<size_t>(activeArchetype_)];
        cursorRho_       = 0.0f;
        cursorTheta_     = 0.0f;
        buildOrthonormalQRBranches();
        evaluateSlerpGenome();
        repaint();
    }

    void triggerMutateButton()
    {
        // Perturb center genome within the active archetype manifold and rebuild QR branches
        std::uniform_real_distribution<float> dist(-0.14f, 0.14f);
        const auto& baseSeed = archetypeSeeds_[static_cast<size_t>(activeArchetype_)];
        for (size_t d = 0; d < kGenomeDim; ++d)
        {
            centerGenome_[d] = std::clamp(0.65f * baseSeed[d] + 0.35f * currentGenome_[d] + dist(rng_), 0.02f, 0.98f);
        }
        cursorRho_ = 0.0f;
        mutatePulsePhase_ = 1.0f;
        buildOrthonormalQRBranches();
        evaluateSlerpGenome();
        repaint();
    }

    void promoteCursorToCenter()
    {
        centerGenome_ = currentGenome_;
        cursorRho_    = 0.0f;
        mutatePulsePhase_ = 1.0f;
        buildOrthonormalQRBranches();
        evaluateSlerpGenome();
        repaint();
    }

    const GenomeVector& getCurrentGenome() const noexcept { return currentGenome_; }
    void setCurrentGenome(const GenomeVector& g)
    {
        currentGenome_ = g;
        centerGenome_  = g;
        buildOrthonormalQRBranches();
        repaint();
    }

    float getCursorRho()   const noexcept { return cursorRho_; }
    float getCursorTheta() const noexcept { return cursorTheta_; }

    void setOnGenomeChanged(GenomeChangedCallback cb) { onGenomeChanged_ = std::move(cb); }

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();
        const auto centre = bounds.getCentre();
        const float maxRadius = std::min(bounds.getWidth(), bounds.getHeight()) * 0.41f;
        const float hubRadius = 44.0f;

        // Subtle semi-transparent radial radar backdrop over the GPU spectrogram
        juce::ColourGradient radarBg(
            ThemeOKLCH::obsidianDeep().withAlpha(0.76f), centre.x, centre.y,
            ThemeOKLCH::obsidianDeep().withAlpha(0.0f),  centre.x + maxRadius * 1.15f, centre.y, true);
        g.setGradientFill(radarBg);
        g.fillEllipse(centre.x - maxRadius * 1.15f, centre.y - maxRadius * 1.15f,
                      maxRadius * 2.3f, maxRadius * 2.3f);

        // Concentric evolutionary distance rings (rho = 0.33, 0.66, 1.0)
        for (float ringRho : { 0.33f, 0.66f, 1.0f })
        {
            const float r = hubRadius + ringRho * (maxRadius - hubRadius);
            g.setColour(ThemeOKLCH::borderSubtle().withAlpha(0.55f));
            g.drawEllipse(centre.x - r, centre.y - r, r * 2.0f, r * 2.0f, 1.0f);
        }

        // Draw the 6 Radial Genetic Branches (60 deg apart)
        for (size_t b = 0; b < kNumBranches; ++b)
        {
            const float angle = static_cast<float>(b) * (juce::MathConstants<float>::twoPi / 6.0f)
                              - juce::MathConstants<float>::halfPi;
            const juce::Point<float> pStart(centre.x + hubRadius * std::cos(angle),
                                            centre.y + hubRadius * std::sin(angle));
            const juce::Point<float> pEnd(centre.x + maxRadius * std::cos(angle),
                                          centre.y + maxRadius * std::sin(angle));

            const juce::Colour branchCol = ThemeOKLCH::getSlotColour(static_cast<int>(b));

            // Glowing branch axis line
            g.setColour(branchCol.withAlpha(0.38f));
            g.drawLine(pStart.x, pStart.y, pEnd.x, pEnd.y, 3.5f);
            g.setColour(branchCol.withAlpha(0.88f));
            g.drawLine(pStart.x, pStart.y, pEnd.x, pEnd.y, 1.6f);

            // Outer branch node & label
            const float nodeR = 8.0f;
            g.setColour(ThemeOKLCH::obsidianDeep());
            g.fillEllipse(pEnd.x - nodeR, pEnd.y - nodeR, nodeR * 2.0f, nodeR * 2.0f);
            g.setColour(branchCol);
            g.drawEllipse(pEnd.x - nodeR, pEnd.y - nodeR, nodeR * 2.0f, nodeR * 2.0f, 2.0f);

            const float labelDist = maxRadius + 24.0f;
            const float lx = centre.x + labelDist * std::cos(angle);
            const float ly = centre.y + labelDist * std::sin(angle);
            const juce::Rectangle<float> labelBox(lx - 68.0f, ly - 10.0f, 136.0f, 20.0f);

            g.setColour(ThemeOKLCH::obsidianDeep().withAlpha(0.82f));
            g.fillRoundedRectangle(labelBox, 4.0f);
            g.setColour(branchCol.withAlpha(0.75f));
            g.drawRoundedRectangle(labelBox, 4.0f, 1.0f);

            g.setColour(ThemeOKLCH::textPrimary());
            g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
            g.drawText(juce::String(std::string(ThemeOKLCH::kBranchNames[b])),
                       labelBox.toNearestInt(), juce::Justification::centred, false);
        }

        // Draw active SLERP cursor position (rho, theta)
        if (cursorRho_ > 0.005f)
        {
            const float r = hubRadius + cursorRho_ * (maxRadius - hubRadius);
            const float drawAngle = cursorTheta_ - juce::MathConstants<float>::halfPi;
            const juce::Point<float> cursorPt(centre.x + r * std::cos(drawAngle),
                                              centre.y + r * std::sin(drawAngle));

            g.setColour(ThemeOKLCH::accentLaser().withAlpha(0.65f));
            g.drawLine(centre.x, centre.y, cursorPt.x, cursorPt.y, 2.2f);

            g.setColour(ThemeOKLCH::accentLaser().withAlpha(0.28f));
            g.fillEllipse(cursorPt.x - 16.0f, cursorPt.y - 16.0f, 32.0f, 32.0f);

            g.setColour(ThemeOKLCH::obsidianDeep());
            g.fillEllipse(cursorPt.x - 8.5f, cursorPt.y - 8.5f, 17.0f, 17.0f);
            g.setColour(ThemeOKLCH::accentLaser());
            g.drawEllipse(cursorPt.x - 8.5f, cursorPt.y - 8.5f, 17.0f, 17.0f, 2.5f);
        }

        // Pulse wave when Mutate is clicked
        if (mutatePulsePhase_ > 0.01f)
        {
            const float pulseR = hubRadius + (1.0f - mutatePulsePhase_) * (maxRadius - hubRadius);
            g.setColour(ThemeOKLCH::accentMutate().withAlpha(mutatePulsePhase_ * 0.65f));
            g.drawEllipse(centre.x - pulseR, centre.y - pulseR, pulseR * 2.0f, pulseR * 2.0f, 2.5f);
        }

        // Central "THE CHIMERA BUTTON" (Mutate / Generate Spectrum)
        juce::ColourGradient hubGrad(
            ThemeOKLCH::elevatedCard().brighter(isHoveringCenter_ ? 0.22f : 0.08f),
            centre.x, centre.y - hubRadius,
            ThemeOKLCH::obsidianDeep(),
            centre.x, centre.y + hubRadius, false);
        g.setGradientFill(hubGrad);
        g.fillEllipse(centre.x - hubRadius, centre.y - hubRadius, hubRadius * 2.0f, hubRadius * 2.0f);

        g.setColour(isHoveringCenter_ ? ThemeOKLCH::accentMutate() : ThemeOKLCH::accentLaser());
        g.drawEllipse(centre.x - hubRadius, centre.y - hubRadius, hubRadius * 2.0f, hubRadius * 2.0f, 2.4f);

        g.setColour(ThemeOKLCH::textPrimary());
        g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
        g.drawText("MUTATE", static_cast<int>(centre.x - hubRadius), static_cast<int>(centre.y - 14.0f),
                   static_cast<int>(hubRadius * 2.0f), 16, juce::Justification::centred, false);

        g.setColour(ThemeOKLCH::textSecondary());
        g.setFont(juce::FontOptions(9.5f));
        const juce::String subLabel = (cursorRho_ > 0.02f)
            ? ("RHO " + juce::String(juce::roundToInt(cursorRho_ * 100.0f)) + "%")
            : "CHIMERA";
        g.drawText(subLabel, static_cast<int>(centre.x - hubRadius), static_cast<int>(centre.y + 2.0f),
                   static_cast<int>(hubRadius * 2.0f), 14, juce::Justification::centred, false);
    }

    void mouseMove(const juce::MouseEvent& e) override
    {
        const auto centre = getLocalBounds().toFloat().getCentre();
        const float dist = centre.getDistanceFrom(e.position);
        const bool hov = (dist <= 44.0f);
        if (hov != isHoveringCenter_)
        {
            isHoveringCenter_ = hov;
            setMouseCursor(hov ? juce::MouseCursor::PointingHandCursor : juce::MouseCursor::CrosshairCursor);
            repaint();
        }
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        const auto centre = getLocalBounds().toFloat().getCentre();
        const float dist = centre.getDistanceFrom(e.position);

        if (dist <= 44.0f)
        {
            draggingCursor_ = false;
            triggerMutateButton();
        }
        else
        {
            draggingCursor_ = true;
            updateCursorFromMouse(e.position);
        }
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (draggingCursor_)
            updateCursorFromMouse(e.position);
    }

    void mouseUp(const juce::MouseEvent&) override
    {
        draggingCursor_ = false;
    }

    void mouseDoubleClick(const juce::MouseEvent& e) override
    {
        const auto centre = getLocalBounds().toFloat().getCentre();
        if (centre.getDistanceFrom(e.position) > 44.0f)
        {
            promoteCursorToCenter();
        }
    }

private:
    void timerCallback() override
    {
        if (mutatePulsePhase_ > 0.005f)
        {
            mutatePulsePhase_ *= 0.88f;
            if (mutatePulsePhase_ <= 0.005f)
                mutatePulsePhase_ = 0.0f;
            repaint();
        }
    }

    void updateCursorFromMouse(juce::Point<float> pos)
    {
        const auto bounds = getLocalBounds().toFloat();
        const auto centre = bounds.getCentre();
        const float maxRadius = std::min(bounds.getWidth(), bounds.getHeight()) * 0.41f;
        const float hubRadius = 44.0f;

        const float dx = pos.x - centre.x;
        const float dy = pos.y - centre.y;
        const float dist = std::sqrt(dx * dx + dy * dy);

        cursorRho_ = std::clamp((dist - hubRadius) / std::max(1.0f, maxRadius - hubRadius), 0.0f, 1.0f);

        float angle = std::atan2(dy, dx) + juce::MathConstants<float>::halfPi;
        if (angle < 0.0f)
            angle += juce::MathConstants<float>::twoPi;

        // Magnetic snap within +/- 6 degrees of any of the 6 radial branch axes
        constexpr float kBranchStep = juce::MathConstants<float>::twoPi / 6.0f;
        const float nearestBranchAngle = std::round(angle / kBranchStep) * kBranchStep;
        if (std::abs(angle - nearestBranchAngle) < 0.105f)
            angle = std::fmod(nearestBranchAngle + juce::MathConstants<float>::twoPi, juce::MathConstants<float>::twoPi);

        cursorTheta_ = angle;
        evaluateSlerpGenome();
        repaint();
    }

    static float logit(float p) noexcept
    {
        const float clamped = std::clamp(p, 0.01f, 0.99f);
        return std::log(clamped / (1.0f - clamped));
    }

    static float sigmoid(float z) noexcept
    {
        return 1.0f / (1.0f + std::exp(-std::clamp(z, -12.0f, 12.0f)));
    }

    void initializeArchetypeSeeds()
    {
        // 6 Musical Archetypes in [0, 1]^24
        for (size_t a = 0; a < 6; ++a)
        {
            for (size_t d = 0; d < kGenomeDim; ++d)
            {
                const float phase = static_cast<float>(a + 1) * 1.15f + static_cast<float>(d) * 0.48f;
                archetypeSeeds_[a][d] = std::clamp(0.5f + 0.38f * std::sin(phase), 0.05f, 0.95f);
            }
        }
    }

    // Gram-Schmidt QR Orthonormalization of the 6 semantic tangent directions in R^24
    void buildOrthonormalQRBranches()
    {
        for (size_t b = 0; b < kNumBranches; ++b)
        {
            for (size_t d = 0; d < kGenomeDim; ++d)
            {
                const float freq = static_cast<float>(b + 1) * 0.75f;
                branchBasisU_[b][d] = std::cos(static_cast<float>(d) * freq + static_cast<float>(b) * 1.047f);
            }

            // Orthogonalize against previous branches 0 .. b-1
            for (size_t prev = 0; prev < b; ++prev)
            {
                float dot = 0.0f;
                for (size_t d = 0; d < kGenomeDim; ++d)
                    dot += branchBasisU_[b][d] * branchBasisU_[prev][d];
                for (size_t d = 0; d < kGenomeDim; ++d)
                    branchBasisU_[b][d] -= dot * branchBasisU_[prev][d];
            }

            // Normalize to unit L2 norm
            float normSq = 0.0f;
            for (size_t d = 0; d < kGenomeDim; ++d)
                normSq += branchBasisU_[b][d] * branchBasisU_[b][d];
            const float invNorm = 1.0f / std::sqrt(std::max(1e-6f, normSq));
            for (size_t d = 0; d < kGenomeDim; ++d)
                branchBasisU_[b][d] *= invNorm;
        }
    }

    // Logit-Riemannian SLERP evaluation: g(rho, theta) = sigma(z_0 + rho * R_branch * u(theta))
    void evaluateSlerpGenome()
    {
        constexpr float kBranchStep = juce::MathConstants<float>::twoPi / 6.0f;
        constexpr float kBranchRadius = 3.2f;

        const float scaledTheta = cursorTheta_ / kBranchStep;
        const size_t b0 = static_cast<size_t>(std::floor(scaledTheta)) % kNumBranches;
        const size_t b1 = (b0 + 1) % kNumBranches;
        const float frac = scaledTheta - std::floor(scaledTheta);

        // Orthogonal spherical interpolation between orthonormal basis vectors u_{b0} and u_{b1}
        const float omega = juce::MathConstants<float>::halfPi * frac;
        const float w0 = std::cos(omega);
        const float w1 = std::sin(omega);

        for (size_t d = 0; d < kGenomeDim; ++d)
        {
            const float z0 = logit(centerGenome_[d]);
            const float uDir = w0 * branchBasisU_[b0][d] + w1 * branchBasisU_[b1][d];
            currentGenome_[d] = sigmoid(z0 + cursorRho_ * kBranchRadius * uDir);
        }

        if (onGenomeChanged_)
            onGenomeChanged_(currentGenome_, cursorRho_, cursorTheta_);
    }

    std::array<GenomeVector, 6> archetypeSeeds_ {};
    std::array<GenomeVector, kNumBranches> branchBasisU_ {};
    GenomeVector centerGenome_  {};
    GenomeVector currentGenome_ {};

    int activeArchetype_     { 0 };
    float cursorRho_         { 0.0f };
    float cursorTheta_       { 0.0f };
    float mutatePulsePhase_  { 0.0f };
    bool isHoveringCenter_   { false };
    bool draggingCursor_     { false };
    std::mt19937 rng_        { 0x5EC711E4u };

    GenomeChangedCallback onGenomeChanged_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChimeraRadialHub)
};

} // namespace spectile::ui
