#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace spectile::ui
{

// =============================================================================
// 1. OBSIDIAN SPECTRAL LAB — DARK HIGH-CONTRAST THEME & LAYOUT CONSTANTS
// =============================================================================

struct LayoutMetrics
{
    static constexpr int kDefaultWidth          = 1280;
    static constexpr int kDefaultHeight         = 800;
    static constexpr int kMinWidth              = 1024;
    static constexpr int kMinHeight             = 640;
    static constexpr int kMaxWidth              = 2560;
    static constexpr int kMaxHeight             = 1600;

    static constexpr int kTopBarHeight          = 52;
    static constexpr int kBottomBarHeight       = 124;
    static constexpr int kSideDockWidth         = 248;
    static constexpr int kSlotCount             = 8;
    static constexpr int kSlotsPerSide          = 4;
    static constexpr int kGenomeDimension       = 24;
    static constexpr int kRadialBranches        = 6;
    static constexpr int kBandLockCount         = 4;

    static constexpr float kCornerRadius        = 6.0f;
    static constexpr float kPanelBorderWidth    = 1.25f;
    static constexpr float kMatrixHeaderSpecRatio = 0.24f; // Top 24% spectrogram banner in Deep Matrix View
};

struct ObsidianPalette
{
    // Core Obsidian Spectral Lab surfaces (#0A0B0E, #12141C, #1A1D28)
    static constexpr uint32_t kBgDeepObsidian   = 0xFF0A0B0E;
    static constexpr uint32_t kBgPanelSlate     = 0xFF12141C;
    static constexpr uint32_t kBgElevatedCard   = 0xFF1A1D28;
    static constexpr uint32_t kBgGlassOverlay   = 0xD810121A;

    // Structural lines, grid hairlines & accents
    static constexpr uint32_t kBorderSubtle     = 0xFF252A3A;
    static constexpr uint32_t kBorderHighlight  = 0xFF3A425A;
    static constexpr uint32_t kGridHairline     = 0x334A5578;
    static constexpr uint32_t kTextPrimary      = 0xFFF0F4FC;
    static constexpr uint32_t kTextSecondary    = 0xFF9AA4B8;
    static constexpr uint32_t kTextMuted        = 0xFF5E677A;

    // Functional status accents
    static constexpr uint32_t kAccentLaser      = 0xFF00F5D4;
    static constexpr uint32_t kAccentMutate     = 0xFFFF2A85;
    static constexpr uint32_t kAccentWarning    = 0xFFFFB703;
    static constexpr uint32_t kAccentLockGold   = 0xFFFFD166;
};

// =============================================================================
// 2. PERCEPTUALLY UNIFORM OKLCH COLOR SPACE & 8 SAMPLE SLOT PALETTE
// =============================================================================
// All 8 sample slot colors share identical perceptual lightness (L = 0.74)
// and chroma (C = 0.17) in OKLCH space so that no sample visually dominates
// the polychromatic spectrogram at equal dB energy.

struct OKLCHColor
{
    float L { 0.74f }; // Perceptual Lightness in [0, 1]
    float C { 0.17f }; // Perceptual Chroma in [0, 0.4]
    float hDeg { 0.0f }; // Hue angle in degrees [0, 360)
};

struct RGBLinear
{
    float r { 0.0f };
    float g { 0.0f };
    float b { 0.0f };
};

struct SlotVisualIdentity
{
    int index;                  // 0..7 (S1..S8)
    std::string_view code;      // "S1" .. "S8"
    std::string_view name;      // Perceptual color name
    uint32_t argbHex;           // Calibrated sRGB hex representation
    OKLCHColor oklch;           // Iso-luminant OKLCH coordinates
};

class ThemeOKLCH
{
public:
    static constexpr std::array<SlotVisualIdentity, LayoutMetrics::kSlotCount> kSlotPalette {{
        { 0, "S1", "Cyan Ray",          0xFF00E5FF, { 0.74f, 0.17f, 205.0f } },
        { 1, "S2", "Amber Flare",       0xFFFF9100, { 0.74f, 0.17f,  58.0f } },
        { 2, "S3", "Magenta Pulse",     0xFFFF1744, { 0.74f, 0.17f,  15.0f } },
        { 3, "S4", "Emerald Harmonics", 0xFF00E676, { 0.74f, 0.17f, 148.0f } },
        { 4, "S5", "Violet Partial",    0xFFD500F9, { 0.74f, 0.17f, 315.0f } },
        { 5, "S6", "Solar Gold",        0xFFFFEA00, { 0.74f, 0.17f,  98.0f } },
        { 6, "S7", "Coral Dust",        0xFFFF5252, { 0.74f, 0.17f,  28.0f } },
        { 7, "S8", "Ice Indigo",        0xFF536DFE, { 0.74f, 0.17f, 268.0f } }
    }};

    static constexpr std::array<std::string_view, 6> kArchetypeNames {{
        "A1: Harmonic Interleave",
        "A2: Transient-Formant Graft",
        "A3: Voronoi Spectral Shatter",
        "A4: Complementarity Hole-Fill",
        "A5: Spectral Freeze Morph",
        "A6: Formant Choir Synthesis"
    }};

    static constexpr std::array<std::string_view, 6> kBranchNames {{
        "Role Rotation",
        "Voronoi Shatter",
        "Source-Filter Graft",
        "Transient/Sustain",
        "Perlin Liquefy",
        "Archetype Bridge"
    }};

    static constexpr std::array<std::string_view, 4> kPlayModeNames {{
        "Pitched MPE (16v)",
        "Spectral Drone",
        "Rhythmic Loop Sync",
        "Live Sidechain S1"
    }};

    // Converts OKLCH -> OKLab -> Linear sRGB (Björn Ottosson's reference matrices)
    static inline RGBLinear oklchToLinearSRGB(const OKLCHColor& c) noexcept
    {
        constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
        const float hRad = c.hDeg * kDegToRad;
        const float a = c.C * std::cos(hRad);
        const float b = c.C * std::sin(hRad);

        const float l_ = c.L + 0.3963377774f * a + 0.2158037573f * b;
        const float m_ = c.L - 0.1055613458f * a - 0.0638541728f * b;
        const float s_ = c.L - 0.0894841775f * a - 1.2914855480f * b;

        const float l = l_ * l_ * l_;
        const float m = m_ * m_ * m_;
        const float s = s_ * s_ * s_;

        const float rLin = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
        const float gLin = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
        const float bLin = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;

        return {
            std::clamp(rLin, 0.0f, 1.0f),
            std::clamp(gLin, 0.0f, 1.0f),
            std::clamp(bLin, 0.0f, 1.0f)
        };
    }

    static inline float linearToSrgbGamma(float x) noexcept
    {
        x = std::clamp(x, 0.0f, 1.0f);
        return (x <= 0.0031308f) ? (12.92f * x)
                                 : (1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f);
    }

    static inline juce::Colour fromOKLCH(float L, float C, float hDeg, float alpha = 1.0f) noexcept
    {
        const auto lin = oklchToLinearSRGB({ L, C, hDeg });
        return juce::Colour::fromFloatRGBA(
            linearToSrgbGamma(lin.r),
            linearToSrgbGamma(lin.g),
            linearToSrgbGamma(lin.b),
            std::clamp(alpha, 0.0f, 1.0f));
    }

    static inline juce::Colour getSlotColour(int slotIndex, float alpha = 1.0f) noexcept
    {
        const int idx = std::clamp(slotIndex, 0, LayoutMetrics::kSlotCount - 1);
        return juce::Colour(kSlotPalette[static_cast<size_t>(idx)].argbHex).withAlpha(alpha);
    }

    static inline juce::Colour getSlotColourWithLightness(int slotIndex, float customL, float alpha = 1.0f) noexcept
    {
        const int idx = std::clamp(slotIndex, 0, LayoutMetrics::kSlotCount - 1);
        const auto base = kSlotPalette[static_cast<size_t>(idx)].oklch;
        return fromOKLCH(customL, base.C, base.hDeg, alpha);
    }

    // Returns normalized sRGB float triplets for OpenGL uniform u_SlotColors[8]
    static inline std::array<std::array<float, 3>, LayoutMetrics::kSlotCount> getShaderSlotColors() noexcept
    {
        std::array<std::array<float, 3>, LayoutMetrics::kSlotCount> out {};
        for (size_t i = 0; i < LayoutMetrics::kSlotCount; ++i)
        {
            const uint32_t hex = kSlotPalette[i].argbHex;
            out[i][0] = static_cast<float>((hex >> 16) & 0xFFu) / 255.0f;
            out[i][1] = static_cast<float>((hex >> 8)  & 0xFFu) / 255.0f;
            out[i][2] = static_cast<float>( hex        & 0xFFu) / 255.0f;
        }
        return out;
    }

    static inline juce::Colour obsidianDeep()    noexcept { return juce::Colour(ObsidianPalette::kBgDeepObsidian); }
    static inline juce::Colour panelSlate()      noexcept { return juce::Colour(ObsidianPalette::kBgPanelSlate); }
    static inline juce::Colour elevatedCard()    noexcept { return juce::Colour(ObsidianPalette::kBgElevatedCard); }
    static inline juce::Colour borderSubtle()    noexcept { return juce::Colour(ObsidianPalette::kBorderSubtle); }
    static inline juce::Colour borderHighlight() noexcept { return juce::Colour(ObsidianPalette::kBorderHighlight); }
    static inline juce::Colour textPrimary()     noexcept { return juce::Colour(ObsidianPalette::kTextPrimary); }
    static inline juce::Colour textSecondary()   noexcept { return juce::Colour(ObsidianPalette::kTextSecondary); }
    static inline juce::Colour textMuted()       noexcept { return juce::Colour(ObsidianPalette::kTextMuted); }
    static inline juce::Colour accentLaser()     noexcept { return juce::Colour(ObsidianPalette::kAccentLaser); }
    static inline juce::Colour accentMutate()    noexcept { return juce::Colour(ObsidianPalette::kAccentMutate); }
    static inline juce::Colour accentWarning()   noexcept { return juce::Colour(ObsidianPalette::kAccentWarning); }
    static inline juce::Colour accentLockGold()  noexcept { return juce::Colour(ObsidianPalette::kAccentLockGold); }
};

// =============================================================================
// 3. CUSTOM JUCE 8 LOOK-AND-FEEL FOR "OBSIDIAN SPECTRAL LAB"
// =============================================================================

class ObsidianSpectralLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    ObsidianSpectralLookAndFeel()
    {
        setColour(juce::ResizableWindow::backgroundColourId, ThemeOKLCH::obsidianDeep());
        setColour(juce::Slider::rotarySliderFillColourId,    ThemeOKLCH::accentLaser());
        setColour(juce::Slider::thumbColourId,               ThemeOKLCH::textPrimary());
        setColour(juce::TextButton::buttonColourId,          ThemeOKLCH::elevatedCard());
        setColour(juce::TextButton::buttonOnColourId,        ThemeOKLCH::accentLaser().withAlpha(0.25f));
        setColour(juce::TextButton::textColourOffId,         ThemeOKLCH::textSecondary());
        setColour(juce::TextButton::textColourOnId,          ThemeOKLCH::textPrimary());
        setColour(juce::ComboBox::backgroundColourId,        ThemeOKLCH::elevatedCard());
        setColour(juce::ComboBox::outlineColourId,           ThemeOKLCH::borderSubtle());
        setColour(juce::ComboBox::textColourId,              ThemeOKLCH::textPrimary());
        setColour(juce::PopupMenu::backgroundColourId,       ThemeOKLCH::panelSlate());
        setColour(juce::PopupMenu::textColourId,             ThemeOKLCH::textPrimary());
        setColour(juce::PopupMenu::highlightedBackgroundColourId, ThemeOKLCH::accentLaser().withAlpha(0.22f));
    }

    void drawRotarySlider(juce::Graphics& g,
                          int x, int y, int width, int height,
                          float sliderPosProportional,
                          float rotaryStartAngle,
                          float rotaryEndAngle,
                          juce::Slider& slider) override
    {
        const auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(4.0f);
        const float radius = std::min(bounds.getWidth(), bounds.getHeight()) * 0.5f;
        const auto centre = bounds.getCentre();
        const float rx = centre.x - radius;
        const float ry = centre.y - radius;
        const float rw = radius * 2.0f;
        const float angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

        const juce::Colour accent = slider.findColour(juce::Slider::rotarySliderFillColourId);

        // Inner obsidian dial body
        const float bodyRadius = radius * 0.74f;
        juce::ColourGradient bodyGrad(
            ThemeOKLCH::elevatedCard().brighter(0.08f), centre.x, centre.y - bodyRadius,
            ThemeOKLCH::obsidianDeep(), centre.x, centre.y + bodyRadius, false);
        g.setGradientFill(bodyGrad);
        g.fillEllipse(centre.x - bodyRadius, centre.y - bodyRadius, bodyRadius * 2.0f, bodyRadius * 2.0f);

        g.setColour(ThemeOKLCH::borderHighlight());
        g.drawEllipse(centre.x - bodyRadius, centre.y - bodyRadius, bodyRadius * 2.0f, bodyRadius * 2.0f, 1.2f);

        // Background track arc
        juce::Path trackArc;
        trackArc.addCentredArc(centre.x, centre.y, radius * 0.88f, radius * 0.88f,
                               0.0f, rotaryStartAngle, rotaryEndAngle, true);
        g.setColour(ThemeOKLCH::borderSubtle());
        g.strokePath(trackArc, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Active value arc
        juce::Path valueArc;
        valueArc.addCentredArc(centre.x, centre.y, radius * 0.88f, radius * 0.88f,
                               0.0f, rotaryStartAngle, angle, true);
        g.setColour(accent);
        g.strokePath(valueArc, juce::PathStrokeType(3.2f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Subtle outer glow on value arc
        g.setColour(accent.withAlpha(0.22f));
        g.strokePath(valueArc, juce::PathStrokeType(6.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // Pointer tick line
        juce::Path pointer;
        const float pointerLen = bodyRadius * 0.78f;
        pointer.addRoundedRectangle(-1.3f, -bodyRadius + 2.5f, 2.6f, pointerLen * 0.55f, 1.2f);
        g.setColour(ThemeOKLCH::textPrimary());
        g.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(centre.x, centre.y));

        juce::ignoreUnused(rx, ry, rw);
    }

    void drawButtonBackground(juce::Graphics& g,
                              juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override
    {
        const auto bounds = button.getLocalBounds().toFloat().reduced(0.75f);
        const bool toggled = button.getToggleState();
        const juce::Colour baseCol = toggled
            ? button.findColour(juce::TextButton::buttonOnColourId)
            : backgroundColour;

        juce::Colour fillCol = baseCol;
        if (shouldDrawButtonAsDown)
            fillCol = fillCol.brighter(0.18f);
        else if (shouldDrawButtonAsHighlighted)
            fillCol = fillCol.brighter(0.08f);

        g.setColour(fillCol);
        g.fillRoundedRectangle(bounds, LayoutMetrics::kCornerRadius);

        const juce::Colour borderCol = toggled
            ? ThemeOKLCH::accentLaser()
            : (shouldDrawButtonAsHighlighted ? ThemeOKLCH::borderHighlight() : ThemeOKLCH::borderSubtle());
        g.setColour(borderCol);
        g.drawRoundedRectangle(bounds, LayoutMetrics::kCornerRadius, toggled ? 1.5f : 1.0f);
    }
};

} // namespace spectile::ui
