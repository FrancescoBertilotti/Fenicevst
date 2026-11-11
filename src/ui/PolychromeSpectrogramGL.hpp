#pragma once

#include "ThemeOKLCH.hpp"
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_opengl/juce_opengl.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>

namespace spectile::ui
{

// =============================================================================
// 1. IEEE-754 FLOAT32 <-> FLOAT16 CONVERSION & 6.65 KB/HOP GPU PACKET
// =============================================================================

inline uint16_t float32ToFloat16(float value) noexcept
{
    uint32_t fbits = 0;
    std::memcpy(&fbits, &value, sizeof(float));
    const uint32_t sign = (fbits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((fbits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = fbits & 0x007FFFFFu;

    if (exp <= 0)
    {
        if (exp < -10)
            return static_cast<uint16_t>(sign);
        mant = (mant | 0x00800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float float16ToFloat32(uint16_t h) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;

    uint32_t fbits = 0;
    if (exp == 0)
    {
        if (mant == 0)
            fbits = sign;
        else
        {
            uint32_t m = mant;
            uint32_t e = 127 - 14;
            while ((m & 0x0400u) == 0)
            {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            fbits = sign | (e << 23) | (m << 13);
        }
    }
    else if (exp == 31)
    {
        fbits = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        fbits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(float));
    return out;
}

// 512 logarithmic bands * 13 bytes/band = 6,656 bytes (~6.65 KB/hop)
struct alignas(64) SpectralColumnPacket
{
    static constexpr size_t kNumLogBands = 512;

    // Interleaved [MagNorm_fp16, TransientFlux_fp16] for GL_RG16F (4 bytes/band = 2048 B)
    std::array<uint16_t, kNumLogBands * 2> magTransRg16f {};

    // Normalized mask weights M_1..M_4 in [0, 255] for GL_RGBA8 (4 bytes/band = 2048 B)
    std::array<uint8_t, kNumLogBands * 4> maskWeights1to4 {};

    // Normalized mask weights M_5..M_8 in [0, 255] for GL_RGBA8 (4 bytes/band = 2048 B)
    std::array<uint8_t, kNumLogBands * 4> maskWeights5to8 {};

    // Dominant sample index argmax_i M_i(t, b) in {0..7} for Voronoi edge detection (512 B)
    std::array<uint8_t, kNumLogBands> domSampleId {};

    // Real-time conquered bin ratio per slot [0, 1] for side-dock telemetry
    std::array<float, LayoutMetrics::kSlotCount> slotConqueredRatio {};
};

// =============================================================================
// 2. WAIT-FREE SINGLE-PRODUCER SINGLE-CONSUMER (SPSC) RING BUFFER
// =============================================================================

template <typename T, size_t Capacity = 128>
class LockFreeSpscRingBuffer
{
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    LockFreeSpscRingBuffer() : storage_(Capacity) {}

    bool tryPush(const T& item) noexcept
    {
        const size_t w = writeIndex_.load(std::memory_order_relaxed);
        const size_t nextW = (w + 1) & kMask;
        if (nextW == readIndex_.load(std::memory_order_acquire))
            return false; // Ring buffer full

        storage_[w] = item;
        writeIndex_.store(nextW, std::memory_order_release);
        return true;
    }

    bool tryPop(T& outItem) noexcept
    {
        const size_t r = readIndex_.load(std::memory_order_relaxed);
        if (r == writeIndex_.load(std::memory_order_acquire))
            return false; // Ring buffer empty

        outItem = storage_[r];
        readIndex_.store((r + 1) & kMask, std::memory_order_release);
        return true;
    }

    size_t sizeApprox() const noexcept
    {
        const size_t w = writeIndex_.load(std::memory_order_acquire);
        const size_t r = readIndex_.load(std::memory_order_acquire);
        return (w - r) & kMask;
    }

private:
    static constexpr size_t kMask = Capacity - 1;
    alignas(64) std::atomic<size_t> writeIndex_ { 0 };
    alignas(64) std::atomic<size_t> readIndex_  { 0 };
    std::vector<T> storage_;
};

// =============================================================================
// 3. HARDWARE-ACCELERATED 60/120 FPS POLYCHROMATIC SPECTROGRAM (GLSL 410 CORE)
// =============================================================================

struct BandLockDividerState
{
    bool enabled { false };
    float freqHz { 150.0f };
    int targetSlot { 0 }; // 0..7 -> S1..S8
};

class PolychromeSpectrogramGL final : public juce::Component,
                                      private juce::OpenGLRenderer,
                                      private juce::Timer
{
public:
    static constexpr int kHistoryColumns = 512;
    static constexpr int kFreqBands      = static_cast<int>(SpectralColumnPacket::kNumLogBands);
    static constexpr float kMinFreqHz    = 20.0f;
    static constexpr float kMaxFreqHz    = 20000.0f;

    using BandLockChangedCallback = std::function<void(int lockIndex, float newFreqHz, int targetSlot)>;

    explicit PolychromeSpectrogramGL(LockFreeSpscRingBuffer<SpectralColumnPacket, 128>& spscQueue)
        : spscQueue_(spscQueue)
    {
        setOpaque(true);

        // Default musical Band Lock configuration
        bandLocks_[0] = { true,   145.0f, 0 }; // Sub-Lock < 145 Hz -> S1 (Cyan Ray)
        bandLocks_[1] = { false,  650.0f, 1 }; // Low-Mid Lock -> S2
        bandLocks_[2] = { false, 2800.0f, 3 }; // Formant Presence -> S4
        bandLocks_[3] = { false, 8500.0f, 6 }; // Air / Brilliance -> S7

        cpuFallbackImage_ = juce::Image(juce::Image::ARGB, kHistoryColumns, kFreqBands, true);

        glContext_.setOpenGLVersionRequired(juce::OpenGLContext::openGL4_1);
        glContext_.setRenderer(this);
        glContext_.setContinuousRepainting(true);
        glContext_.setComponentPaintingEnabled(true);
        glContext_.attachTo(*this);

        startTimerHz(60);
    }

    ~PolychromeSpectrogramGL() override
    {
        stopTimer();
        glContext_.detach();
    }

    void setBandLockState(int index, const BandLockDividerState& state)
    {
        if (index >= 0 && index < LayoutMetrics::kBandLockCount)
        {
            bandLocks_[static_cast<size_t>(index)] = state;
            repaint();
        }
    }

    const BandLockDividerState& getBandLockState(int index) const noexcept
    {
        return bandLocks_[static_cast<size_t>(std::clamp(index, 0, LayoutMetrics::kBandLockCount - 1))];
    }

    void setOnBandLockChanged(BandLockChangedCallback cb) { onBandLockChanged_ = std::move(cb); }

    void setCompactBannerMode(bool compact) noexcept
    {
        compactBannerMode_ = compact;
        repaint();
    }

    void setVoronoiBorderIntensity(float intensity) noexcept
    {
        voronoiEdgeIntensity_.store(std::clamp(intensity, 0.0f, 1.0f), std::memory_order_relaxed);
    }

    std::array<float, LayoutMetrics::kSlotCount> getLatestConqueredRatios() const noexcept
    {
        return latestConqueredRatios_;
    }

    float getEstimatedGpuFps() const noexcept
    {
        return measuredFps_.load(std::memory_order_relaxed);
    }

    // Utility: Frequency (Hz) <-> Normalized Log Coordinate [0, 1]
    static float freqHzToNormLog(float freqHz) noexcept
    {
        const float clamped = std::clamp(freqHz, kMinFreqHz, kMaxFreqHz);
        return std::log(clamped / kMinFreqHz) / std::log(kMaxFreqHz / kMinFreqHz);
    }

    static float normLogToFreqHz(float normY) noexcept
    {
        const float clamped = std::clamp(normY, 0.0f, 1.0f);
        return kMinFreqHz * std::pow(kMaxFreqHz / kMinFreqHz, clamped);
    }

    // =========================================================================
    // OPENGL 4.1 CORE RENDERING PIPELINE (glTexSubImage2D + GLSL 410 SHADER)
    // =========================================================================

    void newOpenGLContextCreated() override
    {
        using namespace juce::gl;

        compileSpectrogramShader();
        createQuadGeometry();

        glGenTextures(4, textures_.data());

        // 1. u_MagTransTex: GL_RG16F (Magnitude dB normalized + Transient Flux)
        glBindTexture(GL_TEXTURE_2D, textures_[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, kHistoryColumns, kFreqBands, 0, GL_RG, GL_HALF_FLOAT, nullptr);

        // 2. u_Mask14Tex: GL_RGBA8 (Weights M_1 .. M_4)
        glBindTexture(GL_TEXTURE_2D, textures_[1]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kHistoryColumns, kFreqBands, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // 3. u_Mask58Tex: GL_RGBA8 (Weights M_5 .. M_8)
        glBindTexture(GL_TEXTURE_2D, textures_[2]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kHistoryColumns, kFreqBands, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        // 4. u_DomSampleTex: GL_R8 (Dominant sample ID /255.0 with GL_NEAREST for crisp 4-neighbor Sobel)
        glBindTexture(GL_TEXTURE_2D, textures_[3]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, kHistoryColumns, kFreqBands, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

        glBindTexture(GL_TEXTURE_2D, 0);
        glInitialized_.store(true, std::memory_order_release);
    }

    void renderOpenGL() override
    {
        using namespace juce::gl;

        // Measure GPU frame rate
        const double nowMs = juce::Time::getMillisecondCounterHiRes();
        if (lastFrameTimeMs_ > 0.0)
        {
            const double dt = std::max(1.0, nowMs - lastFrameTimeMs_);
            const float instFps = static_cast<float>(1000.0 / dt);
            measuredFps_.store(0.92f * measuredFps_.load(std::memory_order_relaxed) + 0.08f * instFps,
                               std::memory_order_relaxed);
        }
        lastFrameTimeMs_ = nowMs;

        // Drain incoming 6.65 KB/hop packets from the lock-free SPSC ring buffer
        SpectralColumnPacket packet;
        int uploadedColumns = 0;
        while (spscQueue_.tryPop(packet))
        {
            uploadSingleColumnToGPU(packet);
            latestConqueredRatios_ = packet.slotConqueredRatio;
            ++uploadedColumns;
        }

        // If no real-time audio packets arrived recently, synthesize a procedural preview column
        if (uploadedColumns == 0 && (nowMs - lastPacketTimeMs_) > 22.0)
        {
            generateProceduralPreviewColumn(packet, static_cast<float>(nowMs * 0.001));
            uploadSingleColumnToGPU(packet);
            latestConqueredRatios_ = packet.slotConqueredRatio;
        }
        else if (uploadedColumns > 0)
        {
            lastPacketTimeMs_ = nowMs;
        }

        const float desktopScale = static_cast<float>(glContext_.getRenderingScale());
        const int vpWidth  = juce::roundToInt(static_cast<float>(getWidth())  * desktopScale);
        const int vpHeight = juce::roundToInt(static_cast<float>(getHeight()) * desktopScale);

        glViewport(0, 0, vpWidth, vpHeight);
        glClearColor(0.039f, 0.043f, 0.055f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (shaderProgram_ != nullptr && vao_ != 0)
        {
            shaderProgram_->use();

            for (int i = 0; i < 4; ++i)
            {
                glActiveTexture(static_cast<GLenum>(GL_TEXTURE0 + i));
                glBindTexture(GL_TEXTURE_2D, textures_[static_cast<size_t>(i)]);
            }

            if (locMagTransTex_ >= 0)  glUniform1i(locMagTransTex_,  0);
            if (locMask14Tex_ >= 0)    glUniform1i(locMask14Tex_,    1);
            if (locMask58Tex_ >= 0)    glUniform1i(locMask58Tex_,    2);
            if (locDomSampleTex_ >= 0) glUniform1i(locDomSampleTex_, 3);

            const float writeHeadNorm = static_cast<float>(writeColumnIndex_) / static_cast<float>(kHistoryColumns);
            if (locWriteHeadNorm_ >= 0)
                glUniform1f(locWriteHeadNorm_, writeHeadNorm);

            if (locTexelSize_ >= 0)
                glUniform2f(locTexelSize_, 1.0f / static_cast<float>(kHistoryColumns), 1.0f / static_cast<float>(kFreqBands));

            if (locVoronoiEdge_ >= 0)
                glUniform1f(locVoronoiEdge_, voronoiEdgeIntensity_.load(std::memory_order_relaxed));

            if (locSlotColors_ >= 0)
            {
                const auto colors = ThemeOKLCH::getShaderSlotColors();
                glUniform3fv(locSlotColors_, LayoutMetrics::kSlotCount, colors[0].data());
            }

            if (locBandLockY_ >= 0)
            {
                std::array<float, 4> lockNorms {};
                for (size_t b = 0; b < 4; ++b)
                    lockNorms[b] = bandLocks_[b].enabled ? freqHzToNormLog(bandLocks_[b].freqHz) : -1.0f;
                glUniform4fv(locBandLockY_, 1, lockNorms.data());
            }

            glBindVertexArray(vao_);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindVertexArray(0);
        }
    }

    void openGLContextClosing() override
    {
        using namespace juce::gl;
        glInitialized_.store(false, std::memory_order_release);
        shaderProgram_.reset();

        if (vbo_ != 0) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
        if (vao_ != 0) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
        if (textures_[0] != 0)
        {
            glDeleteTextures(4, textures_.data());
            textures_.fill(0);
        }
    }

    // =========================================================================
    // 2D OVERLAY (FREQUENCY AXIS LABELS, BAND LOCK HANDLES & FALLBACK)
    // =========================================================================

    void paint(juce::Graphics& g) override
    {
        const auto bounds = getLocalBounds().toFloat();

        if (!glInitialized_.load(std::memory_order_acquire))
        {
            g.drawImage(cpuFallbackImage_, bounds, juce::RectanglePlacement::stretchToFit);
        }

        // Subtle logarithmic frequency gridlines (50 Hz, 150 Hz, 500 Hz, 1.5 kHz, 5 kHz, 12 kHz)
        static constexpr std::array<float, 6> kGridFreqs { 50.0f, 150.0f, 500.0f, 1500.0f, 5000.0f, 12000.0f };
        g.setFont(juce::FontOptions(10.5f));

        for (float fHz : kGridFreqs)
        {
            const float normY = freqHzToNormLog(fHz);
            const float y = bounds.getBottom() - normY * bounds.getHeight();

            g.setColour(juce::Colour(ObsidianPalette::kGridHairline));
            g.drawHorizontalLine(juce::roundToInt(y), bounds.getX(), bounds.getRight());

            if (!compactBannerMode_)
            {
                juce::String label = (fHz >= 1000.0f)
                    ? juce::String(fHz / 1000.0f, 1) + " kHz"
                    : juce::String(juce::roundToInt(fHz)) + " Hz";
                g.setColour(ThemeOKLCH::textMuted());
                g.drawText(label, static_cast<int>(bounds.getRight()) - 58, static_cast<int>(y) - 13, 52, 12,
                           juce::Justification::centredRight, false);
            }
        }

        // Draw interactive Band Lock tags & handles
        for (int i = 0; i < LayoutMetrics::kBandLockCount; ++i)
        {
            const auto& lock = bandLocks_[static_cast<size_t>(i)];
            if (!lock.enabled)
                continue;

            const float normY = freqHzToNormLog(lock.freqHz);
            const float y = bounds.getBottom() - normY * bounds.getHeight();
            const juce::Colour slotCol = ThemeOKLCH::getSlotColour(lock.targetSlot);
            const bool isHoveredOrDragged = (hoveredLockIndex_ == i || draggedLockIndex_ == i);

            g.setColour(slotCol.withAlpha(isHoveredOrDragged ? 0.95f : 0.72f));
            g.drawLine(bounds.getX(), y, bounds.getRight(), y, isHoveredOrDragged ? 2.2f : 1.4f);

            const juce::String tagText = "LOCK " + juce::String(i + 1) + " ["
                + juce::String(std::string(ThemeOKLCH::kSlotPalette[static_cast<size_t>(lock.targetSlot)].code))
                + "] " + juce::String(juce::roundToInt(lock.freqHz)) + " Hz";

            const juce::Rectangle<float> pill(bounds.getX() + 10.0f, y - 10.0f, 134.0f, 19.0f);
            g.setColour(ThemeOKLCH::obsidianDeep().withAlpha(0.88f));
            g.fillRoundedRectangle(pill, 4.0f);
            g.setColour(slotCol);
            g.drawRoundedRectangle(pill, 4.0f, isHoveredOrDragged ? 1.6f : 1.1f);
            g.setFont(juce::FontOptions(11.0f, juce::Font::bold));
            g.drawText(tagText, pill.toNearestInt(), juce::Justification::centred, false);
        }

        // Border frame
        g.setColour(ThemeOKLCH::borderSubtle());
        g.drawRoundedRectangle(bounds.reduced(0.5f), LayoutMetrics::kCornerRadius, 1.0f);
    }

    // =========================================================================
    // INTERACTIVE BAND-LOCK DIVIDER DRAGGING
    // =========================================================================

    void mouseMove(const juce::MouseEvent& e) override
    {
        const int hit = findBandLockNearY(static_cast<float>(e.y));
        if (hit != hoveredLockIndex_)
        {
            hoveredLockIndex_ = hit;
            setMouseCursor(hit >= 0 ? juce::MouseCursor::UpDownResizeCursor : juce::MouseCursor::NormalCursor);
            repaint();
        }
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        draggedLockIndex_ = findBandLockNearY(static_cast<float>(e.y));
        if (draggedLockIndex_ >= 0 && e.mods.isRightButtonDown())
        {
            // Right-click cycles the locked sample slot S1..S8
            auto& lock = bandLocks_[static_cast<size_t>(draggedLockIndex_)];
            lock.targetSlot = (lock.targetSlot + 1) % LayoutMetrics::kSlotCount;
            if (onBandLockChanged_)
                onBandLockChanged_(draggedLockIndex_, lock.freqHz, lock.targetSlot);
            repaint();
        }
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (draggedLockIndex_ >= 0)
        {
            const float h = std::max(1.0f, static_cast<float>(getHeight()));
            const float normY = std::clamp(1.0f - static_cast<float>(e.y) / h, 0.02f, 0.98f);
            auto& lock = bandLocks_[static_cast<size_t>(draggedLockIndex_)];
            lock.freqHz = normLogToFreqHz(normY);

            if (onBandLockChanged_)
                onBandLockChanged_(draggedLockIndex_, lock.freqHz, lock.targetSlot);
            repaint();
        }
    }

    void mouseUp(const juce::MouseEvent&) override
    {
        draggedLockIndex_ = -1;
    }

private:
    void timerCallback() override
    {
        if (!glInitialized_.load(std::memory_order_acquire))
        {
            SpectralColumnPacket packet;
            int count = 0;
            while (spscQueue_.tryPop(packet))
            {
                updateCpuFallbackColumn(packet);
                latestConqueredRatios_ = packet.slotConqueredRatio;
                ++count;
            }
            if (count == 0)
            {
                const float tSec = static_cast<float>(juce::Time::getMillisecondCounterHiRes() * 0.001);
                generateProceduralPreviewColumn(packet, tSec);
                updateCpuFallbackColumn(packet);
                latestConqueredRatios_ = packet.slotConqueredRatio;
            }
            repaint();
        }
    }

    int findBandLockNearY(float mouseY) const noexcept
    {
        const float h = std::max(1.0f, static_cast<float>(getHeight()));
        for (int i = 0; i < LayoutMetrics::kBandLockCount; ++i)
        {
            const auto& lock = bandLocks_[static_cast<size_t>(i)];
            if (!lock.enabled)
                continue;
            const float lockY = h - freqHzToNormLog(lock.freqHz) * h;
            if (std::abs(mouseY - lockY) <= 8.0f)
                return i;
        }
        return -1;
    }

    void uploadSingleColumnToGPU(const SpectralColumnPacket& pkt)
    {
        using namespace juce::gl;
        const int col = writeColumnIndex_;

        glBindTexture(GL_TEXTURE_2D, textures_[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, col, 0, 1, kFreqBands, GL_RG, GL_HALF_FLOAT, pkt.magTransRg16f.data());

        glBindTexture(GL_TEXTURE_2D, textures_[1]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, col, 0, 1, kFreqBands, GL_RGBA, GL_UNSIGNED_BYTE, pkt.maskWeights1to4.data());

        glBindTexture(GL_TEXTURE_2D, textures_[2]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, col, 0, 1, kFreqBands, GL_RGBA, GL_UNSIGNED_BYTE, pkt.maskWeights5to8.data());

        glBindTexture(GL_TEXTURE_2D, textures_[3]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, col, 0, 1, kFreqBands, GL_RED, GL_UNSIGNED_BYTE, pkt.domSampleId.data());

        glBindTexture(GL_TEXTURE_2D, 0);
        writeColumnIndex_ = (writeColumnIndex_ + 1) % kHistoryColumns;
    }

    void updateCpuFallbackColumn(const SpectralColumnPacket& pkt)
    {
        const int col = writeColumnIndex_;
        for (int b = 0; b < kFreqBands; ++b)
        {
            const float magNorm = float16ToFloat32(pkt.magTransRg16f[static_cast<size_t>(b * 2)]);
            const int dom = std::clamp(static_cast<int>(pkt.domSampleId[static_cast<size_t>(b)]), 0, 7);
            const juce::Colour baseCol = ThemeOKLCH::getSlotColour(dom);
            const float lum = std::pow(std::clamp(magNorm, 0.0f, 1.0f), 1.35f);
            cpuFallbackImage_.setPixelAt(col, kFreqBands - 1 - b, baseCol.withMultipliedBrightness(lum));
        }
        writeColumnIndex_ = (writeColumnIndex_ + 1) % kHistoryColumns;
    }

    static void generateProceduralPreviewColumn(SpectralColumnPacket& pkt, float timeSec) noexcept
    {
        std::array<float, LayoutMetrics::kSlotCount> slotCounts {};
        for (size_t b = 0; b < SpectralColumnPacket::kNumLogBands; ++b)
        {
            const float y = static_cast<float>(b) / static_cast<float>(SpectralColumnPacket::kNumLogBands - 1);

            // Harmonic ridge + Voronoi tile drift simulation
            const float harmonicComb = 0.5f + 0.5f * std::sin(y * 42.0f + timeSec * 2.4f);
            const float formantEnv   = std::exp(-3.5f * (y - 0.35f) * (y - 0.35f))
                                     + 0.7f * std::exp(-8.0f * (y - 0.68f) * (y - 0.68f));
            const float magNorm = std::clamp(formantEnv * (0.35f + 0.65f * harmonicComb), 0.02f, 0.98f);
            const float trans   = (std::sin(timeSec * 8.0f + y * 6.0f) > 0.92f) ? 0.85f : 0.05f;

            pkt.magTransRg16f[b * 2]     = float32ToFloat16(magNorm);
            pkt.magTransRg16f[b * 2 + 1] = float32ToFloat16(trans);

            // Voronoi region assignment across the 8 samples
            const float voronoiWave = 3.5f + 3.49f * std::sin(y * 9.5f + timeSec * 1.15f + std::cos(timeSec * 0.7f - y * 5.0f));
            const int domIdx = std::clamp(static_cast<int>( std::floor(voronoiWave) ), 0, 7);
            const int secIdx = (domIdx + 1) & 7;
            const float frac = voronoiWave - std::floor(voronoiWave);
            const float feather = std::clamp((frac - 0.5f) * 3.0f + 0.5f, 0.0f, 1.0f);

            std::array<uint8_t, 8> weights {};
            weights[static_cast<size_t>(domIdx)] = static_cast<uint8_t>(std::round((1.0f - feather) * 255.0f));
            weights[static_cast<size_t>(secIdx)] = static_cast<uint8_t>(std::round(feather * 255.0f));

            for (size_t i = 0; i < 4; ++i)
            {
                pkt.maskWeights1to4[b * 4 + i] = weights[i];
                pkt.maskWeights5to8[b * 4 + i] = weights[i + 4];
            }

            const uint8_t winner = (feather > 0.5f) ? static_cast<uint8_t>(secIdx) : static_cast<uint8_t>(domIdx);
            pkt.domSampleId[b] = winner;
            slotCounts[winner] += 1.0f;
        }

        for (size_t i = 0; i < LayoutMetrics::kSlotCount; ++i)
            pkt.slotConqueredRatio[i] = slotCounts[i] / static_cast<float>(SpectralColumnPacket::kNumLogBands);
    }

    void createQuadGeometry()
    {
        using namespace juce::gl;
        static constexpr float kQuadVertices[] = {
            // posX,  posY,   u,    v
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVertices), kQuadVertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<const void*>(0));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<const void*>(2 * sizeof(float)));

        glBindVertexArray(0);
    }

    void compileSpectrogramShader()
    {
        using namespace juce::gl;

        static constexpr const char* kVertexShaderGLSL410 = R"GLSL(
            #version 410 core
            layout(location = 0) in vec2 a_Position;
            layout(location = 1) in vec2 a_TexCoord;
            out vec2 v_UV;

            void main()
            {
                v_UV = a_TexCoord;
                gl_Position = vec4(a_Position, 0.0, 1.0);
            }
        )GLSL";

        static constexpr const char* kFragmentShaderGLSL410 = R"GLSL(
            #version 410 core
            in vec2 v_UV;
            out vec4 fragColor;

            uniform sampler2D u_MagTransTex;   // R: normalized Mag dB [0,1], G: Transient Flux [0,1]
            uniform sampler2D u_Mask14Tex;     // RGBA: M_1 .. M_4
            uniform sampler2D u_Mask58Tex;     // RGBA: M_5 .. M_8
            uniform sampler2D u_DomSampleTex;  // R: Dominant sample ID / 255.0

            uniform float u_WriteHeadNorm;
            uniform vec2  u_TexelSize;
            uniform float u_VoronoiEdgeIntensity;
            uniform vec3  u_SlotColors[8];
            uniform vec4  u_BandLockY;

            void main()
            {
                // 1. Unwrap circular ring-buffer coordinate along time axis X
                vec2 ringUV = vec2(fract(v_UV.x + u_WriteHeadNorm), v_UV.y);

                vec2 magTrans = texture(u_MagTransTex, ringUV).rg;
                vec4 m14      = texture(u_Mask14Tex,   ringUV);
                vec4 m58      = texture(u_Mask58Tex,   ringUV);

                // 2. Weighted polychromatic OKLCH-calibrated color blend: sum_{i=1}^8 M_i * Color_i
                vec3 polyColor = m14.r * u_SlotColors[0]
                               + m14.g * u_SlotColors[1]
                               + m14.b * u_SlotColors[2]
                               + m14.a * u_SlotColors[3]
                               + m58.r * u_SlotColors[4]
                               + m58.g * u_SlotColors[5]
                               + m58.b * u_SlotColors[6]
                               + m58.a * u_SlotColors[7];

                float weightSum = dot(m14, vec4(1.0)) + dot(m58, vec4(1.0));
                if (weightSum > 0.001)
                    polyColor /= weightSum;

                // 3. Psychoacoustic luminance curve with incandescent hot-core on peaks > -12 dB
                float magNorm   = clamp(magTrans.r, 0.0, 1.0);
                float transient = clamp(magTrans.g, 0.0, 1.0);
                float lumCurve  = pow(magNorm, 1.45);
                float hotCore   = smoothstep(0.78, 0.99, magNorm);

                vec3 obsidianBg = vec3(0.039, 0.043, 0.055);
                vec3 specRGB    = mix(obsidianBg, polyColor * (0.20 + 1.15 * lumCurve), lumCurve);
                specRGB         = mix(specRGB, vec3(1.0, 0.99, 0.96), hotCore * 0.72 + transient * 0.25);

                // 4. 4-Neighbor Sobel / Difference Voronoi Cell Border Detector on u_DomSampleTex
                float domC = round(texture(u_DomSampleTex, ringUV).r * 255.0);
                float domR = round(texture(u_DomSampleTex, ringUV + vec2(u_TexelSize.x, 0.0)).r * 255.0);
                float domL = round(texture(u_DomSampleTex, ringUV - vec2(u_TexelSize.x, 0.0)).r * 255.0);
                float domU = round(texture(u_DomSampleTex, ringUV + vec2(0.0, u_TexelSize.y)).r * 255.0);
                float domD = round(texture(u_DomSampleTex, ringUV - vec2(0.0, u_TexelSize.y)).r * 255.0);

                float edgeDiff = step(0.5, abs(domC - domR) + abs(domC - domL) + abs(domC - domU) + abs(domC - domD));
                float edgeMask = edgeDiff * smoothstep(0.10, 0.35, magNorm) * u_VoronoiEdgeIntensity;
                vec3  neonEdge = mix(polyColor, vec3(0.92, 0.98, 1.0), 0.55);
                specRGB = mix(specRGB, neonEdge, edgeMask * 0.68);

                // 5. Draggable Frequency Band Lock glowing horizontal dividers
                for (int b = 0; b < 4; ++b)
                {
                    float lockY = u_BandLockY[b];
                    if (lockY > 0.0)
                    {
                        float distY = abs(v_UV.y - lockY);
                        float glow  = exp(-distY * 220.0) * 0.45;
                        specRGB += vec3(0.0, 0.96, 0.83) * glow;
                    }
                }

                // 6. Laser Playhead sweep glow at the newest column (v_UV.x -> 1.0)
                float distToPlayhead = 1.0 - v_UV.x;
                float playheadLaser  = exp(-distToPlayhead * 95.0) * 0.55;
                specRGB += vec3(0.0, 0.96, 0.83) * playheadLaser;

                fragColor = vec4(clamp(specRGB, 0.0, 1.0), 1.0);
            }
        )GLSL";

        auto prog = std::make_unique<juce::OpenGLShaderProgram>(glContext_);
        if (prog->addVertexShader(kVertexShaderGLSL410)
            && prog->addFragmentShader(kFragmentShaderGLSL410)
            && prog->link())
        {
            shaderProgram_  = std::move(prog);
            const GLuint pid = shaderProgram_->getProgramID();
            locMagTransTex_  = glGetUniformLocation(pid, "u_MagTransTex");
            locMask14Tex_    = glGetUniformLocation(pid, "u_Mask14Tex");
            locMask58Tex_    = glGetUniformLocation(pid, "u_Mask58Tex");
            locDomSampleTex_ = glGetUniformLocation(pid, "u_DomSampleTex");
            locWriteHeadNorm_= glGetUniformLocation(pid, "u_WriteHeadNorm");
            locTexelSize_    = glGetUniformLocation(pid, "u_TexelSize");
            locVoronoiEdge_  = glGetUniformLocation(pid, "u_VoronoiEdgeIntensity");
            locSlotColors_   = glGetUniformLocation(pid, "u_SlotColors");
            locBandLockY_    = glGetUniformLocation(pid, "u_BandLockY");
        }
    }

    LockFreeSpscRingBuffer<SpectralColumnPacket, 128>& spscQueue_;
    juce::OpenGLContext glContext_;
    std::unique_ptr<juce::OpenGLShaderProgram> shaderProgram_;

    std::array<GLuint, 4> textures_ { 0, 0, 0, 0 };
    GLuint vao_ { 0 };
    GLuint vbo_ { 0 };

    GLint locMagTransTex_   { -1 };
    GLint locMask14Tex_     { -1 };
    GLint locMask58Tex_     { -1 };
    GLint locDomSampleTex_  { -1 };
    GLint locWriteHeadNorm_ { -1 };
    GLint locTexelSize_     { -1 };
    GLint locVoronoiEdge_   { -1 };
    GLint locSlotColors_    { -1 };
    GLint locBandLockY_     { -1 };

    int writeColumnIndex_ { 0 };
    std::atomic<bool> glInitialized_ { false };
    std::atomic<float> voronoiEdgeIntensity_ { 0.85f };
    std::atomic<float> measuredFps_ { 60.0f };
    double lastFrameTimeMs_  { 0.0 };
    double lastPacketTimeMs_ { 0.0 };

    std::array<BandLockDividerState, LayoutMetrics::kBandLockCount> bandLocks_ {};
    std::array<float, LayoutMetrics::kSlotCount> latestConqueredRatios_ {};
    int hoveredLockIndex_ { -1 };
    int draggedLockIndex_ { -1 };
    bool compactBannerMode_ { false };

    juce::Image cpuFallbackImage_;
    BandLockChangedCallback onBandLockChanged_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PolychromeSpectrogramGL)
};

} // namespace spectile::ui
