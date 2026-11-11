#pragma once
// ==============================================================================
// SpecTile — src/engine/ParameterLayout.hpp
// Settore 3: Complete APVTS Parameter Layout, Wait-Free Atomic Parameter Cache
//            & Resilient Preset Serialization with Triple Sample Fallback
//            (1. Relative Path, 2. SHA-256 Content Hash, 3. Embedded FLAC Base64)
// ==============================================================================

#include "StaggeredHopScheduler.hpp"
#include "SpecTileVoice.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#if __has_include(<juce_audio_processors/juce_audio_processors.h>)
    #include <juce_audio_processors/juce_audio_processors.h>
    #include <juce_audio_formats/juce_audio_formats.h>
    #include <juce_cryptography/juce_cryptography.h>
    #define SPECTILE_HAS_JUCE 1
#else
    #define SPECTILE_HAS_JUCE 0
#endif

namespace spectile::engine {

// ==============================================================================
// 1. Canonical Parameter IDs (Shared Across DSP, Rule Engine, APVTS & UI)
// ==============================================================================
namespace param_ids {
    // 3 Primary Macro Controls (Bottom Bar / Simple View)
    inline constexpr std::string_view kMacroTileSize    = "macro_tile_size";
    inline constexpr std::string_view kMacroOrderMutate = "macro_order_mutate";
    inline constexpr std::string_view kMacroPhaseGlue   = "macro_phase_glue";

    // Central Radial Hub & Genetic Evolution Branches
    inline constexpr std::string_view kArchetypeSelect  = "archetype_select";
    inline constexpr std::string_view kBranchSelect     = "branch_select";
    inline constexpr std::string_view kBranchMorph      = "branch_morph";
    inline constexpr std::string_view kBranchAngle      = "branch_angle";

    // Spectral Partitioning & Frequency Band Locks (Quadrant 1)
    inline constexpr std::string_view kSubLockHz        = "sub_lock_hz";
    inline constexpr std::string_view kSubLockSource    = "sub_lock_source";

    // Deep Rule Matrix & Markov / Sparsity Controls (Quadrant 2)
    inline constexpr std::string_view kSoftmaxTemp      = "softmax_temp";
    inline constexpr std::string_view kSparsityGate     = "sparsity_gate";
    inline constexpr std::string_view kMarkovStayTime   = "markov_stay_t";
    inline constexpr std::string_view kMarkovStayFreq   = "markov_stay_f";
    inline constexpr std::string_view kTransientReset   = "transient_reset";
    inline constexpr std::string_view kVoronoiWeight    = "voronoi_weight";
    inline constexpr std::string_view kPerlinWeight     = "perlin_weight";
    inline constexpr std::string_view kCombWeight       = "comb_weight";
    inline constexpr std::string_view kAntiMasking      = "anti_masking";

    // Source-Filter & Phase Surgery (Quadrant 3)
    inline constexpr std::string_view kCrossSynthGamma  = "cross_synth_gamma";
    inline constexpr std::string_view kCrossSynthPerm   = "cross_synth_perm";
    inline constexpr std::string_view kFormantShift     = "formant_shift";
    inline constexpr std::string_view kFormantKeytrack  = "formant_keytrack";
    inline constexpr std::string_view kPhaseMode        = "phase_mode";

    // Global Performance, Polyphony & Envelope Controls
    inline constexpr std::string_view kPlayMode         = "play_mode";
    inline constexpr std::string_view kPolyphonyMode    = "polyphony_mode";
    inline constexpr std::string_view kTimeStretch      = "time_stretch";
    inline constexpr std::string_view kMasterGainDb     = "master_gain_db";
    inline constexpr std::string_view kEnvAttackSec     = "env_attack_sec";
    inline constexpr std::string_view kEnvDecaySec      = "env_decay_sec";
    inline constexpr std::string_view kEnvSustain       = "env_sustain";
    inline constexpr std::string_view kEnvReleaseSec    = "env_release_sec";

    // Per-Slot Parameter ID Builders (Slots i = 1..8)
    [[nodiscard]] inline std::string slotWeight(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_weight";
    }
    [[nodiscard]] inline std::string slotGainDb(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_gain_db";
    }
    [[nodiscard]] inline std::string slotPan(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_pan";
    }
    [[nodiscard]] inline std::string slotPitchSemi(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_pitch_semi";
    }
    [[nodiscard]] inline std::string slotLoopStart(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_loop_start";
    }
    [[nodiscard]] inline std::string slotLoopEnd(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_loop_end";
    }
    [[nodiscard]] inline std::string slotEnableH(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_hpr_h";
    }
    [[nodiscard]] inline std::string slotEnableP(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_hpr_p";
    }
    [[nodiscard]] inline std::string slotEnableR(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_hpr_r";
    }
    [[nodiscard]] inline std::string slotLock(std::size_t idx0) {
        return "slot_" + std::to_string(idx0 + 1) + "_lock";
    }
} // namespace param_ids

// ==============================================================================
// 2. Triple-Fallback Sample Slot Metadata for Self-Contained Preset State
// ==============================================================================
struct SampleSlotPresetState {
    std::size_t slotIndex    = 0;
    std::string relativePath; // (1) Relative path from preset/project directory
    std::string absolutePath; // Fallback absolute path on local machine
    std::string sha256Hash;   // (2) Cryptographic SHA-256 digest for relocated files
    std::string flacBase64;   // (3) Optional lossless FLAC Base64 payload for 100% self-contained DAW state
};

// ==============================================================================
// 3. Wait-Free Atomic Parameter Pointer Cache for Real-Time Audio Thread
// ==============================================================================
struct AtomicParameterCache {
    std::atomic<float>* macroTileSize    = nullptr;
    std::atomic<float>* macroOrderMutate = nullptr;
    std::atomic<float>* macroPhaseGlue   = nullptr;

    std::atomic<float>* archetypeSelect  = nullptr;
    std::atomic<float>* branchSelect     = nullptr;
    std::atomic<float>* branchMorph      = nullptr;
    std::atomic<float>* branchAngle      = nullptr;

    std::atomic<float>* subLockHz        = nullptr;
    std::atomic<float>* subLockSource    = nullptr;

    std::atomic<float>* softmaxTemp      = nullptr;
    std::atomic<float>* sparsityGate     = nullptr;
    std::atomic<float>* markovStayTime   = nullptr;
    std::atomic<float>* markovStayFreq   = nullptr;
    std::atomic<float>* transientReset   = nullptr;
    std::atomic<float>* voronoiWeight    = nullptr;
    std::atomic<float>* perlinWeight     = nullptr;
    std::atomic<float>* combWeight       = nullptr;
    std::atomic<float>* antiMasking      = nullptr;

    std::atomic<float>* crossSynthGamma  = nullptr;
    std::atomic<float>* crossSynthPerm   = nullptr;
    std::atomic<float>* formantShift     = nullptr;
    std::atomic<float>* formantKeytrack  = nullptr;
    std::atomic<float>* phaseMode        = nullptr;

    std::atomic<float>* playMode         = nullptr;
    std::atomic<float>* polyphonyMode    = nullptr;
    std::atomic<float>* timeStretch      = nullptr;
    std::atomic<float>* masterGainDb     = nullptr;
    std::atomic<float>* envAttackSec     = nullptr;
    std::atomic<float>* envDecaySec      = nullptr;
    std::atomic<float>* envSustain       = nullptr;
    std::atomic<float>* envReleaseSec    = nullptr;

    struct SlotPointers {
        std::atomic<float>* weight    = nullptr;
        std::atomic<float>* gainDb    = nullptr;
        std::atomic<float>* pan       = nullptr;
        std::atomic<float>* pitchSemi = nullptr;
        std::atomic<float>* loopStart = nullptr;
        std::atomic<float>* loopEnd   = nullptr;
        std::atomic<float>* enableH   = nullptr;
        std::atomic<float>* enableP   = nullptr;
        std::atomic<float>* enableR   = nullptr;
        std::atomic<float>* lock      = nullptr;
    };
    std::array<SlotPointers, kMaxSources> slots{};

    [[nodiscard]] static float loadOrDefault(const std::atomic<float>* ptr, float fallback) noexcept {
        return (ptr != nullptr) ? ptr->load(std::memory_order_relaxed) : fallback;
    }

    // Populates HopSynthesisParams in O(1) without locking on the audio thread
    [[nodiscard]] HopSynthesisParams readHopSynthesisParams(double sampleRate) const noexcept {
        HopSynthesisParams hp{};
        hp.sampleRate      = sampleRate;
        hp.crossSynthGamma = loadOrDefault(crossSynthGamma, 0.85f);
        hp.phaseGlueBeta   = loadOrDefault(macroPhaseGlue, 0.85f);
        hp.subLockCutoffHz = loadOrDefault(subLockHz, 140.0f);
        hp.subLockSourceId = static_cast<int>(std::lrintf(loadOrDefault(subLockSource, 1.0f))) - 1;

        const int pModeIdx = std::clamp(static_cast<int>(std::lrintf(loadOrDefault(phaseMode, 2.0f))), 0, 3);
        hp.phaseMode       = static_cast<PhaseMode>(pModeIdx);

        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const auto& sp = slots[i];
            auto& dst      = hp.slots[i];
            dst.weight         = loadOrDefault(sp.weight, 1.0f);
            const float dB     = loadOrDefault(sp.gainDb, 0.0f);
            dst.gainLinear     = std::pow(10.0f, dB * 0.05f);
            dst.pan            = loadOrDefault(sp.pan, 0.0f);
            dst.pitchSemitones = loadOrDefault(sp.pitchSemi, 0.0f);
            dst.loopStartNorm  = loadOrDefault(sp.loopStart, 0.0f);
            dst.loopEndNorm    = std::max(dst.loopStartNorm + 0.01f, loadOrDefault(sp.loopEnd, 1.0f));
            dst.enableH        = loadOrDefault(sp.enableH, 1.0f) >= 0.5f;
            dst.enableP        = loadOrDefault(sp.enableP, 1.0f) >= 0.5f;
            dst.enableR        = loadOrDefault(sp.enableR, 1.0f) >= 0.5f;
            dst.slotLocked     = loadOrDefault(sp.lock, 0.0f) >= 0.5f;
        }
        return hp;
    }
};

#if SPECTILE_HAS_JUCE

// ==============================================================================
// 4. JUCE APVTS ParameterLayout Builder & State Serializer
// ==============================================================================
struct SpecTileParameterLayout {
    [[nodiscard]] static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout() {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
        params.reserve(96);

        auto normRange = [](float minV, float maxV, float step = 0.001f, float skew = 1.0f) {
            return juce::NormalisableRange<float>(minV, maxV, step, skew);
        };

        // --- 3 Primary Macro Controls ---
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kMacroTileSize), 1},
            "Macro 1: Spectral Tile Size",
            normRange(0.0f, 1.0f),
            0.45f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kMacroOrderMutate), 1},
            "Macro 2: Rule Order vs Mutation",
            normRange(0.0f, 1.0f),
            0.35f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kMacroPhaseGlue), 1},
            "Macro 3: Phase & Spectral Glue",
            normRange(0.0f, 1.0f),
            0.85f));

        // --- Central Radial Hub & 6 Evolutionary Branches ---
        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{std::string(param_ids::kArchetypeSelect), 1},
            "Spectral Archetype",
            juce::StringArray{
                "A1: Harmonic Interleave",
                "A2: Transient-Formant Graft",
                "A3: Voronoi Spectral Shatter",
                "A4: Complementarity Hole-Filler",
                "A5: Spectral Freeze Morph",
                "A6: Formant Choir"},
            0));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{std::string(param_ids::kBranchSelect), 1},
            "Evolutionary Branch",
            juce::StringArray{
                "Branch 1: Sample Role Rotation",
                "Branch 2: Geometric Fragmentation",
                "Branch 3: Source-Filter Cross-Synth",
                "Branch 4: Transient/Sustain Contrast",
                "Branch 5: Perlin Liquefaction & Freeze",
                "Branch 6: Cross-Archetype Geodesic"},
            0));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kBranchMorph), 1},
            "Branch Radial Morph (rho)",
            normRange(0.0f, 1.0f),
            0.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kBranchAngle), 1},
            "Branch Polar Angle (theta)",
            normRange(0.0f, kTwoPi),
            0.0f));

        // --- Sub-Bass Frequency Band Lock ---
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kSubLockHz), 1},
            "Sub-Bass Band Lock Cutoff (Hz)",
            normRange(20.0f, 350.0f, 0.5f, 0.6f),
            140.0f));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{std::string(param_ids::kSubLockSource), 1},
            "Sub-Bass Locked Source",
            juce::StringArray{"Disabled", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"},
            1)); // Default locked to S1

        // --- Deep Rule Matrix Parameters ---
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kSoftmaxTemp), 1},
            "Rule Softmax Temperature",
            normRange(0.03f, 1.50f, 0.001f, 0.5f),
            0.22f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kSparsityGate), 1},
            "Spectral Sparsity Gate",
            normRange(0.0f, 1.0f),
            0.35f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kMarkovStayTime), 1},
            "Markov Temporal Persistence (P_stay,t)",
            normRange(0.0f, 0.99f),
            0.82f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kMarkovStayFreq), 1},
            "Markov Frequency Persistence (P_stay,f)",
            normRange(0.0f, 0.99f),
            0.75f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kTransientReset), 1},
            "Transient Markov Reset",
            normRange(0.0f, 1.0f),
            0.90f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kVoronoiWeight), 1},
            "Voronoi 2D Tiling Weight",
            normRange(0.0f, 1.0f),
            0.60f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kPerlinWeight), 1},
            "Simplex/Perlin 2D Field Weight",
            normRange(0.0f, 1.0f),
            0.40f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kCombWeight), 1},
            "Harmonic Comb Interleaving Weight",
            normRange(0.0f, 1.0f),
            0.30f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kAntiMasking), 1},
            "Anti-Masking Complementarity Hole-Filler",
            normRange(0.0f, 1.0f),
            0.55f));

        // --- Source-Filter & Phase Surgery ---
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kCrossSynthGamma), 1},
            "Source-Filter Cross-Synthesis (gamma)",
            normRange(0.0f, 1.0f),
            0.85f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kCrossSynthPerm), 1},
            "Envelope/Fine Permutation (chi_cross)",
            normRange(0.0f, 1.0f),
            0.50f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kFormantShift), 1},
            "Formant Shift Ratio",
            normRange(0.5f, 2.0f, 0.001f, 0.7f),
            1.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kFormantKeytrack), 1},
            "Formant Keytrack",
            normRange(0.0f, 1.0f),
            0.15f));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{std::string(param_ids::kPhaseMode), 1},
            "Phase Reconstruction Mode",
            juce::StringArray{
                "Raw Source Phase",
                "Identity Phase-Locked",
                "Causal MS-PGHI + Phase Glue",
                "Spectral Freeze"},
            2)); // Default Causal MS-PGHI

        // --- Global Play Mode, Polyphony & ADSR ---
        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{std::string(param_ids::kPlayMode), 1},
            "Play Mode",
            juce::StringArray{
                "Pitched Instrument (MIDI/MPE)",
                "Spectral Drone & Texture Generator (Latch)",
                "Rhythmic Loop Mosaicing (DAW Synced)",
                "Live Sidechain + Sample Pool Hybrid"},
            0));

        params.push_back(std::make_unique<juce::AudioParameterChoice>(
            juce::ParameterID{std::string(param_ids::kPolyphonyMode), 1},
            "Polyphony Architecture",
            juce::StringArray{
                "Mode A: Shared Spectral Bus (16 Voices)",
                "Mode B: True Per-Voice Chimeric ISTFT (6 MPE Voices)"},
            0));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kTimeStretch), 1},
            "Spectral Scan Speed / Time-Stretch",
            normRange(0.05f, 2.0f, 0.001f, 0.7f),
            1.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kMasterGainDb), 1},
            "Master Output Gain (dB)",
            normRange(-36.0f, 12.0f, 0.1f),
            0.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kEnvAttackSec), 1},
            "Amp Attack (s)",
            normRange(0.002f, 5.0f, 0.001f, 0.35f),
            0.015f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kEnvDecaySec), 1},
            "Amp Decay (s)",
            normRange(0.01f, 5.0f, 0.001f, 0.4f),
            0.25f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kEnvSustain), 1},
            "Amp Sustain",
            normRange(0.0f, 1.0f),
            0.85f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{std::string(param_ids::kEnvReleaseSec), 1},
            "Amp Release (s)",
            normRange(0.01f, 10.0f, 0.001f, 0.35f),
            0.45f));

        // --- Per-Slot Parameters (w_1..w_8, Gain, Pan, Pitch, Loop, H/P/R Filters, Lock) ---
        for (std::size_t i = 0; i < kMaxSources; ++i) {
            const std::string prefix = "S" + std::to_string(i + 1) + " ";

            params.push_back(std::make_unique<juce::AudioParameterFloat>(
                juce::ParameterID{param_ids::slotWeight(i), 1},
                prefix + "Presence Weight",
                normRange(0.0f, 1.0f),
                1.0f));

            params.push_back(std::make_unique<juce::AudioParameterFloat>(
                juce::ParameterID{param_ids::slotGainDb(i), 1},
                prefix + "Gain (dB)",
                normRange(-36.0f, 12.0f, 0.1f),
                0.0f));

            params.push_back(std::make_unique<juce::AudioParameterFloat>(
                juce::ParameterID{param_ids::slotPan(i), 1},
                prefix + "Stereo Pan",
                normRange(-1.0f, 1.0f),
                0.0f));

            params.push_back(std::make_unique<juce::AudioParameterFloat>(
                juce::ParameterID{param_ids::slotPitchSemi(i), 1},
                prefix + "Pitch Transpose (st)",
                normRange(-24.0f, 24.0f, 0.01f),
                0.0f));

            params.push_back(std::make_unique<juce::AudioParameterFloat>(
                juce::ParameterID{param_ids::slotLoopStart(i), 1},
                prefix + "Loop Start",
                normRange(0.0f, 0.99f),
                0.0f));

            params.push_back(std::make_unique<juce::AudioParameterFloat>(
                juce::ParameterID{param_ids::slotLoopEnd(i), 1},
                prefix + "Loop End",
                normRange(0.01f, 1.0f),
                1.0f));

            params.push_back(std::make_unique<juce::AudioParameterBool>(
                juce::ParameterID{param_ids::slotEnableH(i), 1},
                prefix + "Filter [H] Harmonics",
                true));

            params.push_back(std::make_unique<juce::AudioParameterBool>(
                juce::ParameterID{param_ids::slotEnableP(i), 1},
                prefix + "Filter [P] Percussive",
                true));

            params.push_back(std::make_unique<juce::AudioParameterBool>(
                juce::ParameterID{param_ids::slotEnableR(i), 1},
                prefix + "Filter [R] Residual",
                true));

            params.push_back(std::make_unique<juce::AudioParameterBool>(
                juce::ParameterID{param_ids::slotLock(i), 1},
                prefix + "Mutation Lock",
                false));
        }

        return {params.begin(), params.end()};
    }

    static void bindAtomicPointers(
        juce::AudioProcessorValueTreeState& apvts,
        AtomicParameterCache&               cache) noexcept
    {
        auto raw = [&apvts](std::string_view id) noexcept -> std::atomic<float>* {
            return apvts.getRawParameterValue(juce::StringRef(id.data()));
        };

        cache.macroTileSize    = raw(param_ids::kMacroTileSize);
        cache.macroOrderMutate = raw(param_ids::kMacroOrderMutate);
        cache.macroPhaseGlue   = raw(param_ids::kMacroPhaseGlue);

        cache.archetypeSelect  = raw(param_ids::kArchetypeSelect);
        cache.branchSelect     = raw(param_ids::kBranchSelect);
        cache.branchMorph      = raw(param_ids::kBranchMorph);
        cache.branchAngle      = raw(param_ids::kBranchAngle);

        cache.subLockHz        = raw(param_ids::kSubLockHz);
        cache.subLockSource    = raw(param_ids::kSubLockSource);

        cache.softmaxTemp      = raw(param_ids::kSoftmaxTemp);
        cache.sparsityGate     = raw(param_ids::kSparsityGate);
        cache.markovStayTime   = raw(param_ids::kMarkovStayTime);
        cache.markovStayFreq   = raw(param_ids::kMarkovStayFreq);
        cache.transientReset   = raw(param_ids::kTransientReset);
        cache.voronoiWeight    = raw(param_ids::kVoronoiWeight);
        cache.perlinWeight     = raw(param_ids::kPerlinWeight);
        cache.combWeight       = raw(param_ids::kCombWeight);
        cache.antiMasking      = raw(param_ids::kAntiMasking);

        cache.crossSynthGamma  = raw(param_ids::kCrossSynthGamma);
        cache.crossSynthPerm   = raw(param_ids::kCrossSynthPerm);
        cache.formantShift     = raw(param_ids::kFormantShift);
        cache.formantKeytrack  = raw(param_ids::kFormantKeytrack);
        cache.phaseMode        = raw(param_ids::kPhaseMode);

        cache.playMode         = raw(param_ids::kPlayMode);
        cache.polyphonyMode    = raw(param_ids::kPolyphonyMode);
        cache.timeStretch      = raw(param_ids::kTimeStretch);
        cache.masterGainDb     = raw(param_ids::kMasterGainDb);
        cache.envAttackSec     = raw(param_ids::kEnvAttackSec);
        cache.envDecaySec      = raw(param_ids::kEnvDecaySec);
        cache.envSustain       = raw(param_ids::kEnvSustain);
        cache.envReleaseSec    = raw(param_ids::kEnvReleaseSec);

        for (std::size_t i = 0; i < kMaxSources; ++i) {
            auto& sp     = cache.slots[i];
            sp.weight    = apvts.getRawParameterValue(param_ids::slotWeight(i));
            sp.gainDb    = apvts.getRawParameterValue(param_ids::slotGainDb(i));
            sp.pan       = apvts.getRawParameterValue(param_ids::slotPan(i));
            sp.pitchSemi = apvts.getRawParameterValue(param_ids::slotPitchSemi(i));
            sp.loopStart = apvts.getRawParameterValue(param_ids::slotLoopStart(i));
            sp.loopEnd   = apvts.getRawParameterValue(param_ids::slotLoopEnd(i));
            sp.enableH   = apvts.getRawParameterValue(param_ids::slotEnableH(i));
            sp.enableP   = apvts.getRawParameterValue(param_ids::slotEnableP(i));
            sp.enableR   = apvts.getRawParameterValue(param_ids::slotEnableR(i));
            sp.lock      = apvts.getRawParameterValue(param_ids::slotLock(i));
        }
    }

    // --------------------------------------------------------------------------
    // Lossless FLAC Base64 Encoding & Decoding for Self-Contained Preset State
    // --------------------------------------------------------------------------
    [[nodiscard]] static std::string encodeMonoAudioToFlacBase64(
        const std::vector<float>& audioMono,
        double                    sampleRate)
    {
        if (audioMono.empty()) {
            return {};
        }

        juce::FlacAudioFormat flacFormat;
        auto memStream = std::make_unique<juce::MemoryOutputStream>();
        juce::OutputStream* rawStreamPtr = memStream.get();

        std::unique_ptr<juce::AudioFormatWriter> writer(
            flacFormat.createWriterFor(
                rawStreamPtr,
                sampleRate,
                1u,
                24,
                {},
                0));

        if (!writer) {
            return {};
        }
        // Ownership of memStream was transferred to writer upon successful creation
        (void)memStream.release();

        juce::AudioBuffer<float> tempBuf(1, static_cast<int>(audioMono.size()));
        tempBuf.copyFrom(0, 0, audioMono.data(), static_cast<int>(audioMono.size()));
        writer->writeFromAudioSampleBuffer(tempBuf, 0, tempBuf.getNumSamples());
        writer->flush();

        auto* outMem = dynamic_cast<juce::MemoryOutputStream*>(rawStreamPtr);
        if (outMem == nullptr) {
            return {};
        }
        return juce::Base64::toBase64(outMem->getData(), outMem->getDataSize()).toStdString();
    }

    [[nodiscard]] static bool decodeFlacBase64ToMonoAudio(
        const std::string&  flacBase64,
        std::vector<float>& outAudioMono,
        double&             outSampleRate)
    {
        if (flacBase64.empty()) {
            return false;
        }

        juce::MemoryOutputStream decodedBytes;
        if (!juce::Base64::convertFromBase64(decodedBytes, juce::String(flacBase64))) {
            return false;
        }

        juce::FlacAudioFormat flacFormat;
        auto memInput = std::make_unique<juce::MemoryInputStream>(
            decodedBytes.getData(),
            decodedBytes.getDataSize(),
            false);

        std::unique_ptr<juce::AudioFormatReader> reader(
            flacFormat.createReaderFor(memInput.release(), true));
        if (!reader || reader->lengthInSamples <= 0) {
            return false;
        }

        outSampleRate = reader->sampleRate;
        const int numSamples = static_cast<int>(reader->lengthInSamples);
        juce::AudioBuffer<float> tempBuf(1, numSamples);
        reader->read(&tempBuf, 0, numSamples, 0, true, false);

        outAudioMono.assign(
            tempBuf.getReadPointer(0),
            tempBuf.getReadPointer(0) + numSamples);
        return true;
    }
};

#endif // SPECTILE_HAS_JUCE

} // namespace spectile::engine
