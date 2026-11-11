#pragma once

/**
 * @file SpectralGenome.hpp
 * @brief Settore 2 — 24-D Spectral Genome, 6 Pre-Calibrated Musical Archetypes,
 *        6 Orthogonal QR+SLERP Evolutionary Branches in Logit Tangent Space,
 *        and 3 Macro Controls Mapping for SpecTile.
 *
 * @details
 * Implements the mathematical formulation from Section 2.2 of the SpecTile
 * Executive Architecture Plan:
 *  - The 24-dimensional Spectral Genome g in [0, 1]^24 partitioned into
 *    5 Functional Chromosomes and decoded into physical DSP/Rule parameters.
 *  - The Mixture Model of 6 Pre-Calibrated Musical Archetypes (A_1 .. A_6)
 *    with exact centroids mu^(a) and standard deviations sigma^(a) designed
 *    to prevent the "Spectral Mud Paradox".
 *  - Logit-Riemannian tangent projection z_0 = ln(g_0 / (1 - g_0)), generation
 *    of 6 semantically distinct directional vectors, orthonormalization via
 *    Modified Gram-Schmidt QR Factorization (U^T U = I_6), and C^infinity
 *    spherical geodesic interpolation (SLERP) g(rho, theta).
 *  - Continuous mapping equations for the 3 Macro Controls:
 *      1. Spectral Tile Size (Delta_t x Delta_f)
 *      2. Rule Order vs. Mutation
 *      3. Phase & Spectral Glue
 *
 * All operations are real-time safe, deterministic, zero-allocation, and C++20 constexpr/noexcept ready.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string_view>

namespace spectile::rules {

// ============================================================================
// Core Dimensional Constants
// ============================================================================

inline constexpr std::size_t kGenomeDim      = 24;
inline constexpr std::size_t kNumArchetypes  = 6;
inline constexpr std::size_t kNumBranches    = 6;
inline constexpr std::size_t kMaxSamples     = 8;
inline constexpr std::size_t kDefaultBands   = 64;
inline constexpr std::size_t kMaxBands       = 128;
inline constexpr std::size_t kDefaultFftBins = 2049;
inline constexpr std::size_t kPaddedFftBins  = 2064; // Multiple of 16 floats (64-byte AVX2/AVX-512 aligned)

using GenomeVec  = std::array<float, kGenomeDim>;
using TangentVec = std::array<float, kGenomeDim>;

// ============================================================================
// The 24 Genes & 5 Functional Chromosomes
// ============================================================================

/**
 * @brief Enumeration of the 24 genes in the Spectral Genome g in [0, 1]^24,
 *        grouped into 5 Functional Chromosomes.
 */
enum class Gene : std::size_t {
    // --- Chromosome I: Global Rule Family Mixing & Softmax/Sparsity (Genes 0..4) ---
    AlphaGeo            = 0,  ///< Weight of Family 1 (Geometric Tiling) alpha_Geo in [0.0, 4.0]
    AlphaDesc           = 1,  ///< Weight of Family 2 (Descriptor Competition) alpha_Desc in [0.0, 4.0]
    AlphaMark           = 2,  ///< Weight of Family 3 (2D Causal Markov) alpha_Mark in [0.0, 3.5]
    SoftmaxTemperature  = 3,  ///< Softmax temperature tau in [0.03, 1.50] (log-scaled)
    SparsityGate        = 4,  ///< Spectral Sparsity Gating threshold eta_sparse in [0.0, 0.85]

    // --- Chromosome II: Geometric Tiling Sub-Rules & Scales (Genes 5..10) ---
    VoronoiMix          = 5,  ///< Laguerre 2D Voronoi weight beta_Vor in [0.0, 1.0]
    SimplexMix          = 6,  ///< Multi-octave 2D Simplex fBm weight beta_Perlin in [0.0, 1.0]
    CombMix             = 7,  ///< von Mises Harmonic/ERB Comb weight beta_Comb in [0.0, 1.0]
    TileScaleTime       = 8,  ///< Temporal tile scale sigma_t in [0.010 s, 1.200 s] & seed density Q
    TileScaleFreq       = 9,  ///< Frequency tile scale sigma_f in [0.5 ERB, 16.0 ERB]
    GeoDriftAndSharpness= 10, ///< Voronoi/Simplex LFO drift speed & Comb concentration kappa_comb

    // --- Chromosome III: Spectromorphological Competition & Anti-Masking (Genes 11..15) ---
    OnsetSplitWeight    = 11, ///< Onset/Continuant Spectromorphological Split gain gamma_split in [0.0, 3.5]
    OnsetFluxThreshold  = 12, ///< Spectral Flux sigmoid threshold theta_flux in [0.08, 0.88]
    FlatnessCompWeight  = 13, ///< Tonal vs. Noisy competition gain gamma_flat in [-2.5, +2.5]
    HoleFillerWeight    = 14, ///< Energy Complementarity weight gamma_comp in [0.0, 3.5]
    AntiMaskingLambda   = 15, ///< Anti-Masking coefficient lambda_anti in [0.0, 1.5]

    // --- Chromosome IV: 2D Causal Markov Continuity & Transient Reset (Genes 16..18) ---
    MarkovStayTime      = 16, ///< Temporal persistence probability P_stay,t in [0.15, 0.995]
    MarkovStayFreq      = 17, ///< Frequency persistence probability P_stay,f in [0.15, 0.990]
    TransientResetEta   = 18, ///< Instantaneous Markov memory reset on onset attacks eta_reset in [0.0, 1.0]

    // --- Chromosome V: Cross-Synthesis, Phase Glue, Feathering & Sub-Lock (Genes 19..23) ---
    CrossSynthesisChi   = 19, ///< Local Source-Filter permutation blend chi_cross in [0.0, 1.0]
    EnvelopeExponent    = 20, ///< Formant True Envelope exponent gamma in [0.25, 1.45]
    PhaseGlueBeta       = 21, ///< Causal MS-PGHI vs. Identity Phase-Locked blend beta_glue in [0.0, 1.0]
    Feathering2DSigma   = 22, ///< 2D Spectral Feathering Gaussian smoothing width in [0.0, 1.0]
    SubBassLockStrength = 23, ///< Sub-150Hz Butterworth Frequency Band Lock strength Lambda_sub in [0.0, 12.0]

    Count               = 24
};

/**
 * @brief Identifiers for the 5 Functional Chromosomes of the Spectral Genome.
 */
enum class Chromosome : uint8_t {
    GlobalMixAndSparsity = 0, ///< Genes 0..4
    GeometricTiling      = 1, ///< Genes 5..10
    FeatureCompetition   = 2, ///< Genes 11..15
    MarkovContinuity     = 3, ///< Genes 16..18
    CrossSynthAndGlue    = 4  ///< Genes 19..23
};

/**
 * @brief Identifiers for the 6 Pre-Calibrated Musical Archetypes (A_1 .. A_6).
 */
enum class ArchetypeId : std::size_t {
    HarmonicInterleave    = 0, ///< A_1: Cyclic odd/even harmonic & ERB comb weaving
    TransientFormantGraft = 1, ///< A_2: Attack transients grafted onto vocal/acoustic formant envelopes
    VoronoiSpectralShatter= 2, ///< A_3: Crisp, fast-drifting Laguerre Voronoi time-frequency shards
    ComplementarityFill   = 3, ///< A_4: Anti-masking hole filler injecting micro-textures into spectral valleys
    SpectralFreezeMorph   = 4, ///< A_5: Nebulous multi-octave Simplex clouds with high Markov persistence
    FormantChoir          = 5  ///< A_6: Full cross-synthesis formant choir with high MS-PGHI phase glue
};

// ============================================================================
// Mathematical Utilities (Logit / Sigmoid / Clamping / PRNG)
// ============================================================================

namespace detail {

inline constexpr float kEps      = 1.0e-6f;
inline constexpr float kLogitEps = 1.0e-4f;
inline constexpr float kTwoPi    = 2.0f * std::numbers::pi_v<float>;
inline constexpr float kHalfPi   = 0.5f * std::numbers::pi_v<float>;

[[nodiscard]] constexpr float clamp01(float x) noexcept {
    return std::clamp(x, 0.0f, 1.0f);
}

[[nodiscard]] constexpr float clampOpen01(float x) noexcept {
    return std::clamp(x, kLogitEps, 1.0f - kLogitEps);
}

[[nodiscard]] constexpr float lerp(float a, float b, float t) noexcept {
    return a + t * (b - a);
}

/**
 * @brief Standard logistic sigmoid sigma(z) = 1 / (1 + exp(-z)).
 */
[[nodiscard]] inline float sigmoid(float z) noexcept {
    const float clamped = std::clamp(z, -20.0f, 20.0f);
    return 1.0f / (1.0f + std::exp(-clamped));
}

/**
 * @brief Logit tangent projection z = ln(g / (1 - g)).
 */
[[nodiscard]] inline float logit(float g) noexcept {
    const float gSafe = clampOpen01(g);
    return std::log(gSafe / (1.0f - gSafe));
}

/**
 * @brief Deterministic stateless SplitMix64 / Xorshift PRNG for real-time safe mutation.
 */
struct DeterministicRng {
    uint64_t state{0x9E3779B97F4A7C15ULL};

    constexpr explicit DeterministicRng(uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept
        : state(seed != 0 ? seed : 0x9E3779B97F4A7C15ULL) {}

    [[nodiscard]] constexpr uint64_t nextU64() noexcept {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    /// Uniform float in (0, 1)
    [[nodiscard]] constexpr float nextUniformOpen01() noexcept {
        const uint32_t u = static_cast<uint32_t>(nextU64() >> 40); // 24-bit mantissa
        return (static_cast<float>(u) + 0.5f) * (1.0f / 16777216.0f);
    }

    /// Standard normal N(0, 1) via Box-Muller
    [[nodiscard]] inline float nextGaussian() noexcept {
        const float u1 = std::max(nextUniformOpen01(), 1.0e-7f);
        const float u2 = nextUniformOpen01();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(kTwoPi * u2);
    }
};

} // namespace detail

// ============================================================================
// 3 Macro Controls State
// ============================================================================

/**
 * @brief The 3 Front-Panel Macro Controls from Section 4.1 ("The Chimera Button").
 *        Each value is normalized in [0, 1], with 0.5 representing the neutral center.
 */
struct MacroControls {
    float spectralTileSize    = 0.5f; ///< Macro 1: Micro FFT shards (0) <-> Wide formant bands (1)
    float ruleOrderVsMutation = 0.5f; ///< Macro 2: Deterministic descriptor/comb order (0) <-> Stochastic Voronoi/LFO mutation (1)
    float phaseAndSpectralGlue= 0.5f; ///< Macro 3: Raw crisp switching (0) <-> Smooth 2D feathering + full MS-PGHI glue (1)
};

// ============================================================================
// Decoded Physical DSP & Rule Parameters
// ============================================================================

/**
 * @brief Complete set of physical DSP and Rule Engine parameters decoded from
 *        the 24-D Spectral Genome g in [0, 1]^24 and modulated by the 3 Macro Controls.
 */
struct alignas(64) DecodedGenomeParams {
    // --- Chromosome I: Family Gains, Softmax & Sparsity ---
    float alphaGeo          = 1.5f;  ///< Family 1 logit gain alpha_Geo in [0.0, 4.5]
    float alphaDesc         = 1.5f;  ///< Family 2 logit gain alpha_Desc in [0.0, 4.5]
    float alphaMark         = 1.2f;  ///< Family 3 logit gain alpha_Mark in [0.0, 3.8]
    float softmaxTempTau    = 0.18f; ///< Softmax temperature tau in [0.03, 1.50]
    float sparsityGateEta   = 0.25f; ///< Spectral Sparsity Gating threshold eta_sparse in [0.0, 0.85]

    // --- Chromosome II: Geometric Tiling (Voronoi, Simplex fBm, von Mises Comb) ---
    float betaVoronoi       = 0.45f; ///< Weight of 2D Laguerre Voronoi potential beta_Vor
    float betaSimplex       = 0.30f; ///< Weight of 2D Simplex fBm potential beta_Perlin
    float betaComb          = 0.25f; ///< Weight of von Mises Comb potential beta_Comb
    float tileScaleTimeSec  = 0.18f; ///< Temporal tile scale sigma_t in [0.010, 1.500] seconds
    float tileScaleFreqErb  = 3.0f;  ///< Frequency tile scale sigma_f in [0.5, 18.0] ERB bands
    std::size_t voronoiSeeds= 32;    ///< Active Halton seed count Q in [N, 128]
    std::size_t simplexOctaves = 3;  ///< Simplex fBm octave count in [1, 4]
    float lfoDriftHz        = 0.45f; ///< Seed & Simplex drift LFO frequency in [0.02, 5.0] Hz
    float combKappa         = 3.5f;  ///< von Mises concentration parameter kappa_comb in [0.5, 12.0]
    bool  combHarmonicMode  = false; ///< true = odd/even harmonic partials, false = cyclic ERB bands

    // --- Chromosome III: Descriptor Competition & Anti-Masking ---
    float gammaSplit        = 1.4f;  ///< Onset/Continuant split weight gamma_split in [0.0, 3.5]
    float onsetFluxThreshold= 0.35f; ///< Spectral Flux threshold theta_flux in [0.05, 0.90]
    float onsetFluxSlope    = 12.0f; ///< Sigmoid steepness k_flux in [4.0, 24.0]
    float gammaFlatness     = 0.8f;  ///< Tonal vs. Noisy competition weight gamma_flat in [-2.5, +2.5]
    float gammaComplement   = 1.2f;  ///< Energy Complementarity weight gamma_comp in [0.0, 3.5]
    float lambdaAntiMask    = 0.75f; ///< Anti-Masking Hole-Filler strength lambda_anti in [0.0, 1.5]

    // --- Chromosome IV: 2D Causal Markov Chain & Transient Reset ---
    float markovStayTimeProb= 0.85f; ///< Horizontal persistence P_stay,t in [0.10, 0.996]
    float markovStayFreqProb= 0.78f; ///< Vertical persistence P_stay,f in [0.10, 0.992]
    float markovKappaTime   = 1.73f; ///< Log-odds temporal gain kappa_t = ln(P_t / (1 - P_t))
    float markovKappaFreq   = 1.26f; ///< Log-odds frequency gain kappa_f = ln(P_f / (1 - P_f))
    float transientResetEta = 0.85f; ///< Instantaneous attack reset factor eta_reset in [0.0, 1.0]

    // --- Chromosome V: Local Cross-Synthesis, Phase Glue, Feathering & Sub-Lock ---
    float crossSynthChi     = 0.35f; ///< Source-Filter permutation blend chi_cross in [0.0, 1.0]
    std::size_t crossShiftDelta = 1; ///< Circular permutation shift Delta in {1 .. N-1}
    float envelopeGamma     = 0.85f; ///< Formant True Envelope exponent gamma in [0.25, 1.50]
    float phaseGlueBeta     = 0.75f; ///< MS-PGHI vs. Identity Phase-Locked blend beta_glue in [0.0, 1.0]
    float featherTimeAlpha  = 0.35f; ///< Temporal feathering IIR coefficient in [0.0, 0.92]
    float featherFreqSigma  = 2.5f;  ///< Gaussian frequency feathering sigma in FFT bins [0.0, 12.0]
    float subBassLockLambda = 6.0f;  ///< Sub-150Hz Band Lock logit boost Lambda_sub in [0.0, 14.0]
    float subBassCutoffHz   = 150.0f;///< Butterworth Band Lock cutoff frequency in [80.0, 180.0] Hz
    std::size_t subBassOwner= 0;     ///< Default slot index owning the sub-bass (< 150 Hz)
};

// ============================================================================
// Table of the 6 Pre-Calibrated Musical Archetypes (A_1 .. A_6)
// ============================================================================

/**
 * @brief Specification of a Pre-Calibrated Musical Archetype in the 24-D Genome space.
 */
struct ArchetypeSpec {
    ArchetypeId      id;
    std::string_view name;
    std::string_view description;
    GenomeVec        mu;    ///< Pre-calibrated 24-D centroid vector mu^(a) in (0, 1)^24
    GenomeVec        sigma; ///< Pre-calibrated 24-D standard deviation vector sigma^(a)
};

/**
 * @brief Complete Table of the 6 Pre-Calibrated Musical Archetypes (Section 2.2).
 *
 * Calibrated so that every archetype avoids the "Spectral Mud Paradox":
 *  - Softmax temperature tau is matched to tile size and sparsity gate.
 *  - Small tiles enforce either high Markov persistence or high Transient Reset.
 *  - Sub-150Hz bass lock is preserved on bass/percussive/full-range archetypes.
 */
inline constexpr std::array<ArchetypeSpec, kNumArchetypes> kArchetypeTable = {{
    // ------------------------------------------------------------------------
    // A_1: Harmonic Interleave
    // Alternating odd/even partials & cyclic ERB bands via von Mises Comb,
    // high tonal purity, strong vertical/horizontal coherence, high phase glue.
    // ------------------------------------------------------------------------
    {
        ArchetypeId::HarmonicInterleave,
        "Harmonic Interleave",
        "Cyclic von Mises harmonic comb weaving odd and even partials across samples with high phase coherence.",
        // mu^(1):
        // [I: Geo, Desc, Mark, Tau, Sparse]
        { 0.78f, 0.45f, 0.68f, 0.22f, 0.55f,
        // [II: Vor, Simp, Comb, ScaleT, ScaleF, Drift/Sharp]
          0.12f, 0.15f, 0.92f, 0.65f, 0.25f, 0.82f,
        // [III: Split, FluxThr, Flat, Comp, AntiMask]
          0.30f, 0.48f, 0.85f, 0.35f, 0.40f,
        // [IV: StayT, StayF, ResetEta]
          0.84f, 0.32f, 0.60f,
        // [V: CrossChi, EnvGamma, PhaseGlue, Feather, SubLock]
          0.35f, 0.72f, 0.88f, 0.42f, 0.75f },
        // sigma^(1):
        { 0.08f, 0.10f, 0.08f, 0.06f, 0.08f,
          0.06f, 0.07f, 0.05f, 0.12f, 0.08f, 0.09f,
          0.10f, 0.08f, 0.07f, 0.10f, 0.09f,
          0.06f, 0.08f, 0.10f,
          0.10f, 0.07f, 0.05f, 0.08f, 0.08f }
    },

    // ------------------------------------------------------------------------
    // A_2: Transient-Formant Graft
    // Spectromorphological Onset/Continuant split with instantaneous Markov
    // reset on attacks and deep Source-Filter cross-synthesis on sustains.
    // ------------------------------------------------------------------------
    {
        ArchetypeId::TransientFormantGraft,
        "Transient-Formant Graft",
        "Crisp onset attack replacement via Spectral Flux paired with True Envelope formant grafting on sustains.",
        // mu^(2):
        // [I: Geo, Desc, Mark, Tau, Sparse]
        { 0.32f, 0.90f, 0.72f, 0.18f, 0.62f,
        // [II: Vor, Simp, Comb, ScaleT, ScaleF, Drift/Sharp]
          0.25f, 0.35f, 0.30f, 0.42f, 0.58f, 0.40f,
        // [III: Split, FluxThr, Flat, Comp, AntiMask]
          0.94f, 0.32f, 0.68f, 0.48f, 0.55f,
        // [IV: StayT, StayF, ResetEta]
          0.86f, 0.78f, 0.96f,
        // [V: CrossChi, EnvGamma, PhaseGlue, Feather, SubLock]
          0.85f, 0.80f, 0.82f, 0.48f, 0.85f },
        // sigma^(2):
        { 0.09f, 0.06f, 0.08f, 0.05f, 0.08f,
          0.08f, 0.09f, 0.08f, 0.10f, 0.10f, 0.09f,
          0.04f, 0.07f, 0.09f, 0.10f, 0.09f,
          0.05f, 0.07f, 0.03f,
          0.07f, 0.06f, 0.06f, 0.08f, 0.06f }
    },

    // ------------------------------------------------------------------------
    // A_3: Voronoi Spectral Shatter
    // High-density Laguerre 2D Voronoi tessellation with fast LFO drift,
    // crisp low Softmax temperature tau, and sharp neon tile boundaries.
    // ------------------------------------------------------------------------
    {
        ArchetypeId::VoronoiSpectralShatter,
        "Voronoi Spectral Shatter",
        "Dynamic Laguerre 2D Voronoi tessellation fracturing the spectrum into drifting crystalline shards.",
        // mu^(3):
        // [I: Geo, Desc, Mark, Tau, Sparse]
        { 0.92f, 0.35f, 0.52f, 0.12f, 0.70f,
        // [II: Vor, Simp, Comb, ScaleT, ScaleF, Drift/Sharp]
          0.95f, 0.20f, 0.15f, 0.28f, 0.32f, 0.78f,
        // [III: Split, FluxThr, Flat, Comp, AntiMask]
          0.45f, 0.40f, 0.42f, 0.38f, 0.45f,
        // [IV: StayT, StayF, ResetEta]
          0.62f, 0.65f, 0.82f,
        // [V: CrossChi, EnvGamma, PhaseGlue, Feather, SubLock]
          0.28f, 0.68f, 0.74f, 0.26f, 0.78f },
        // sigma^(3):
        { 0.05f, 0.10f, 0.09f, 0.04f, 0.07f,
          0.04f, 0.08f, 0.07f, 0.09f, 0.09f, 0.08f,
          0.10f, 0.09f, 0.11f, 0.10f, 0.09f,
          0.08f, 0.08f, 0.07f,
          0.09f, 0.08f, 0.07f, 0.07f, 0.08f }
    },

    // ------------------------------------------------------------------------
    // A_4: Complementarity Fill
    // Anti-masking Hole-Filler (Phi^Comp) dominance: secondary samples inject
    // micro-textures and foley into spectral valleys without masking main peaks.
    // ------------------------------------------------------------------------
    {
        ArchetypeId::ComplementarityFill,
        "Complementarity Fill",
        "Anti-masking spectral hole-filler that weaves secondary textures into the valleys of dominant sources.",
        // mu^(4):
        // [I: Geo, Desc, Mark, Tau, Sparse]
        { 0.38f, 0.94f, 0.66f, 0.24f, 0.58f,
        // [II: Vor, Simp, Comb, ScaleT, ScaleF, Drift/Sharp]
          0.35f, 0.42f, 0.25f, 0.48f, 0.45f, 0.46f,
        // [III: Split, FluxThr, Flat, Comp, AntiMask]
          0.55f, 0.38f, 0.35f, 0.95f, 0.92f,
        // [IV: StayT, StayF, ResetEta]
          0.76f, 0.72f, 0.78f,
        // [V: CrossChi, EnvGamma, PhaseGlue, Feather, SubLock]
          0.22f, 0.74f, 0.80f, 0.46f, 0.90f },
        // sigma^(4):
        { 0.09f, 0.05f, 0.08f, 0.06f, 0.08f,
          0.09f, 0.09f, 0.08f, 0.10f, 0.09f, 0.09f,
          0.10f, 0.08f, 0.10f, 0.04f, 0.05f,
          0.07f, 0.07f, 0.08f,
          0.08f, 0.07f, 0.06f, 0.08f, 0.05f }
    },

    // ------------------------------------------------------------------------
    // A_5: Spectral Freeze Morph
    // Multi-octave 2D Simplex fBm clouds + ultra-high horizontal Markov
    // persistence + wide 2D Gaussian feathering + full MS-PGHI Phase Glue.
    // ------------------------------------------------------------------------
    {
        ArchetypeId::SpectralFreezeMorph,
        "Spectral Freeze Morph",
        "Evolving multi-octave Simplex fBm clouds with high Markov temporal persistence and seamless MS-PGHI glue.",
        // mu^(5):
        // [I: Geo, Desc, Mark, Tau, Sparse]
        { 0.82f, 0.30f, 0.92f, 0.52f, 0.32f,
        // [II: Vor, Simp, Comb, ScaleT, ScaleF, Drift/Sharp]
          0.28f, 0.94f, 0.18f, 0.85f, 0.76f, 0.25f,
        // [III: Split, FluxThr, Flat, Comp, AntiMask]
          0.20f, 0.62f, 0.72f, 0.42f, 0.38f,
        // [IV: StayT, StayF, ResetEta]
          0.95f, 0.88f, 0.22f,
        // [V: CrossChi, EnvGamma, PhaseGlue, Feather, SubLock]
          0.52f, 0.78f, 0.95f, 0.84f, 0.70f },
        // sigma^(5):
        { 0.07f, 0.09f, 0.05f, 0.08f, 0.08f,
          0.08f, 0.04f, 0.07f, 0.07f, 0.08f, 0.07f,
          0.08f, 0.09f, 0.08f, 0.09f, 0.09f,
          0.03f, 0.05f, 0.08f,
          0.10f, 0.07f, 0.03f, 0.06f, 0.09f }
    },

    // ------------------------------------------------------------------------
    // A_6: Formant Choir
    // Deep Source-Filter Cross-Synthesis (chi_cross -> 0.92) + Harmonic Comb +
    // Tonal Flatness preference, sculpting crystalline/instrumental excitations
    // with vocal/acoustic True Envelopes.
    // ------------------------------------------------------------------------
    {
        ArchetypeId::FormantChoir,
        "Formant Choir",
        "Full Source-Filter chimeric resynthesis sculpting harmonic comb structures with vocal True Envelopes.",
        // mu^(6):
        // [I: Geo, Desc, Mark, Tau, Sparse]
        { 0.65f, 0.72f, 0.80f, 0.34f, 0.48f,
        // [II: Vor, Simp, Comb, ScaleT, ScaleF, Drift/Sharp]
          0.22f, 0.58f, 0.76f, 0.72f, 0.64f, 0.52f,
        // [III: Split, FluxThr, Flat, Comp, AntiMask]
          0.42f, 0.45f, 0.90f, 0.50f, 0.52f,
        // [IV: StayT, StayF, ResetEta]
          0.88f, 0.82f, 0.58f,
        // [V: CrossChi, EnvGamma, PhaseGlue, Feather, SubLock]
          0.93f, 0.86f, 0.92f, 0.68f, 0.82f },
        // sigma^(6):
        { 0.08f, 0.08f, 0.07f, 0.07f, 0.08f,
          0.08f, 0.09f, 0.08f, 0.09f, 0.09f, 0.08f,
          0.09f, 0.08f, 0.05f, 0.09f, 0.09f,
          0.05f, 0.06f, 0.09f,
          0.04f, 0.06f, 0.04f, 0.07f, 0.07f }
    }
}};

// ============================================================================
// 6 QR+SLERP Orthogonal Evolutionary Branches in Logit Tangent Space
// ============================================================================

/**
 * @brief Holds the pre-computed Logit-Riemannian tangent basis for a seed genome g_0:
 *  - Seed genome g_0 in (0, 1)^24 and its logit projection z_0 in R^24
 *  - 6 Orthonormalized tangent branch vectors U = [u_1, ..., u_6] (U^T U = I_6)
 *    computed via Modified Gram-Schmidt QR Factorization
 *  - 6 Prior attraction vectors c_prior^(m) preventing high-radius mutations
 *    from falling into the Spectral Mud Paradox.
 */
struct EvolutionaryBranchBasis {
    GenomeVec  seedGenome{};
    TangentVec z0{};
    ArchetypeId primaryArchetype{ArchetypeId::HarmonicInterleave};
    std::array<TangentVec, kNumBranches> orthonormalBranches{}; ///< Columns u_1 .. u_6 of Q in QR
    std::array<TangentVec, kNumBranches> priorAnchors{};        ///< c_prior^(1..6) in logit space
    float branchRadiusScale = 3.2f;                             ///< R_branch geodesic reach in logit units
};

/**
 * @brief Names of the 6 semantically distinct Evolutionary Branches (Section 2.2).
 */
inline constexpr std::array<std::string_view, kNumBranches> kBranchNames = {{
    "Branch 1: Sample Role & Density Rotation",
    "Branch 2: Geometric Voronoi Fragmentation",
    "Branch 3: Source-Filter Cross-Synthesis Surge",
    "Branch 4: Transient/Sustain Contrast & Reset",
    "Branch 5: Simplex Liquefaction & Freeze",
    "Branch 6: Cross-Archetype Geodesic Bridge"
}};

// ============================================================================
// SpectralGenomeEngine — Sampling, Decoding, Macro Mapping & QR+SLERP Evolution
// ============================================================================

class SpectralGenomeEngine {
public:
    /**
     * @brief Samples a musical seed genome g_0 in [0, 1]^24 from a specified Archetype
     *        (or from the 6-Archetype Mixture Model) and applies anti-mud invariant guards.
     */
    [[nodiscard]] static GenomeVec sampleFromArchetype(
        ArchetypeId archetype,
        uint64_t seed,
        float mutationSpread = 1.0f) noexcept
    {
        const auto idx = std::min<std::size_t>(static_cast<std::size_t>(archetype), kNumArchetypes - 1);
        const ArchetypeSpec& spec = kArchetypeTable[idx];
        detail::DeterministicRng rng(seed);

        GenomeVec g{};
        const float spread = std::clamp(mutationSpread, 0.0f, 2.5f);

        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            // Sample in logit-normal space around mu^(a)[d] so boundaries (0, 1) are respected naturally
            const float zMu = detail::logit(spec.mu[d]);
            // Convert standard deviation in probability space to tangent logit scale via 1 / (mu * (1 - mu))
            const float deriv = 1.0f / std::max(spec.mu[d] * (1.0f - spec.mu[d]), 0.08f);
            const float zSigma = spec.sigma[d] * deriv * spread;
            const float zSample = zMu + rng.nextGaussian() * zSigma;
            g[d] = detail::sigmoid(zSample);
        }

        enforceAntiMudInvariants(g);
        return g;
    }

    /**
     * @brief Enforces the 3 Anti-Mud Invariants (Section 2.2) so a genome never produces:
     *        (1) Entropic collapse to 1/N (high Softmax tau with low Sparsity Gate),
     *        (2) Phase gargling on micro-tiles (small scale without Markov persistence or Phase Glue),
     *        (3) Total rule starvation (all 3 family weights simultaneously near zero).
     */
    static constexpr void enforceAntiMudInvariants(GenomeVec& g) noexcept {
        for (float& val : g) {
            val = detail::clampOpen01(val);
        }

        // Invariant 1: Prevent all three rule families from collapsing to zero simultaneously
        const float familySum = g[static_cast<std::size_t>(Gene::AlphaGeo)]
                              + g[static_cast<std::size_t>(Gene::AlphaDesc)]
                              + g[static_cast<std::size_t>(Gene::AlphaMark)];
        if (familySum < 0.75f) {
            const float boost = 0.75f / std::max(familySum, 1.0e-3f);
            g[static_cast<std::size_t>(Gene::AlphaGeo)]  = detail::clampOpen01(g[static_cast<std::size_t>(Gene::AlphaGeo)] * boost);
            g[static_cast<std::size_t>(Gene::AlphaDesc)] = detail::clampOpen01(g[static_cast<std::size_t>(Gene::AlphaDesc)] * boost);
            g[static_cast<std::size_t>(Gene::AlphaMark)] = detail::clampOpen01(g[static_cast<std::size_t>(Gene::AlphaMark)] * boost);
        }

        // Invariant 2: Anti-Entropic Collapse — if Softmax temperature is high (> 0.60),
        // enforce a minimum Sparsity Gate so weak samples do not create muddy 1/N averaging
        const float tauNorm = g[static_cast<std::size_t>(Gene::SoftmaxTemperature)];
        if (tauNorm > 0.60f) {
            const float minSparsity = 0.35f * (tauNorm - 0.60f) / 0.40f + 0.22f;
            g[static_cast<std::size_t>(Gene::SparsityGate)] =
                std::max(g[static_cast<std::size_t>(Gene::SparsityGate)], minSparsity);
        }

        // Invariant 3: Micro-tile Phase Coherence — if temporal tile scale is very small (< 0.25),
        // enforce either high Phase Glue or high Transient Reset + Markov persistence
        const float scaleT = g[static_cast<std::size_t>(Gene::TileScaleTime)];
        if (scaleT < 0.25f) {
            g[static_cast<std::size_t>(Gene::PhaseGlueBeta)] =
                std::max(g[static_cast<std::size_t>(Gene::PhaseGlueBeta)], 0.62f);
            g[static_cast<std::size_t>(Gene::TransientResetEta)] =
                std::max(g[static_cast<std::size_t>(Gene::TransientResetEta)], 0.65f);
        }
    }

    /**
     * @brief Builds the 6 Orthogonal Evolutionary Branches around a seed genome g_0
     *        using Modified Gram-Schmidt QR Factorization in the 24-D Logit Tangent Space.
     *
     * @param g0              Seed genome in (0, 1)^24
     * @param baseArchetype   Primary archetype of g0 (used to orient Branch 6 toward the complementary archetype)
     * @param branchRadius    Geodesic excursion scale R_branch in logit space (default 3.2)
     */
    [[nodiscard]] static EvolutionaryBranchBasis buildOrthogonalBranchesQR(
        const GenomeVec& g0,
        ArchetypeId baseArchetype = ArchetypeId::HarmonicInterleave,
        float branchRadius = 3.2f) noexcept
    {
        EvolutionaryBranchBasis basis{};
        basis.seedGenome       = g0;
        basis.primaryArchetype = baseArchetype;
        basis.branchRadiusScale= branchRadius;

        // 1. Project seed genome g_0 into the unconstrained logit tangent space R^24
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            basis.z0[d] = detail::logit(g0[d]);
        }

        // 2. Construct the 6 semantically distinct raw tangent vectors V = [v_1, ..., v_6] in R^24
        std::array<TangentVec, kNumBranches> rawVectors{};

        // --- Branch 1: Sample Role & Density Rotation (Comb <-> Complementarity + Sparsity shift) ---
        rawVectors[0] = {
            +0.25f, +0.35f, -0.15f, -0.20f, +0.45f, // Chromosome I
            -0.20f, +0.15f, +0.65f, +0.10f, -0.30f, +0.40f, // Chromosome II
            +0.20f, -0.15f, +0.45f, +0.55f, +0.50f, // Chromosome III
            +0.10f, -0.35f, +0.20f, // Chromosome IV
            +0.15f, +0.10f, +0.20f, -0.15f, +0.25f  // Chromosome V
        };

        // --- Branch 2: Geometric Fragmentation (Voronoi Shatter, fast LFO drift, crisp low tau) ---
        rawVectors[1] = {
            +0.80f, -0.30f, -0.25f, -0.65f, +0.50f, // Chromosome I
            +0.90f, -0.35f, -0.30f, -0.70f, -0.60f, +0.75f, // Chromosome II
            +0.15f, -0.20f, -0.15f, -0.10f, +0.10f, // Chromosome III
            -0.40f, -0.30f, +0.45f, // Chromosome IV
            -0.20f, -0.10f, +0.15f, -0.55f, +0.20f  // Chromosome V
        };

        // --- Branch 3: Source-Filter Cross-Synthesis Surge (Formant Grafting chi_cross & gamma_env) ---
        rawVectors[2] = {
            -0.15f, +0.35f, +0.30f, +0.15f, +0.10f, // Chromosome I
            -0.25f, +0.20f, +0.45f, +0.35f, +0.40f, +0.15f, // Chromosome II
            +0.25f, +0.10f, +0.60f, +0.15f, +0.20f, // Chromosome III
            +0.35f, +0.40f, +0.10f, // Chromosome IV
            +0.95f, +0.70f, +0.55f, +0.40f, +0.30f  // Chromosome V
        };

        // --- Branch 4: Transient/Sustain Contrast & Instantaneous Markov Reset ---
        rawVectors[3] = {
            -0.30f, +0.85f, +0.25f, -0.35f, +0.40f, // Chromosome I
            +0.10f, -0.25f, -0.15f, -0.35f, +0.20f, +0.25f, // Chromosome II
            +0.95f, -0.55f, +0.30f, +0.60f, +0.65f, // Chromosome III
            +0.30f, +0.20f, +0.95f, // Chromosome IV
            +0.30f, +0.15f, +0.25f, -0.20f, +0.60f  // Chromosome V
        };

        // --- Branch 5: Simplex Liquefaction & Freeze (Multi-octave fBm + High Markov Stay + Wide Feather) ---
        rawVectors[4] = {
            +0.45f, -0.45f, +0.85f, +0.65f, -0.40f, // Chromosome I
            -0.40f, +0.95f, -0.30f, +0.80f, +0.70f, -0.50f, // Chromosome II
            -0.55f, +0.45f, +0.40f, -0.20f, -0.25f, // Chromosome III
            +0.90f, +0.75f, -0.70f, // Chromosome IV
            +0.25f, +0.35f, +0.80f, +0.85f, +0.10f  // Chromosome V
        };

        // --- Branch 6: Cross-Archetype Geodesic Bridge (Tangent vector toward complementary archetype) ---
        const std::size_t currentArchIdx = static_cast<std::size_t>(baseArchetype) % kNumArchetypes;
        const std::size_t bridgeArchIdx  = (currentArchIdx + 3) % kNumArchetypes; // Diametrically opposite archetype
        const GenomeVec& targetMu        = kArchetypeTable[bridgeArchIdx].mu;
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            rawVectors[5][d] = detail::logit(targetMu[d]) - basis.z0[d];
        }

        // 3. Modified Gram-Schmidt QR Factorization (V = U * R, with U^T * U = I_6)
        // Guarantees mathematically that all 6 branches explore mutually orthogonal timbral directions.
        for (std::size_t col = 0; col < kNumBranches; ++col) {
            TangentVec u = rawVectors[col];

            // Subtract projections onto all previously orthonormalized columns 0 .. col-1
            // (Two-pass Modified Gram-Schmidt for maximum floating-point orthogonality)
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t prev = 0; prev < col; ++prev) {
                    const float proj = dotProduct(u, basis.orthonormalBranches[prev]);
                    for (std::size_t d = 0; d < kGenomeDim; ++d) {
                        u[d] -= proj * basis.orthonormalBranches[prev][d];
                    }
                }
            }

            // Normalize column u_col to unit Euclidean length ||u_col||_2 = 1
            const float norm = vectorNorm(u);
            if (norm > 1.0e-5f) {
                const float invNorm = 1.0f / norm;
                for (std::size_t d = 0; d < kGenomeDim; ++d) {
                    basis.orthonormalBranches[col][d] = u[d] * invNorm;
                }
            } else {
                // Fallback canonical unit vector orthogonal to previous columns if degenerate
                TangentVec fallback{};
                fallback[col % kGenomeDim] = 1.0f;
                for (std::size_t prev = 0; prev < col; ++prev) {
                    const float proj = dotProduct(fallback, basis.orthonormalBranches[prev]);
                    for (std::size_t d = 0; d < kGenomeDim; ++d) {
                        fallback[d] -= proj * basis.orthonormalBranches[prev][d];
                    }
                }
                const float fbNorm = std::max(vectorNorm(fallback), 1.0e-6f);
                for (std::size_t d = 0; d < kGenomeDim; ++d) {
                    basis.orthonormalBranches[col][d] = fallback[d] / fbNorm;
                }
            }

            // 4. Compute the archetype prior anchor c_prior^(col) in logit space
            const std::size_t priorArchIdx = (currentArchIdx + col) % kNumArchetypes;
            const GenomeVec& priorMu       = kArchetypeTable[priorArchIdx].mu;
            for (std::size_t d = 0; d < kGenomeDim; ++d) {
                basis.priorAnchors[col][d] = 0.35f * (detail::logit(priorMu[d]) - basis.z0[d]);
            }
        }

        return basis;
    }

    /**
     * @brief Evaluates the continuous C^infinity QR+SLERP Genome Evolution Equation (Section 2.2):
     *
     *        g(rho, theta) = sigma( z_0 + rho * R_branch * u(theta) + rho^2 * c_prior(theta) )
     *
     *        where u(theta) is interpolated along the unit hypersphere S^23 via exact SLERP
     *        between adjacent orthonormal branch vectors u_m and u_{(m+1) mod 6}.
     *
     * @param basis  Pre-computed QR orthonormal branch basis around g_0
     * @param rho    Radial excursion distance in [0, 1] (0 = seed genome g_0, 1 = branch tip)
     * @param theta  Angular coordinate in [0, 2*pi) selecting/blending across the 6 radial branches
     */
    [[nodiscard]] static GenomeVec evaluateBranchSlerp(
        const EvolutionaryBranchBasis& basis,
        float rho,
        float theta) noexcept
    {
        const float r = detail::clamp01(rho);
        if (r <= 1.0e-6f) {
            return basis.seedGenome;
        }

        // Wrap angle theta into [0, 2*pi)
        float wrappedTheta = std::fmod(theta, detail::kTwoPi);
        if (wrappedTheta < 0.0f) {
            wrappedTheta += detail::kTwoPi;
        }

        // Map angle to branch sector m in {0..5} and local fraction s in [0, 1)
        const float sectorCoord = (wrappedTheta / detail::kTwoPi) * static_cast<float>(kNumBranches);
        const auto m0 = static_cast<std::size_t>(sectorCoord) % kNumBranches;
        const auto m1 = (m0 + 1) % kNumBranches;
        const float s = sectorCoord - std::floor(sectorCoord);

        // Since u_m0 and u_m1 are orthonormal (u_m0 . u_m1 = 0, inter-branch angle Omega = pi/2),
        // exact geodesic SLERP on the hypersphere S^23 is:
        //   u(theta) = cos(s * pi/2) * u_m0 + sin(s * pi/2) * u_m1
        const float w0 = std::cos(s * detail::kHalfPi);
        const float w1 = std::sin(s * detail::kHalfPi);

        const float rBranch = r * basis.branchRadiusScale;
        const float rSq     = r * r;

        GenomeVec mutated{};
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            const float uTheta     = w0 * basis.orthonormalBranches[m0][d]
                                   + w1 * basis.orthonormalBranches[m1][d];
            const float cPriorTheta= w0 * basis.priorAnchors[m0][d]
                                   + w1 * basis.priorAnchors[m1][d];
            const float zMutated   = basis.z0[d] + rBranch * uTheta + rSq * cPriorTheta;
            mutated[d] = detail::sigmoid(zMutated);
        }

        enforceAntiMudInvariants(mutated);
        return mutated;
    }

    /**
     * @brief General 24-D Spherical Linear Interpolation (SLERP) in Logit-Riemannian space
     *        between any two genomes gA and gB for t in [0, 1].
     */
    [[nodiscard]] static GenomeVec slerpGenomes(
        const GenomeVec& gA,
        const GenomeVec& gB,
        float t) noexcept
    {
        const float alpha = detail::clamp01(t);
        TangentVec zA{};
        TangentVec zB{};
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            zA[d] = detail::logit(gA[d]);
            zB[d] = detail::logit(gB[d]);
        }

        const float normA = vectorNorm(zA);
        const float normB = vectorNorm(zB);

        GenomeVec out{};
        if (normA < 1.0e-5f || normB < 1.0e-5f) {
            for (std::size_t d = 0; d < kGenomeDim; ++d) {
                out[d] = detail::sigmoid(detail::lerp(zA[d], zB[d], alpha));
            }
            enforceAntiMudInvariants(out);
            return out;
        }

        const float cosOmega = std::clamp(dotProduct(zA, zB) / (normA * normB), -0.9999f, 0.9999f);
        const float omega    = std::acos(cosOmega);
        const float sinOmega = std::sin(omega);

        const float coeffA = (sinOmega > 1.0e-4f) ? (std::sin((1.0f - alpha) * omega) / sinOmega) : (1.0f - alpha);
        const float coeffB = (sinOmega > 1.0e-4f) ? (std::sin(alpha * omega) / sinOmega)          : alpha;
        const float targetNorm = detail::lerp(normA, normB, alpha);

        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            const float unitInterp = coeffA * (zA[d] / normA) + coeffB * (zB[d] / normB);
            out[d] = detail::sigmoid(unitInterp * targetNorm);
        }

        enforceAntiMudInvariants(out);
        return out;
    }

    /**
     * @brief Applies the 3 Macro Controls (Spectral Tile Size, Rule Order vs. Mutation,
     *        Phase & Spectral Glue) to a base genome g in [0, 1]^24 and decodes all
     *        physical DSP and Rule Engine parameters.
     *
     * @param baseGenome     The 24-D genome g in [0, 1]^24
     * @param macros         The 3 front-panel Macro controls in [0, 1]^3 (0.5 = neutral)
     * @param numActiveSlots Number of active sample slots N in [2, 8]
     */
    [[nodiscard]] static DecodedGenomeParams decodeGenome(
        const GenomeVec& baseGenome,
        const MacroControls& macros = {},
        std::size_t numActiveSlots = kMaxSamples) noexcept
    {
        const std::size_t N = std::clamp<std::size_t>(numActiveSlots, 2, kMaxSamples);

        // Apply continuous logit-space offsets from the 3 Macro Controls (neutral at 0.5)
        const float dSize  = (detail::clamp01(macros.spectralTileSize)     - 0.5f) * 2.0f; // [-1, +1]
        const float dOrder = (detail::clamp01(macros.ruleOrderVsMutation)  - 0.5f) * 2.0f; // [-1, +1]
        const float dGlue  = (detail::clamp01(macros.phaseAndSpectralGlue) - 0.5f) * 2.0f; // [-1, +1]

        GenomeVec g = baseGenome;
        auto modulateGene = [](float geneVal, float logitDelta) noexcept -> float {
            return detail::sigmoid(detail::logit(geneVal) + logitDelta);
        };

        // --- Macro 1: Spectral Tile Size (Delta_t x Delta_f) ---
        // dSize < 0: micro FFT shards, high Voronoi seed count, fast scale
        // dSize > 0: wide formant bands, smooth Simplex regions, high frequency persistence
        g[static_cast<std::size_t>(Gene::TileScaleTime)] =
            modulateGene(g[static_cast<std::size_t>(Gene::TileScaleTime)], +2.2f * dSize);
        g[static_cast<std::size_t>(Gene::TileScaleFreq)] =
            modulateGene(g[static_cast<std::size_t>(Gene::TileScaleFreq)], +2.2f * dSize);
        g[static_cast<std::size_t>(Gene::MarkovStayFreq)] =
            modulateGene(g[static_cast<std::size_t>(Gene::MarkovStayFreq)], +1.4f * dSize);

        // --- Macro 2: Rule Order vs. Mutation ---
        // dOrder < 0: deterministic descriptor competition & harmonic comb order, high Markov persistence
        // dOrder > 0: stochastic Voronoi/Simplex mutation, fast LFO drift, stronger Cross-Synthesis
        g[static_cast<std::size_t>(Gene::AlphaGeo)] =
            modulateGene(g[static_cast<std::size_t>(Gene::AlphaGeo)], +1.6f * dOrder);
        g[static_cast<std::size_t>(Gene::AlphaDesc)] =
            modulateGene(g[static_cast<std::size_t>(Gene::AlphaDesc)], -1.2f * dOrder);
        g[static_cast<std::size_t>(Gene::VoronoiMix)] =
            modulateGene(g[static_cast<std::size_t>(Gene::VoronoiMix)], +1.5f * dOrder);
        g[static_cast<std::size_t>(Gene::CombMix)] =
            modulateGene(g[static_cast<std::size_t>(Gene::CombMix)], -1.3f * dOrder);
        g[static_cast<std::size_t>(Gene::GeoDriftAndSharpness)] =
            modulateGene(g[static_cast<std::size_t>(Gene::GeoDriftAndSharpness)], +1.8f * dOrder);
        g[static_cast<std::size_t>(Gene::CrossSynthesisChi)] =
            modulateGene(g[static_cast<std::size_t>(Gene::CrossSynthesisChi)], +1.1f * dOrder);

        // --- Macro 3: Phase & Spectral Glue ---
        // dGlue < 0: raw crisp tile edges, low Softmax tau, minimal feathering, Identity Phase-Locking
        // dGlue > 0: wide 2D Gaussian feathering, high MS-PGHI Phase Glue, high temporal Markov stay
        g[static_cast<std::size_t>(Gene::PhaseGlueBeta)] =
            modulateGene(g[static_cast<std::size_t>(Gene::PhaseGlueBeta)], +2.4f * dGlue);
        g[static_cast<std::size_t>(Gene::Feathering2DSigma)] =
            modulateGene(g[static_cast<std::size_t>(Gene::Feathering2DSigma)], +2.2f * dGlue);
        g[static_cast<std::size_t>(Gene::SoftmaxTemperature)] =
            modulateGene(g[static_cast<std::size_t>(Gene::SoftmaxTemperature)], +1.3f * dGlue);
        g[static_cast<std::size_t>(Gene::MarkovStayTime)] =
            modulateGene(g[static_cast<std::size_t>(Gene::MarkovStayTime)], +1.5f * dGlue);

        enforceAntiMudInvariants(g);

        // ====================================================================
        // Decode normalized genes g[0..23] into physical DSP/Rule units
        // ====================================================================
        DecodedGenomeParams p{};

        // --- Chromosome I ---
        p.alphaGeo        = 4.0f * g[static_cast<std::size_t>(Gene::AlphaGeo)];
        p.alphaDesc       = 4.0f * g[static_cast<std::size_t>(Gene::AlphaDesc)];
        p.alphaMark       = 3.5f * g[static_cast<std::size_t>(Gene::AlphaMark)];
        // Exponential mapping for Softmax temperature tau in [0.03, 1.50]
        p.softmaxTempTau  = 0.03f * std::pow(1.50f / 0.03f, g[static_cast<std::size_t>(Gene::SoftmaxTemperature)]);
        p.sparsityGateEta = 0.85f * g[static_cast<std::size_t>(Gene::SparsityGate)];

        // --- Chromosome II ---
        const float rawVor  = g[static_cast<std::size_t>(Gene::VoronoiMix)];
        const float rawSimp = g[static_cast<std::size_t>(Gene::SimplexMix)];
        const float rawComb = g[static_cast<std::size_t>(Gene::CombMix)];
        const float geoSum  = std::max(rawVor + rawSimp + rawComb, 1.0e-4f);
        p.betaVoronoi = rawVor  / geoSum;
        p.betaSimplex = rawSimp / geoSum;
        p.betaComb    = rawComb / geoSum;

        const float gScaleT = g[static_cast<std::size_t>(Gene::TileScaleTime)];
        const float gScaleF = g[static_cast<std::size_t>(Gene::TileScaleFreq)];
        p.tileScaleTimeSec  = 0.010f * std::pow(120.0f, gScaleT); // [10 ms .. 1200 ms]
        p.tileScaleFreqErb  = 0.5f + 15.5f * (gScaleF * gScaleF); // [0.5 ERB .. 16.0 ERB]

        // Voronoi active seed count Q in [N, 128] inversely related to 2D tile area
        const float areaNorm = 0.5f * (gScaleT + gScaleF);
        const float qFloat   = detail::lerp(128.0f, static_cast<float>(N), areaNorm);
        p.voronoiSeeds       = std::clamp<std::size_t>(static_cast<std::size_t>(std::lround(qFloat)), N, 128);
        p.simplexOctaves     = std::clamp<std::size_t>(1 + static_cast<std::size_t>((1.0f - areaNorm) * 3.9f), 1, 4);

        const float gDriftSharp = g[static_cast<std::size_t>(Gene::GeoDriftAndSharpness)];
        p.lfoDriftHz       = 0.02f * std::pow(200.0f, gDriftSharp); // [0.02 Hz .. 4.0 Hz]
        p.combKappa        = 0.5f + 11.5f * gDriftSharp;            // [0.5 .. 12.0]
        p.combHarmonicMode = (rawComb > 0.55f);

        // --- Chromosome III ---
        p.gammaSplit         = 3.5f * g[static_cast<std::size_t>(Gene::OnsetSplitWeight)];
        p.onsetFluxThreshold = detail::lerp(0.08f, 0.88f, g[static_cast<std::size_t>(Gene::OnsetFluxThreshold)]);
        p.onsetFluxSlope     = detail::lerp(22.0f, 6.0f,  g[static_cast<std::size_t>(Gene::OnsetFluxThreshold)]);
        p.gammaFlatness      = detail::lerp(-2.5f, +2.5f, g[static_cast<std::size_t>(Gene::FlatnessCompWeight)]);
        p.gammaComplement    = 3.5f * g[static_cast<std::size_t>(Gene::HoleFillerWeight)];
        p.lambdaAntiMask     = 1.5f * g[static_cast<std::size_t>(Gene::AntiMaskingLambda)];

        // --- Chromosome IV ---
        p.markovStayTimeProb = detail::lerp(0.15f, 0.995f, g[static_cast<std::size_t>(Gene::MarkovStayTime)]);
        p.markovStayFreqProb = detail::lerp(0.15f, 0.990f, g[static_cast<std::size_t>(Gene::MarkovStayFreq)]);
        p.markovKappaTime    = std::log(p.markovStayTimeProb / (1.0f - p.markovStayTimeProb));
        p.markovKappaFreq    = std::log(p.markovStayFreqProb / (1.0f - p.markovStayFreqProb));
        p.transientResetEta  = g[static_cast<std::size_t>(Gene::TransientResetEta)];

        // --- Chromosome V ---
        p.crossSynthChi      = g[static_cast<std::size_t>(Gene::CrossSynthesisChi)];
        p.crossShiftDelta    = std::clamp<std::size_t>(
            1 + static_cast<std::size_t>(p.crossSynthChi * static_cast<float>(N - 1)), 1, N - 1);
        p.envelopeGamma      = detail::lerp(0.25f, 1.45f, g[static_cast<std::size_t>(Gene::EnvelopeExponent)]);
        p.phaseGlueBeta      = g[static_cast<std::size_t>(Gene::PhaseGlueBeta)];

        const float gFeather = g[static_cast<std::size_t>(Gene::Feathering2DSigma)];
        p.featherTimeAlpha   = 0.90f * (gFeather * gFeather);
        p.featherFreqSigma   = 12.0f * gFeather;

        const float gSubLock = g[static_cast<std::size_t>(Gene::SubBassLockStrength)];
        p.subBassLockLambda  = 12.0f * gSubLock;
        p.subBassCutoffHz    = 150.0f;
        p.subBassOwner       = 0;

        return p;
    }

    // --- Linear Algebra Helpers in R^24 ---
    [[nodiscard]] static constexpr float dotProduct(const TangentVec& a, const TangentVec& b) noexcept {
        float sum = 0.0f;
        for (std::size_t d = 0; d < kGenomeDim; ++d) {
            sum += a[d] * b[d];
        }
        return sum;
    }

    [[nodiscard]] static inline float vectorNorm(const TangentVec& v) noexcept {
        return std::sqrt(dotProduct(v, v));
    }
};

} // namespace spectile::rules
