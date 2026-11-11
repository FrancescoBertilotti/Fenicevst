#pragma once

/**
 * @file GeometricTiler.hpp
 * @brief Settore 2 — Family 1 (Phi^Geo_i): 2D Laguerre Voronoi Tiling via O(B)
 *        Parabolic Lower Envelope (Felzenszwalb & Huttenlocher), Multi-Octave
 *        2D Simplex fBm Noise, and von Mises Harmonic/ERB Comb Interleaving.
 *
 * @details
 * Implements Section 2.1 (Family 1) of the SpecTile Executive Architecture Plan:
 *  1. Weighted 2D Voronoi (Laguerre / Power Diagram) in the normalized
 *     time-frequency plane (t_tilde, f_tilde) with Q in [N, 128] seeds initialized
 *     via low-discrepancy 2D Halton sequences and animated by 2D LFO drift.
 *     At each STFT hop t, the 2D distance reduces to a 1D Lower Envelope of
 *     Parabolas along the frequency axis, solved in exact O(B + Q) linear time
 *     using Felzenszwalb & Huttenlocher's parabola intersection stack.
 *  2. Multi-Octave 2D Simplex Noise (fBm, 1..4 octaves) with golden-ratio
 *     spatial decorrelation across the N sample slots.
 *  3. Spectral Interleaving (von Mises Comb) for cyclic ERB bands or odd/even
 *     harmonic partials:
 *       Phi^Comb_i(t, b) = exp( kappa_comb * cos( 2*pi/N * (u(t, b) - i) ) ) - 1
 *  4. Per-band zero-mean, unit-variance normalization across active samples i.
 *
 * Zero heap allocations, SIMD-friendly alignas(64) SoA buffers, C++20 noexcept.
 */

#include "SpectralGenome.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>

namespace spectile::rules {

// ============================================================================
// Voronoi Seed State & Halton Low-Discrepancy Generator
// ============================================================================

inline constexpr std::size_t kMaxVoronoiSeeds = 128;

/**
 * @brief A single 2D Laguerre Voronoi seed in the normalized time-frequency plane.
 */
struct VoronoiSeed2D {
    float       baseTime     = 0.5f;  ///< Halton base-2 initial time coordinate in [0, 1)
    float       baseFreqNorm = 0.5f;  ///< Halton base-3 initial normalized frequency in [0, 1)
    float       lfoPhase     = 0.0f;  ///< Halton base-5 LFO phase offset in [0, 2*pi)
    float       driftVelRatio= 1.0f;  ///< Halton base-7 velocity multiplier in [0.5, 1.5]
    float       laguerreWeight = 0.0f;///< Power diagram weight w_q (attracts larger cell when > 0)
    std::size_t sampleIndex  = 0;     ///< Owner sample slot index i in {0 .. N-1}
};

/**
 * @brief Output of Family 1 (Geometric Tiling) over B psychoacoustic ERB bands
 *        for up to N = 8 sample slots.
 */
struct alignas(64) GeometricTilingOutput {
    /// Normalized zero-mean, unit-variance geometric logits bar{Phi}^Geo_i(t, b)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> normalizedLogits{};
    /// Raw unnormalized geometric potentials Phi^Geo_i(t, b)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> rawPotentials{};
    /// Dominant Voronoi owner sample index per band (for UI / diagnostics)
    std::array<uint8_t, kMaxBands> dominantVoronoiSample{};
};

// ============================================================================
// Self-Contained 2D Simplex Noise & fBm Engine
// ============================================================================

class SimplexNoise2D {
public:
    /**
     * @brief Evaluates 2D Simplex Noise in [-1, +1] at coordinates (x, y).
     *        Based on Ken Perlin's Simplex grid with Stefan Gustavson's radial attenuation.
     */
    [[nodiscard]] static inline float evaluate(float x, float y) noexcept {
        constexpr float F2 = 0.3660254037844386f; // 0.5 * (sqrt(3.0) - 1.0)
        constexpr float G2 = 0.2113248654051871f; // (3.0 - sqrt(3.0)) / 6.0

        // Skew input space to determine which simplex cell (i, j) we are in
        const float s  = (x + y) * F2;
        const float xs = x + s;
        const float ys = y + s;
        const int   i  = fastFloor(xs);
        const int   j  = fastFloor(ys);

        const float t  = static_cast<float>(i + j) * G2;
        const float X0 = static_cast<float>(i) - t;
        const float Y0 = static_cast<float>(j) - t;
        const float x0 = x - X0; // Unskewed distances from cell origin
        const float y0 = y - Y0;

        // Determine which simplex triangle (equilateral 2D) we are in
        const int i1 = (x0 > y0) ? 1 : 0;
        const int j1 = (x0 > y0) ? 0 : 1;

        const float x1 = x0 - static_cast<float>(i1) + G2;
        const float y1 = y0 - static_cast<float>(j1) + G2;
        const float x2 = x0 - 1.0f + 2.0f * G2;
        const float y2 = y0 - 1.0f + 2.0f * G2;

        // Hash coordinates of the 3 simplex corners
        const uint8_t ii = static_cast<uint8_t>(i & 255);
        const uint8_t jj = static_cast<uint8_t>(j & 255);
        const uint8_t gi0 = permHash(static_cast<uint8_t>(ii + permHash(jj))) % 12;
        const uint8_t gi1 = permHash(static_cast<uint8_t>(ii + i1 + permHash(static_cast<uint8_t>(jj + j1)))) % 12;
        const uint8_t gi2 = permHash(static_cast<uint8_t>(ii + 1 + permHash(static_cast<uint8_t>(jj + 1)))) % 12;

        // Radial attenuation from each of the 3 corners
        float n0 = 0.0f;
        float t0 = 0.5f - x0 * x0 - y0 * y0;
        if (t0 > 0.0f) {
            t0 *= t0;
            n0 = t0 * t0 * gradDot(gi0, x0, y0);
        }

        float n1 = 0.0f;
        float t1 = 0.5f - x1 * x1 - y1 * y1;
        if (t1 > 0.0f) {
            t1 *= t1;
            n1 = t1 * t1 * gradDot(gi1, x1, y1);
        }

        float n2 = 0.0f;
        float t2 = 0.5f - x2 * x2 - y2 * y2;
        if (t2 > 0.0f) {
            t2 *= t2;
            n2 = t2 * t2 * gradDot(gi2, x2, y2);
        }

        // Scale to [-1, +1]
        return 70.0f * (n0 + n1 + n2);
    }

    /**
     * @brief Evaluates Multi-Octave Fractional Brownian Motion (fBm) 2D Simplex Noise.
     *
     * @param x          Base X coordinate (time + LFO drift)
     * @param y          Base Y coordinate (scaled ERB frequency)
     * @param numOctaves Number of octaves in [1, 4]
     * @param lacunarity Frequency multiplier per octave (default 2.03f)
     * @param gain       Amplitude persistence per octave (default 0.5f)
     */
    [[nodiscard]] static inline float fbm(
        float x,
        float y,
        std::size_t numOctaves = 3,
        float lacunarity = 2.03f,
        float gain = 0.5f) noexcept
    {
        const std::size_t octaves = std::clamp<std::size_t>(numOctaves, 1, 4);
        float total    = 0.0f;
        float amp      = 1.0f;
        float freq     = 1.0f;
        float normSum  = 0.0f;

        for (std::size_t o = 0; o < octaves; ++o) {
            // Rotate octave coordinates slightly (cos(0.5), sin(0.5)) to eliminate axial artifacts
            const float rx = (0.87758256f * x - 0.47942554f * y) * freq;
            const float ry = (0.47942554f * x + 0.87758256f * y) * freq;
            total   += amp * evaluate(rx + static_cast<float>(o) * 17.3f, ry - static_cast<float>(o) * 11.7f);
            normSum += amp;
            amp     *= gain;
            freq    *= lacunarity;
        }

        return total / std::max(normSum, 1.0e-6f);
    }

private:
    [[nodiscard]] static constexpr int fastFloor(float x) noexcept {
        const int xi = static_cast<int>(x);
        return (x < static_cast<float>(xi)) ? (xi - 1) : xi;
    }

    /// Deterministic bijective hash replacing a 256-entry lookup table in L1 registers
    [[nodiscard]] static constexpr uint8_t permHash(uint8_t x) noexcept {
        uint16_t h = static_cast<uint16_t>(x);
        h ^= (h >> 4);
        h = static_cast<uint16_t>((h * 0x9E37U) & 0xFFU);
        h ^= (h >> 3);
        return static_cast<uint8_t>((h * 0x5DU) & 0xFFU);
    }

    [[nodiscard]] static constexpr float gradDot(uint8_t hashIdx, float x, float y) noexcept {
        constexpr std::array<std::array<float, 2>, 12> kGrad2 = {{
            { 1.0f,  1.0f}, {-1.0f,  1.0f}, { 1.0f, -1.0f}, {-1.0f, -1.0f},
            { 1.0f,  0.0f}, {-1.0f,  0.0f}, { 1.0f,  0.0f}, {-1.0f,  0.0f},
            { 0.0f,  1.0f}, { 0.0f, -1.0f}, { 0.0f,  1.0f}, { 0.0f, -1.0f}
        }};
        return kGrad2[hashIdx][0] * x + kGrad2[hashIdx][1] * y;
    }
};

// ============================================================================
// GeometricTiler — Family 1 Engine (Voronoi O(B) + Simplex fBm + von Mises Comb)
// ============================================================================

class GeometricTiler {
public:
    GeometricTiler() noexcept {
        initializeHaltonSeeds(kMaxSamples, 0);
    }

    /**
     * @brief Generates Q <= 128 quasi-random low-discrepancy 2D Halton seeds
     *        in the normalized time-frequency plane [0, 1)^2.
     *
     * @param numActiveSamples Number of active sample slots N in [2, 8]
     * @param seedOffset       Starting index offset in the Halton sequence (for seed mutation)
     * @param slotAttraction   Optional per-slot Laguerre weight bias in [-1, +1]
     */
    void initializeHaltonSeeds(
        std::size_t numActiveSamples,
        uint32_t seedOffset = 0,
        const std::array<float, kMaxSamples>& slotAttraction = {}) noexcept
    {
        const std::size_t N = std::clamp<std::size_t>(numActiveSamples, 1, kMaxSamples);
        for (std::size_t q = 0; q < kMaxVoronoiSeeds; ++q) {
            const uint32_t idx = static_cast<uint32_t>(q + 1) + seedOffset * 131U;
            VoronoiSeed2D& seed = seeds_[q];
            seed.baseTime      = radicalInverse(idx, 2U); // Base-2 Halton for time
            seed.baseFreqNorm  = radicalInverse(idx, 3U); // Base-3 Halton for frequency
            seed.lfoPhase      = radicalInverse(idx, 5U) * detail::kTwoPi; // Base-5 for LFO phase
            seed.driftVelRatio = 0.5f + radicalInverse(idx, 7U); // Base-7 for drift velocity
            seed.sampleIndex   = q % N;
            seed.laguerreWeight= slotAttraction[seed.sampleIndex];
        }
    }

    /**
     * @brief Updates the Laguerre power weights w_q for each sample slot (Quadrant 2 UI control).
     */
    void setSlotLaguerreWeights(
        const std::array<float, kMaxSamples>& slotWeights,
        std::size_t numActiveSamples) noexcept
    {
        const std::size_t N = std::clamp<std::size_t>(numActiveSamples, 1, kMaxSamples);
        for (std::size_t q = 0; q < kMaxVoronoiSeeds; ++q) {
            seeds_[q].sampleIndex    = q % N;
            seeds_[q].laguerreWeight = slotWeights[seeds_[q].sampleIndex];
        }
    }

    /**
     * @brief Evaluates Family 1 (Geometric Tiling) over B ERB bands for the current STFT frame t.
     *
     *        Phi^Geo_i(t, b) = beta_Vor * Phi^Vor_i(t, b)
     *                        + beta_Perlin * Phi^Perlin_i(t, b)
     *                        + beta_Comb * Phi^Comb_i(t, b)
     *
     * @param params           Decoded genome & macro parameters
     * @param timeSeconds      Current continuous playback/STFT time in seconds
     * @param numActiveSamples Number of active sample slots N in [2, 8]
     * @param numBands         Number of psychoacoustic ERB bands B in [16, 128] (default 64)
     * @param erbCenterFreqHz  Optional center frequencies of each ERB band in Hz (used in harmonic comb mode)
     * @param fundamentalF0Hz  Fundamental pitch f_0(t) in Hz (used in harmonic odd/even comb mode)
     * @param out              Destination aligned SoA output structure
     */
    void evaluateFrame(
        const DecodedGenomeParams& params,
        float timeSeconds,
        std::size_t numActiveSamples,
        std::size_t numBands,
        const std::array<float, kMaxBands>& erbCenterFreqHz,
        float fundamentalF0Hz,
        GeometricTilingOutput& out) const noexcept
    {
        const std::size_t N = std::clamp<std::size_t>(numActiveSamples, 1, kMaxSamples);
        const std::size_t B = std::clamp<std::size_t>(numBands, 1, kMaxBands);
        const std::size_t Q = std::clamp<std::size_t>(params.voronoiSeeds, N, kMaxVoronoiSeeds);

        // ====================================================================
        // 1. Weighted 2D Voronoi (Laguerre) via O(B) 1D Parabolic Lower Envelope
        //    (Felzenszwalb & Huttenlocher Algorithm)
        // ====================================================================
        std::array<std::array<float, kMaxBands>, kMaxSamples> voronoiPotentials{};
        if (params.betaVoronoi > 1.0e-4f) {
            evaluateVoronoiParabolicEnvelope(params, timeSeconds, N, B, Q, voronoiPotentials, out.dominantVoronoiSample);
        }

        // ====================================================================
        // 2. Multi-Octave 2D Simplex Noise Fields (fBm)
        // ====================================================================
        std::array<std::array<float, kMaxBands>, kMaxSamples> simplexPotentials{};
        if (params.betaSimplex > 1.0e-4f) {
            const float invScaleT = 1.0f / std::max(params.tileScaleTimeSec * 2.5f, 0.02f);
            const float invScaleF = 1.0f / std::max(params.tileScaleFreqErb, 0.5f);
            const float tCoord    = timeSeconds * (invScaleT + 0.35f * params.lfoDriftHz);

            for (std::size_t i = 0; i < N; ++i) {
                // Golden-ratio spatial decorrelation offsets per sample slot i
                const float offsetX = static_cast<float>(i) * 19.1973f;
                const float offsetY = static_cast<float>(i) * 31.4159f;

                for (std::size_t b = 0; b < B; ++b) {
                    const float fCoord = static_cast<float>(b) * invScaleF;
                    simplexPotentials[i][b] = SimplexNoise2D::fbm(
                        tCoord + offsetX,
                        fCoord + offsetY,
                        params.simplexOctaves,
                        2.03f,
                        0.5f);
                }
            }
        }

        // ====================================================================
        // 3. Spectral Interleaving (von Mises Harmonic / ERB Comb)
        //    Phi^Comb_i(t, b) = exp( kappa_comb * cos( 2*pi/N * (u(t,b) - i) ) ) - 1
        // ====================================================================
        std::array<std::array<float, kMaxBands>, kMaxSamples> combPotentials{};
        if (params.betaComb > 1.0e-4f) {
            const float invN      = 1.0f / static_cast<float>(N);
            const float twoPiOverN= detail::kTwoPi * invN;
            const float kappa     = std::clamp(params.combKappa, 0.2f, 12.0f);
            // Norm factor to keep exp(kappa * cos) well-conditioned across kappa values
            const float normScale = 1.0f / std::max(std::exp(kappa) - 1.0f, 1.0e-4f);
            const float lfoPhase  = 0.25f * std::sin(detail::kTwoPi * 0.2f * params.lfoDriftHz * timeSeconds);
            const float safeF0    = std::max(fundamentalF0Hz, 20.0f);
            const float bandStep  = std::max(params.tileScaleFreqErb * 0.5f, 1.0f);

            for (std::size_t b = 0; b < B; ++b) {
                // Coordinate u(t, b): harmonic partial index f(b)/f0 or cyclic ERB band index
                const float u = params.combHarmonicMode
                    ? (erbCenterFreqHz[b] / safeF0 + lfoPhase)
                    : (static_cast<float>(b) / bandStep + lfoPhase);

                for (std::size_t i = 0; i < N; ++i) {
                    const float angle = twoPiOverN * (u - static_cast<float>(i));
                    const float vm    = (std::exp(kappa * std::cos(angle)) - 1.0f) * normScale;
                    combPotentials[i][b] = 2.0f * vm - 1.0f; // Center in [-1, +1]
                }
            }
        }

        // ====================================================================
        // 4. Combine Sub-Rules & Normalize per Band to Zero-Mean, Unit-Variance
        // ====================================================================
        for (std::size_t b = 0; b < B; ++b) {
            float mean = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                const float phi = params.betaVoronoi * voronoiPotentials[i][b]
                                + params.betaSimplex * simplexPotentials[i][b]
                                + params.betaComb    * combPotentials[i][b];
                out.rawPotentials[i][b] = phi;
                mean += phi;
            }
            mean /= static_cast<float>(N);

            float var = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                const float diff = out.rawPotentials[i][b] - mean;
                var += diff * diff;
            }
            const float invStd = 1.0f / std::sqrt(var / static_cast<float>(N) + 1.0e-5f);

            for (std::size_t i = 0; i < N; ++i) {
                out.normalizedLogits[i][b] = (out.rawPotentials[i][b] - mean) * invStd;
            }
            for (std::size_t i = N; i < kMaxSamples; ++i) {
                out.rawPotentials[i][b]    = 0.0f;
                out.normalizedLogits[i][b] = 0.0f;
            }
        }
    }

    /**
     * @brief Radical inverse function phi_base(n) for Halton low-discrepancy sequence generation.
     */
    [[nodiscard]] static constexpr float radicalInverse(uint32_t n, uint32_t base) noexcept {
        float val     = 0.0f;
        float invBase = 1.0f / static_cast<float>(base);
        float factor  = invBase;
        while (n > 0U) {
            const uint32_t digit = n % base;
            val    += static_cast<float>(digit) * factor;
            n      /= base;
            factor *= invBase;
        }
        return val;
    }

private:
    std::array<VoronoiSeed2D, kMaxVoronoiSeeds> seeds_{};

    /**
     * @brief Exact O(B + Q) 1D Parabolic Lower Envelope algorithm (Felzenszwalb & Huttenlocher, 2012)
     *        computing the 2D Laguerre Voronoi distance transform for each sample i across B bands.
     *
     * At fixed frame time t, each seed q at (t_q(t), f_q(t)) belonging to sample i defines a
     * 1D parabola along the ERB band axis b in [0, B-1]:
     *
     *   D_q(b) = ( (b - f_q(t)) / sigma_f )^2 + c_q(t)
     *
     * where the temporal squared distance + Laguerre weight is a constant vertical offset:
     *
     *   c_q(t) = ( d_torus(t_norm, t_q(t)) / sigma_t_norm )^2 - w_q
     *
     * Using Felzenszwalb & Huttenlocher's stack of parabola vertices v[] and intersection
     * boundaries z[], the exact lower envelope min_{q in Q_i} D_q(b) is evaluated in O(B) time.
     */
    void evaluateVoronoiParabolicEnvelope(
        const DecodedGenomeParams& params,
        float timeSeconds,
        std::size_t N,
        std::size_t B,
        std::size_t Q,
        std::array<std::array<float, kMaxBands>, kMaxSamples>& voronoiPotentials,
        std::array<uint8_t, kMaxBands>& dominantSampleOut) const noexcept
    {
        constexpr float kInf = 1.0e12f;

        const float sigmaF    = std::max(params.tileScaleFreqErb, 0.5f);
        const float sigmaFSq  = sigmaF * sigmaF;
        const float invSigmaFSq = 1.0f / sigmaFSq;

        // Normalize time coordinate onto a periodic 4-second torus with LFO drift
        constexpr float kTimePeriodSec = 4.0f;
        const float sigmaTNorm   = std::clamp(params.tileScaleTimeSec / kTimePeriodSec, 0.005f, 1.0f);
        const float invSigmaTSq  = 1.0f / (sigmaTNorm * sigmaTNorm);
        const float currentTNorm = std::fmod(std::max(timeSeconds, 0.0f) / kTimePeriodSec, 1.0f);
        const float omegaLfo     = detail::kTwoPi * params.lfoDriftHz;
        const float maxBandFloat = static_cast<float>(B - 1);

        // Per-sample 1D grid of minimum vertical offsets tilde_f[i][q_band] = sigma_f^2 * c_q(t)
        // and exact sub-band seed abscissa x_pos[i][q_band]
        std::array<std::array<float, kMaxBands>, kMaxSamples> gridOffsets{};
        std::array<std::array<float, kMaxBands>, kMaxSamples> gridAbscissa{};

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t b = 0; b < B; ++b) {
                gridOffsets[i][b]  = kInf;
                gridAbscissa[i][b] = static_cast<float>(b);
            }
        }

        // Step 1: Project the Q active seeds onto the 1D ERB band buckets in O(Q) time
        for (std::size_t q = 0; q < Q; ++q) {
            const VoronoiSeed2D& seed = seeds_[q];
            const std::size_t sampleIdx = seed.sampleIndex % N;

            // Compute instantaneous drifted time coordinate in [0, 1)
            const float lfoArg = omegaLfo * seed.driftVelRatio * timeSeconds + seed.lfoPhase;
            float seedT = seed.baseTime + 0.12f * seed.driftVelRatio * (timeSeconds / kTimePeriodSec)
                        + 0.08f * std::sin(lfoArg);
            seedT -= std::floor(seedT);

            // Shortest toroidal distance on [0, 1)
            float dt = std::abs(currentTNorm - seedT);
            if (dt > 0.5f) {
                dt = 1.0f - dt;
            }

            // Compute instantaneous drifted frequency coordinate in [0, B-1] with reflection at boundaries
            float freqNorm = seed.baseFreqNorm + 0.15f * std::cos(0.73f * lfoArg + 1.618f * seed.lfoPhase);
            // Triangle wave reflection into [0, 1]
            freqNorm = std::fmod(std::abs(freqNorm), 2.0f);
            if (freqNorm > 1.0f) {
                freqNorm = 2.0f - freqNorm;
            }
            const float fExact = freqNorm * maxBandFloat;
            const auto  bBucket = std::clamp<std::size_t>(
                static_cast<std::size_t>(std::lround(fExact)), 0, B - 1);

            // Constant vertical parabola offset c_q(t) scaled by sigma_f^2
            const float temporalCost = (dt * dt) * invSigmaTSq - seed.laguerreWeight;
            const float scaledOffset = temporalCost * sigmaFSq;

            if (scaledOffset < gridOffsets[sampleIdx][bBucket]) {
                gridOffsets[sampleIdx][bBucket]  = scaledOffset;
                gridAbscissa[sampleIdx][bBucket] = fExact;
            }
        }

        // Ensure every active sample has at least one finite parabola so its envelope is always well-defined
        for (std::size_t i = 0; i < N; ++i) {
            bool hasSeed = false;
            for (std::size_t b = 0; b < B; ++b) {
                if (gridOffsets[i][b] < kInf * 0.5f) {
                    hasSeed = true;
                    break;
                }
            }
            if (!hasSeed) {
                const std::size_t fallbackBand = ((i * B) / N) % B;
                gridOffsets[i][fallbackBand]  = 0.0f;
                gridAbscissa[i][fallbackBand] = static_cast<float>(fallbackBand);
            }
        }

        // Step 2: Felzenszwalb & Huttenlocher O(B) 1D Parabolic Lower Envelope per sample i
        std::array<std::size_t, kMaxBands>     v{}; // Stack of parabola bucket indices
        std::array<float,       kMaxBands + 1> z{}; // Stack of intersection boundaries

        for (std::size_t i = 0; i < N; ++i) {
            int k = -1; // Stack pointer (-1 = empty)

            // Pass 2a: Build lower envelope intersection stack left-to-right (q = 0 .. B-1)
            for (std::size_t q = 0; q < B; ++q) {
                const float fq = gridOffsets[i][q];
                if (fq >= kInf * 0.5f) {
                    continue; // Empty bucket: skip in O(1)
                }
                const float xq = gridAbscissa[i][q];

                if (k < 0) {
                    k    = 0;
                    v[0] = q;
                    z[0] = -kInf;
                    z[1] = +kInf;
                    continue;
                }

                // Compute intersection s between parabola at v[k] and parabola at q
                float s = 0.0f;
                while (k >= 0) {
                    const std::size_t p  = v[static_cast<std::size_t>(k)];
                    const float       fp = gridOffsets[i][p];
                    const float       xp = gridAbscissa[i][p];
                    const float       dx = std::max(xq - xp, 1.0e-4f);
                    s = ((fq + xq * xq) - (fp + xp * xp)) / (2.0f * dx);

                    if (s > z[static_cast<std::size_t>(k)]) {
                        break;
                    }
                    --k;
                }

                ++k;
                v[static_cast<std::size_t>(k)]     = q;
                z[static_cast<std::size_t>(k)]     = s;
                z[static_cast<std::size_t>(k + 1)] = +kInf;
            }

            // Pass 2b: Evaluate lower envelope left-to-right across bands b = 0 .. B-1 in O(B)
            std::size_t envIdx = 0;
            for (std::size_t b = 0; b < B; ++b) {
                const float bFloat = static_cast<float>(b);
                while (envIdx < static_cast<std::size_t>(k) && z[envIdx + 1] < bFloat) {
                    ++envIdx;
                }
                const std::size_t bestBucket = v[envIdx];
                const float       dx         = bFloat - gridAbscissa[i][bestBucket];
                const float       distSq     = (dx * dx + gridOffsets[i][bestBucket]) * invSigmaFSq;

                // Convert minimum squared Laguerre distance into smooth localized potential in [-1, +1]
                voronoiPotentials[i][b] = 2.0f * std::exp(-0.5f * std::max(distSq, 0.0f)) - 1.0f;
            }
        }

        // Record dominant Voronoi owner sample per band for UI visualization / diagnostics
        for (std::size_t b = 0; b < B; ++b) {
            float   bestVal = -kInf;
            uint8_t bestIdx = 0;
            for (std::size_t i = 0; i < N; ++i) {
                if (voronoiPotentials[i][b] > bestVal) {
                    bestVal = voronoiPotentials[i][b];
                    bestIdx = static_cast<uint8_t>(i);
                }
            }
            dominantSampleOut[b] = bestIdx;
        }
    }
};

} // namespace spectile::rules
