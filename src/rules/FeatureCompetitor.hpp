#pragma once

/**
 * @file FeatureCompetitor.hpp
 * @brief Settore 2 — Family 2 (Phi^Desc_i): Spectromorphological Onset/Continuant Split,
 *        Tonal vs. Noisy Spectral Flatness Competition, and Energy Complementarity /
 *        Anti-Masking (Hole Filler).
 *
 * @details
 * Implements Section 2.1 (Family 2) of the SpecTile Executive Architecture Plan:
 *
 *   Phi^Desc_i(t, b) = gamma_split * Phi^Split_i(t, b)
 *                    + gamma_flat  * Phi^Flat_i(t, b)
 *                    + gamma_comp  * Phi^Comp_i(t, b)
 *
 *  1. Spectromorphological Onset/Continuant Split (Smalley, 1997):
 *     A smooth sigmoid gate Omega_onset(t, b) driven by local Spectral Flux
 *     assigns transient attack regions (Omega_onset -> 1) to samples with attack
 *     role rho^atk_i > 0 and stationary regions (Omega_onset -> 0) to harmonic
 *     sustain samples with rho^sus_i > 0.
 *  2. Tonal vs. Noisy Competition:
 *     Driven by local Spectral Flatness (Wiener entropy S_flat in [0, 1]),
 *     promoting either pure sinusoidal partials (S_flat -> 0) or stochastic
 *     residual textures (S_flat -> 1).
 *  3. Energy Complementarity / Anti-Masking (Hole Filler):
 *       Phi^Comp_i(t, b) = hat{E}_i(t, b) * (1 - lambda_anti * O_{-i}(t, b))
 *                        - lambda_anti * O_{-i}(t, b) * hat{E}_{-i}(t, b)
 *     For lambda_anti > 0, secondary samples automatically fill the spectral
 *     valleys left empty by dominant samples (O_{-i}(t, b) ~ 0) without ever
 *     masking their primary peaks.
 *
 * All buffers are 64-byte aligned (SoA), zero-allocation, and C++20 noexcept.
 */

#include "SpectralGenome.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace spectile::rules {

// ============================================================================
// Input ERB Band Descriptors & Per-Slot Role Allocation Matrix (Quadrant 2 UI)
// ============================================================================

/**
 * @brief Pre-analyzed or real-time ERB Band Descriptors for a single frame t
 *        across all N sample slots and B psychoacoustic bands (Settore 1 -> Settore 2).
 */
struct alignas(64) FrameErbDescriptors {
    /// Normalized band energy hat{E}_i(t, b) in [0, 1] (perceptually compressed / dB-normalized)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> normalizedEnergy{};
    /// Local Spectral Flux tilde{F}_i(t, b) in [0, 1] (half-wave rectified log-magnitude derivative)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> spectralFlux{};
    /// Local Spectral Flatness (Wiener entropy) S_flat,i(t, b) in [0, 1] (0 = pure tone, 1 = white noise)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> spectralFlatness{};
    /// Optional Harmonic share H_i / |S_i| in [0, 1] from 3-way HPR decomposition
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> harmonicRatio{};
    /// Optional Percussive share P_i / |S_i| in [0, 1] from 3-way HPR decomposition
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> percussiveRatio{};
};

/**
 * @brief Per-Slot Spectromorphological Role Allocation Matrix (Quadrant 2 of Deep Spectral Matrix).
 *        Allows either automatic role assignment from HPR descriptors or surgical user overrides.
 */
struct SampleRoleAllocation {
    /// Attack/Onset role weight rho^atk_i in [-1, +1] (>0 = prefers attacks, <0 = ducks on attacks)
    std::array<float, kMaxSamples> attackRole{ +1.0f, -0.4f, +0.9f, -0.5f, +0.7f, -0.3f, +0.4f, -0.6f };
    /// Sustain/Continuant role weight rho^sus_i in [-1, +1] (>0 = prefers sustains)
    std::array<float, kMaxSamples> sustainRole{ +0.5f, +1.0f, -0.4f, +0.9f, +0.6f, +0.8f, -0.2f, +0.7f };
    /// Tonal vs. Noisy role bias rho^tonal_i in [-1, +1] (+1 = pure harmonics, -1 = noise/residual)
    std::array<float, kMaxSamples> tonalRole  { +1.0f, +0.8f, -0.5f, +0.9f, +0.4f, -0.7f, -0.8f, +0.6f };
    /// Hole-filler affinity weight rho^comp_i in [0, 2] (boosts secondary foley/texture slots in valleys)
    std::array<float, kMaxSamples> holeFillerAffinity{ 0.5f, 0.7f, 1.2f, 1.3f, 1.0f, 1.4f, 1.5f, 1.2f };
    /// Dominant peak weight in occupancy calculation O_{-i}(t, b)
    std::array<float, kMaxSamples> dominanceWeight   { 1.0f, 1.0f, 0.8f, 0.9f, 0.85f, 0.75f, 0.7f, 0.7f };
    /// When true, blends static role weights with instantaneous HPR ratios from the sample data
    bool adaptiveHprRoleInference = true;
};

/**
 * @brief Output of Family 2 (Descriptor-Driven Competition & Anti-Masking) for a single frame t.
 */
struct alignas(64) FeatureCompetitionOutput {
    /// Normalized zero-mean, unit-variance descriptor logits bar{Phi}^Desc_i(t, b)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> normalizedLogits{};
    /// Raw unnormalized descriptor potentials Phi^Desc_i(t, b)
    alignas(64) std::array<std::array<float, kMaxBands>, kMaxSamples> rawPotentials{};
    /// Onset activation gate Omega_onset(t, b) in [0, 1], forwarded to Family 3 for Transient Markov Reset
    alignas(64) std::array<float, kMaxBands> onsetActivationGate{};
};

// ============================================================================
// FeatureCompetitor — Family 2 Evaluation Engine
// ============================================================================

class FeatureCompetitor {
public:
    FeatureCompetitor() noexcept = default;

    /**
     * @brief Configures per-slot spectromorphological roles (Quadrant 2 UI matrix).
     */
    void setRoleAllocation(const SampleRoleAllocation& roles) noexcept {
        roles_ = roles;
    }

    [[nodiscard]] const SampleRoleAllocation& getRoleAllocation() const noexcept {
        return roles_;
    }

    /**
     * @brief Evaluates Family 2 (Descriptor Competition & Anti-Masking Hole Filler)
     *        across all N active samples and B psychoacoustic ERB bands.
     *
     * @param params           Decoded genome & macro parameters
     * @param descriptors      Current frame ERB band descriptors (Energy, Flux, Flatness, HPR)
     * @param numActiveSamples Number of active sample slots N in [2, 8]
     * @param numBands         Number of psychoacoustic ERB bands B in [16, 128]
     * @param out              Output struct receiving bar{Phi}^Desc_i(t, b) and Omega_onset(t, b)
     */
    void evaluateFrame(
        const DecodedGenomeParams& params,
        const FrameErbDescriptors& descriptors,
        std::size_t numActiveSamples,
        std::size_t numBands,
        FeatureCompetitionOutput& out) const noexcept
    {
        const std::size_t N = std::clamp<std::size_t>(numActiveSamples, 1, kMaxSamples);
        const std::size_t B = std::clamp<std::size_t>(numBands, 1, kMaxBands);

        // Occupancy sigmoid parameters for Anti-Masking O_{-i}(t, b)
        constexpr float kOccupancyThreshold = 0.42f;
        constexpr float kOccupancySlope     = 10.0f;

        for (std::size_t b = 0; b < B; ++b) {
            // ================================================================
            // 1. Pooled Spectral Flux & Onset Activation Gate Omega_onset(t, b)
            // ================================================================
            float maxFlux = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                maxFlux = std::max(maxFlux, descriptors.spectralFlux[i][b]);
            }

            const float omegaOnset = detail::sigmoid(
                params.onsetFluxSlope * (maxFlux - params.onsetFluxThreshold));
            out.onsetActivationGate[b] = omegaOnset;
            const float omegaContinuant = 1.0f - omegaOnset;

            // Pre-find the top-2 weighted energies in band b so we can compute
            // hat{E}_{-i}(t, b) = max_{j != i} (w_dom_j * hat{E}_j(t, b)) in O(N) instead of O(N^2)!
            float       maxDomEnergy1 = 0.0f;
            float       maxDomEnergy2 = 0.0f;
            std::size_t maxDomIndex1  = 0;

            for (std::size_t i = 0; i < N; ++i) {
                const float weightedE = detail::clamp01(descriptors.normalizedEnergy[i][b])
                                      * roles_.dominanceWeight[i];
                if (weightedE >= maxDomEnergy1) {
                    maxDomEnergy2 = maxDomEnergy1;
                    maxDomEnergy1 = weightedE;
                    maxDomIndex1  = i;
                } else if (weightedE > maxDomEnergy2) {
                    maxDomEnergy2 = weightedE;
                }
            }

            // ================================================================
            // 2. Evaluate Phi^Split_i, Phi^Flat_i, and Phi^Comp_i for each slot i
            // ================================================================
            float bandMean = 0.0f;

            for (std::size_t i = 0; i < N; ++i) {
                const float eNorm    = detail::clamp01(descriptors.normalizedEnergy[i][b]);
                const float flux     = detail::clamp01(descriptors.spectralFlux[i][b]);
                const float flatness = detail::clamp01(descriptors.spectralFlatness[i][b]);
                const float hRatio   = detail::clamp01(descriptors.harmonicRatio[i][b]);
                const float pRatio   = detail::clamp01(descriptors.percussiveRatio[i][b]);

                // --- (A) Spectromorphological Onset / Continuant Split: Phi^Split_i(t, b) ---
                float rhoAtk = roles_.attackRole[i];
                float rhoSus = roles_.sustainRole[i];
                if (roles_.adaptiveHprRoleInference) {
                    // Blend static role matrix with instantaneous HPR dominance
                    rhoAtk = 0.6f * rhoAtk + 0.4f * (2.0f * pRatio - 1.0f);
                    rhoSus = 0.6f * rhoSus + 0.4f * (2.0f * hRatio - 1.0f);
                }

                const float attackFeature  = 0.65f * flux + 0.35f * pRatio * eNorm;
                const float sustainFeature = 0.65f * (1.0f - flatness) * eNorm + 0.35f * hRatio * eNorm;
                const float phiSplit       = omegaOnset      * rhoAtk * attackFeature
                                           + omegaContinuant * rhoSus * sustainFeature;

                // --- (B) Tonal vs. Noisy Competition via Spectral Flatness: Phi^Flat_i(t, b) ---
                // Tonality index in [-1, +1]: +1 for pure sinusoid (flatness = 0), -1 for white noise (flatness = 1)
                const float tonalityIndex = 1.0f - 2.0f * flatness;
                const float phiFlat       = roles_.tonalRole[i] * tonalityIndex * (0.35f + 0.65f * eNorm);

                // --- (C) Energy Complementarity / Anti-Masking (Hole Filler): Phi^Comp_i(t, b) ---
                // Exact formula from Section 2.1:
                //   Phi^Comp_i(t, b) = hat{E}_i(t, b) * (1 - lambda_anti * O_{-i}(t, b))
                //                    - lambda_anti * O_{-i}(t, b) * hat{E}_{-i}(t, b)
                const float eMinusI = (i == maxDomIndex1) ? maxDomEnergy2 : maxDomEnergy1;
                const float oMinusI = detail::sigmoid(kOccupancySlope * (eMinusI - kOccupancyThreshold));

                const float lambdaEff = params.lambdaAntiMask;
                const float rawComp   = eNorm * (1.0f - lambdaEff * oMinusI)
                                      - lambdaEff * oMinusI * eMinusI;
                const float phiComp   = roles_.holeFillerAffinity[i] * rawComp;

                // Combine the 3 sub-rules of Family 2
                const float phiDesc = params.gammaSplit      * phiSplit
                                    + params.gammaFlatness   * phiFlat
                                    + params.gammaComplement * phiComp;

                out.rawPotentials[i][b] = phiDesc;
                bandMean += phiDesc;
            }

            // ================================================================
            // 3. Zero-Mean, Unit-Variance Normalization across i in {0 .. N-1}
            // ================================================================
            bandMean /= static_cast<float>(N);
            float bandVar = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                const float diff = out.rawPotentials[i][b] - bandMean;
                bandVar += diff * diff;
            }
            const float invStd = 1.0f / std::sqrt(bandVar / static_cast<float>(N) + 1.0e-5f);

            for (std::size_t i = 0; i < N; ++i) {
                out.normalizedLogits[i][b] = (out.rawPotentials[i][b] - bandMean) * invStd;
            }
            for (std::size_t i = N; i < kMaxSamples; ++i) {
                out.rawPotentials[i][b]    = 0.0f;
                out.normalizedLogits[i][b] = 0.0f;
            }
        }
    }

private:
    SampleRoleAllocation roles_{};
};

} // namespace spectile::rules
