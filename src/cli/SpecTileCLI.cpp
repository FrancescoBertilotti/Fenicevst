// =============================================================================
// SpecTileCLI.cpp
// =============================================================================
// Native C++20 Standalone Command-Line Synthesizer & Real-Time Profiler for
// SpecTile: Rule-Based Multi-Sample Spectral Mosaicing and Chimeric Resynthesis.
//
// Exercises the production C++20 header-only modules directly:
//   - Settore 1: SpectralTypesSoA, LockFreePrimitives, ERBScaleMapper,
//                HPRAndCepstrumProcessor, SpectralFeathering2D,
//                CausalMSPGHIReconstructor
//   - Settore 2: SpectralGenomeEngine, GeometricTiler, FeatureCompetitor,
//                RuleEnginePipeline
//
// Usage:
//   spectile_cli.exe [--archetype 0..5] [--glue 0.85] [--duration 3.0] [--out path.wav]
// =============================================================================

#include "../core/SpectralTypesSoA.hpp"
#include "../core/LockFreePrimitives.hpp"
#include "../dsp/ERBScaleMapper.hpp"
#include "../dsp/HPRAndCepstrum.hpp"
#include "../dsp/FeatheringAndPGHI.hpp"
#include "../rules/SpectralGenome.hpp"
#include "../rules/GeometricTiler.hpp"
#include "../rules/FeatureCompetitor.hpp"
#include "../rules/RuleEnginePipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace {

constexpr double kSampleRate = 48000.0;
constexpr std::size_t kFftSize = 4096;
constexpr std::size_t kHopSize = 512;
constexpr std::size_t kNumBins = (kFftSize / 2) + 1; // 2049
constexpr std::size_t kNumErbBands = 64;
constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kTwoPi = 2.0f * kPi;

// Simple Radix-2 In-Place Complex FFT/IFFT for WOLA Synthesis
void radix2Fft(std::vector<float>& real, std::vector<float>& imag, bool inverse) noexcept {
    const std::size_t n = real.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    const float sign = inverse ? 1.0f : -1.0f;
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const float angleStep = sign * kTwoPi / static_cast<float>(len);
        const float wStepR = std::cos(angleStep);
        const float wStepI = std::sin(angleStep);

        for (std::size_t i = 0; i < n; i += len) {
            float wR = 1.0f;
            float wI = 0.0f;
            for (std::size_t k = 0; k < half; ++k) {
                const std::size_t u = i + k;
                const std::size_t v = u + half;
                const float tR = wR * real[v] - wI * imag[v];
                const float tI = wR * imag[v] + wI * real[v];
                real[v] = real[u] - tR;
                imag[v] = imag[u] - tI;
                real[u] += tR;
                imag[u] += tI;

                const float nextWR = wR * wStepR - wI * wStepI;
                wI = wR * wStepI + wI * wStepR;
                wR = nextWR;
            }
        }
    }

    if (inverse) {
        const float invN = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            real[i] *= invN;
            imag[i] *= invN;
        }
    }
}

bool writeWav16Mono(const std::string& path, const std::vector<float>& samples, std::uint32_t sampleRate) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        return false;
    }

    // Normalize peak to -1.0 dBFS (0.891)
    float maxAbs = 1.0e-6f;
    for (float s : samples) {
        maxAbs = std::max(maxAbs, std::abs(s));
    }
    const float gain = 0.891f / maxAbs;

    const std::uint32_t numSamples = static_cast<std::uint32_t>(samples.size());
    const std::uint16_t numChannels = 1;
    const std::uint16_t bitsPerSample = 16;
    const std::uint32_t byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const std::uint16_t blockAlign = numChannels * (bitsPerSample / 8);
    const std::uint32_t dataBytes = numSamples * blockAlign;
    const std::uint32_t riffChunkSize = 36 + dataBytes;

    ofs.write("RIFF", 4);
    ofs.write(reinterpret_cast<const char*>(&riffChunkSize), 4);
    ofs.write("WAVE", 4);
    ofs.write("fmt ", 4);
    const std::uint32_t fmtSize = 16;
    const std::uint16_t audioFormat = 1; // PCM
    ofs.write(reinterpret_cast<const char*>(&fmtSize), 4);
    ofs.write(reinterpret_cast<const char*>(&audioFormat), 2);
    ofs.write(reinterpret_cast<const char*>(&numChannels), 2);
    ofs.write(reinterpret_cast<const char*>(&sampleRate), 4);
    ofs.write(reinterpret_cast<const char*>(&byteRate), 4);
    ofs.write(reinterpret_cast<const char*>(&blockAlign), 2);
    ofs.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
    ofs.write("data", 4);
    ofs.write(reinterpret_cast<const char*>(&dataBytes), 4);

    for (float s : samples) {
        const float clamped = std::clamp(s * gain, -1.0f, 1.0f);
        const auto pcm16 = static_cast<std::int16_t>(std::lround(clamped * 32767.0f));
        ofs.write(reinterpret_cast<const char*>(&pcm16), 2);
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t archetypeIdx = 5; // Default: A_6 Formant Choir
    float glueOverride = -1.0f;
    float durationSec = 3.0f;
    std::string outPath = "spectile_proto/output/audio/cpp_native_chimera.wav";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--archetype" && i + 1 < argc) {
            archetypeIdx = std::clamp<std::size_t>(std::stoul(argv[++i]), 0, 5);
        } else if (arg == "--glue" && i + 1 < argc) {
            glueOverride = std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
        } else if (arg == "--duration" && i + 1 < argc) {
            durationSec = std::clamp(std::stof(argv[++i]), 0.5f, 15.0f);
        } else if (arg == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        }
    }

    const std::size_t numFrames = static_cast<std::size_t>((durationSec * kSampleRate) / kHopSize);
    const std::size_t numActiveSamples = 3;

    std::cout << "======================================================================\n";
    std::cout << " SpecTile C++20 Native Spectral Mosaicing & Chimeric Resynthesis CLI\n";
    std::cout << "======================================================================\n";
    std::cout << "  Compiler / SIMD : C++20 SoA 64-Byte Aligned (K_padded = "
              << spectile::core::kDefaultPaddedBins << ")\n";
    std::cout << "  STFT Config     : M = " << kFftSize << ", R = " << kHopSize
              << " (87.5% overlap), fs = " << kSampleRate << " Hz\n";
    std::cout << "  Duration        : " << durationSec << " s (" << numFrames << " STFT hops)\n";

    // -------------------------------------------------------------------------
    // 1. Initialize Settore 1 DSP Processors
    // -------------------------------------------------------------------------
    spectile::dsp::ERBScaleMapper erbMapper(kFftSize, kSampleRate, kNumErbBands);
    spectile::dsp::HPRAndCepstrumProcessor hprCepstrum(kFftSize, kSampleRate);
    spectile::dsp::SpectralFeathering2D feathering2D(kNumBins);
    spectile::dsp::CausalMSPGHIReconstructor mspghi(kFftSize, kHopSize);

    // -------------------------------------------------------------------------
    // 2. Initialize Settore 2 Generative Rule Engine & 24D Spectral Genome
    // -------------------------------------------------------------------------
    const auto archId = static_cast<spectile::rules::ArchetypeId>(archetypeIdx);
    const spectile::rules::GenomeVec seedGenome =
        spectile::rules::SpectralGenomeEngine::sampleFromArchetype(archId, 0x2026CAFEULL, 0.35f);
    const spectile::rules::EvolutionaryBranchBasis qrBasis =
        spectile::rules::SpectralGenomeEngine::buildOrthogonalBranchesQR(seedGenome, archId);

    spectile::rules::MacroControls macros{0.55f, 0.50f, (glueOverride >= 0.0f) ? glueOverride : 0.82f};
    spectile::rules::DecodedGenomeParams decodedParams =
        spectile::rules::SpectralGenomeEngine::decodeGenome(seedGenome, macros, numActiveSamples);
    if (glueOverride >= 0.0f) {
        decodedParams.phaseGlueBeta = glueOverride;
    }

    std::cout << "  Active Archetype: [" << archetypeIdx << "] "
              << spectile::rules::kArchetypeTable[archetypeIdx].name << "\n";
    std::cout << "  Decoded Params  : Voronoi Seeds = " << decodedParams.voronoiSeeds
              << " | Softmax tau = " << std::fixed << std::setprecision(3) << decodedParams.softmaxTempTau
              << " | Phase Glue beta = " << decodedParams.phaseGlueBeta
              << " | Cross-Synth chi = " << decodedParams.crossSynthChi << "\n";

    spectile::rules::RuleEnginePipeline ruleEngine;
    ruleEngine.prepare(static_cast<float>(kSampleRate), kFftSize, kNumErbBands);

    // -------------------------------------------------------------------------
    // 3. Pre-Analyze 3 Procedural Multi-Component Spectral Sources into SoA Pool
    // -------------------------------------------------------------------------
    std::array<spectile::core::SpectralSampleData, 3> sampleStorage;
    std::array<const spectile::core::SpectralSampleData*, 3> samplePoolPtrs{};

    const auto tStartAnalysis = std::chrono::high_resolution_clock::now();
    for (std::size_t s = 0; s < numActiveSamples; ++s) {
        sampleStorage[s].allocate(numFrames, kFftSize, kHopSize, kSampleRate, kNumErbBands);
        sampleStorage[s].fundamentalHz = 110.0f * (1.0f + 0.5f * static_cast<float>(s));
        samplePoolPtrs[s] = &sampleStorage[s];

        spectile::core::SpectralFrameSoA tempFrame(kFftSize);
        const float f0 = 110.0f; // Common fundamental A2 = 110 Hz for harmonic weaving

        for (std::size_t t = 0; t < numFrames; ++t) {
            const float timeSec = static_cast<float>(t * kHopSize) / static_cast<float>(kSampleRate);
            const float omegaStep = kTwoPi * static_cast<float>(kHopSize) / static_cast<float>(kFftSize);
            auto magSpan = sampleStorage[s].mutableFrameMagnitude(t);
            auto phaseSpan = sampleStorage[s].mutableFramePhase(t);
            auto ifSpan = sampleStorage[s].mutableFrameInstFreq(t);
            auto gdSpan = sampleStorage[s].mutableFrameGroupDelay(t);

            for (std::size_t k = 0; k < kNumBins; ++k) {
                const float freqHz = static_cast<float>(k) * static_cast<float>(kSampleRate) / static_cast<float>(kFftSize);
                float mag = 1.0e-5f;

                if (s == 0) {
                    // Slot 0: Warm Bowed Cello Sawtooth + Formant Resonances (280 Hz, 720 Hz, 2100 Hz)
                    const float harm = std::fmod(freqHz, f0);
                    const float distH = std::min(harm, f0 - harm);
                    const float comb = std::exp(-(distH * distH) / 18.0f);
                    const float f1 = std::exp(-std::pow((freqHz - 280.0f) / 140.0f, 2.0f));
                    const float f2 = 0.7f * std::exp(-std::pow((freqHz - 750.0f) / 220.0f, 2.0f));
                    const float f3 = 0.4f * std::exp(-std::pow((freqHz - 2100.0f) / 400.0f, 2.0f));
                    mag = (0.8f * f1 + f2 + f3 + 0.1f) * comb * (120.0f / (freqHz + 120.0f));
                } else if (s == 1) {
                    // Slot 1: Percussive Rhythmic Transient Plucks + Metallic Inharmonic Bell Partials
                    const float beatPhase = std::fmod(timeSec * 3.0f, 1.0f);
                    const float transientEnv = std::exp(-beatPhase * 12.0f);
                    const float bell = std::exp(-std::pow(std::sin(freqHz * 0.018f), 2.0f) * 14.0f);
                    mag = (0.75f * transientEnv + 0.25f * bell) * std::exp(-freqHz / 4500.0f);
                } else {
                    // Slot 2: Breathy Vocal Formant Choir (Moving Vowel Morph + Aspirated Air)
                    const float vowelLfo = std::sin(kTwoPi * 0.45f * timeSec);
                    const float vF1 = 520.0f + 180.0f * vowelLfo;
                    const float vF2 = 1650.0f - 450.0f * vowelLfo;
                    const float envV = std::exp(-std::pow((freqHz - vF1) / 160.0f, 2.0f))
                                     + 0.8f * std::exp(-std::pow((freqHz - vF2) / 260.0f, 2.0f));
                    const float harm = std::fmod(freqHz, f0 * 2.0f);
                    const float distH = std::min(harm, f0 * 2.0f - harm);
                    const float harmonicCore = std::exp(-(distH * distH) / 24.0f);
                    mag = envV * (0.82f * harmonicCore + 0.18f);
                }

                magSpan[k] = mag;
                ifSpan[k] = omegaStep * static_cast<float>(k);
                phaseSpan[k] = std::fmod(ifSpan[k] * static_cast<float>(t), kTwoPi) - kPi;
                gdSpan[k] = 0.0f;
            }
        }

        // Complete 2D HPR Separation + Iterative Cepstral True Envelope + Tikhonov Fine Structure + ERB Descriptors
        hprCepstrum.analyzeSampleComplete(sampleStorage[s], erbMapper);
    }
    const auto tEndAnalysis = std::chrono::high_resolution_clock::now();
    const double analysisMs = std::chrono::duration<double, std::milli>(tEndAnalysis - tStartAnalysis).count();

    // -------------------------------------------------------------------------
    // 4. Real-Time STFT Hop Loop: Settore 2 Rules + Settore 1 MS-PGHI + WOLA
    // -------------------------------------------------------------------------
    spectile::rules::FrameErbDescriptors erbDesc{};
    spectile::rules::MaskMatrix2D rulesMasks{};
    spectile::core::MaskMatrix2D coreMasks(kNumBins, kNumErbBands);
    spectile::core::SpectralFrameSoA outFrame(kFftSize);

    // Precompute matched COLA Gaussian window for WOLA synthesis
    std::vector<float> window(kFftSize, 0.0f);
    const float sigmaWin = static_cast<float>(kFftSize) / 5.0132565f;
    const float centerWin = 0.5f * static_cast<float>(kFftSize);
    for (std::size_t n = 0; n < kFftSize; ++n) {
        const float diff = (static_cast<float>(n) - centerWin) / sigmaWin;
        window[n] = std::exp(-0.5f * diff * diff);
    }

    const std::size_t totalOutSamples = numFrames * kHopSize + kFftSize;
    std::vector<float> outputAudio(totalOutSamples, 0.0f);
    std::vector<float> windowNorm(totalOutSamples, 0.0f);
    std::vector<float> fftReal(kFftSize, 0.0f);
    std::vector<float> fftImag(kFftSize, 0.0f);

    double totalRulesUs = 0.0;
    double totalSynthUs = 0.0;
    double totalPghiUs  = 0.0;

    feathering2D.setParameters(decodedParams.featherFreqSigma * 0.25f, decodedParams.featherTimeAlpha, false);

    for (std::size_t t = 0; t < numFrames; ++t) {
        const float timeSec = static_cast<float>(t * kHopSize) / static_cast<float>(kSampleRate);

        // Continuously evolve genome slightly along QR Branch 0 using geodesic SLERP
        const float rhoLfo = 0.25f * (0.5f + 0.5f * std::sin(kTwoPi * 0.2f * timeSec));
        const auto liveGenome = spectile::rules::SpectralGenomeEngine::evaluateBranchSlerp(qrBasis, rhoLfo, 0.4f * timeSec);
        decodedParams = spectile::rules::SpectralGenomeEngine::decodeGenome(liveGenome, macros, numActiveSamples);
        if (glueOverride >= 0.0f) {
            decodedParams.phaseGlueBeta = glueOverride;
        }

        // Populate ERB descriptors from pre-analyzed sampleStorage for Settore 2 FeatureCompetitor
        for (std::size_t s = 0; s < numActiveSamples; ++s) {
            const auto eSpan    = sampleStorage[s].frameERBEnergy(t);
            const auto fluxSpan = sampleStorage[s].frameERBFlux(t);
            const auto flatSpan = sampleStorage[s].frameERBFlatness(t);
            const auto harmSpan = sampleStorage[s].frameERBHarmonicity(t);
            for (std::size_t b = 0; b < kNumErbBands; ++b) {
                erbDesc.normalizedEnergy[s][b] = eSpan[b];
                erbDesc.spectralFlux[s][b]     = fluxSpan[b];
                erbDesc.spectralFlatness[s][b] = flatSpan[b];
                erbDesc.harmonicRatio[s][b]    = harmSpan[b];
                erbDesc.percussiveRatio[s][b]  = 1.0f - harmSpan[b];
            }
        }

        // --- Stage A: Settore 2 Rule Engine Pipeline ---
        const auto t0 = std::chrono::high_resolution_clock::now();
        ruleEngine.processHop(decodedParams, erbDesc, timeSec, 110.0f, numActiveSamples, rulesMasks);
        const auto t1 = std::chrono::high_resolution_clock::now();
        totalRulesUs += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Bridge Settore 2 MaskMatrix2D -> Settore 1 core::MaskMatrix2D
        for (std::size_t s = 0; s < numActiveSamples; ++s) {
            auto dstEnv = coreMasks.binEnvSlot(s);
            auto dstFine = coreMasks.binFineSlot(s);
            for (std::size_t k = 0; k < kNumBins; ++k) {
                dstEnv[k] = rulesMasks.envMasksFft[s][k];
                dstFine[k] = rulesMasks.fineMasksFft[s][k];
            }
        }
        coreMasks.updatePurityAndNormalizeBins(numActiveSamples);
        feathering2D.process(coreMasks, numActiveSamples);

        // --- Stage B: Source-Filter & HPR Chimeric Magnitude Recombination ---
        const auto t2 = std::chrono::high_resolution_clock::now();
        const std::array<std::size_t, 3> frameIndices{t, t, t};
        const std::array<float, 3> pitchRatios{1.0f, 1.0f, 1.0f};
        spectile::dsp::CausalMSPGHIReconstructor::synthesizeChimericMagnitude(
            samplePoolPtrs,
            frameIndices,
            pitchRatios,
            coreMasks,
            outFrame,
            decodedParams.envelopeGamma);
        const auto t3 = std::chrono::high_resolution_clock::now();
        totalSynthUs += std::chrono::duration<double, std::micro>(t3 - t2).count();

        // --- Stage C: Causal MS-PGHI Phase Reconstruction on Static FlatMaxHeap ---
        const auto t4 = std::chrono::high_resolution_clock::now();
        mspghi.reconstructPhase(outFrame, coreMasks, decodedParams.phaseGlueBeta);
        const auto t5 = std::chrono::high_resolution_clock::now();
        totalPghiUs += std::chrono::duration<double, std::micro>(t5 - t4).count();

        // --- Stage D: Hermitian Symmetry + Radix-2 IFFT + WOLA Overlap-Add ---
        std::fill(fftReal.begin(), fftReal.end(), 0.0f);
        std::fill(fftImag.begin(), fftImag.end(), 0.0f);
        for (std::size_t k = 0; k < kNumBins; ++k) {
            fftReal[k] = outFrame.real[k];
            fftImag[k] = outFrame.imag[k];
            if (k > 0 && k < kNumBins - 1) {
                fftReal[kFftSize - k] = outFrame.real[k];
                fftImag[kFftSize - k] = -outFrame.imag[k];
            }
        }
        radix2Fft(fftReal, fftImag, true);

        const std::size_t writeOffset = t * kHopSize;
        for (std::size_t n = 0; n < kFftSize; ++n) {
            const float w = window[n];
            outputAudio[writeOffset + n] += fftReal[n] * w;
            windowNorm[writeOffset + n] += w * w;
        }
    }

    for (std::size_t n = 0; n < totalOutSamples; ++n) {
        if (windowNorm[n] > 1.0e-5f) {
            outputAudio[n] /= windowNorm[n];
        }
    }

    const double avgRulesUs = totalRulesUs / static_cast<double>(numFrames);
    const double avgSynthUs = totalSynthUs / static_cast<double>(numFrames);
    const double avgPghiUs  = totalPghiUs  / static_cast<double>(numFrames);
    const double avgTotalUs = avgRulesUs + avgSynthUs + avgPghiUs;
    const double hopBudgetUs = (static_cast<double>(kHopSize) / kSampleRate) * 1.0e6; // 10,666.67 us
    const double cpuLoadPercent = (avgTotalUs / hopBudgetUs) * 100.0;

    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " Real-Time C++20 Profiling (Per STFT Hop Budget = "
              << std::fixed << std::setprecision(1) << hopBudgetUs << " us)\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  Pre-Analysis (HPR + True Envelope) : " << std::setprecision(2) << analysisMs << " ms total\n";
    std::cout << "  Settore 2 Rule Engine (Voronoi+2D) : " << avgRulesUs << " us / hop\n";
    std::cout << "  Source-Filter Chimeric Recomb      : " << avgSynthUs << " us / hop\n";
    std::cout << "  Causal MS-PGHI Max-Heap Phase Glue : " << avgPghiUs  << " us / hop\n";
    std::cout << "  Total Real-Time DSP + Rules Cost   : " << avgTotalUs << " us / hop ("
              << std::setprecision(2) << cpuLoadPercent << "% of 1 core)\n";
    std::cout << "----------------------------------------------------------------------\n";

    if (writeWav16Mono(outPath, outputAudio, static_cast<std::uint32_t>(kSampleRate))) {
        std::cout << "  [OK] Wrote C++20 Chimeric Audio -> " << outPath << "\n";
    } else {
        std::cout << "  [WARN] Could not open output path: " << outPath << "\n";
    }
    std::cout << "======================================================================\n";
    return 0;
}
