"""
test_spectile_proto.py
======================
Automated Unit Test Suite (`unittest`) for SpecTile Python Golden Prototype (Settore 5).
Mathematically verifies:
1. Cooley-Tukey Radix-2 FFT/IFFT & 2-Real-in-1-Complex RFFT/IRFFT invertibility
2. Gaussian Window NOLA/COLA conditions & WOLA STFT/ISTFT exact reconstruction
3. Psychoacoustic ERB Filterbank (Glasberg & Moore) & Cosine-Tapered B -> K Partition of Unity (sum M_i = 1)
4. 3-Way HPR Decomposition (W_H + W_P + W_R = 1, H + P + R = |S|, directional selectivity)
5. Iterative Cepstral True Envelope (upper-envelope guarantee E_i >= |S_i| and bounded F_i)
6. 4 Rule Families, Sub-150Hz Band Lock, and 24D Genome QR-Orthonormality (U^T U = I_6)
7. Causal MS-PGHI Max-Heap Phase Reconstruction: superior Spectral Convergence (SC_dB)
   and Phase Gradient Coherence (PGC) compared to Naive ISM Bin Switching.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

# Ensure parent directory is in sys.path so spectile_proto can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spectile_proto.spectile_engine import (
    ARCHETYPES,
    ERBFilterbank,
    RuleConfig,
    SpectralGenomeEngine,
    analyze_sample,
    check_nola_cola,
    compute_rule_masks,
    decompose_hpr,
    erb_bandwidth_hz,
    erb_rate_to_hz,
    extract_true_envelope,
    feather_masks_2d,
    fft_radix2,
    gaussian_window,
    hz_to_erb_rate,
    irfft_batch,
    istft,
    rfft_batch,
    stft,
    synthesize_chimera,
)


class TestSpecTileEngine(unittest.TestCase):
    """Comprehensive mathematical verification of the SpecTile DSP & Rule Engine."""

    def test_01_fft_and_rfft_batch_invertibility(self) -> None:
        """Verifies Radix-2 FFT/IFFT and paired RFFT/IRFFT batch accuracy (< 1e-11)."""
        rng = random.Random(42)
        M = 512
        x = [complex(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(M)]
        X = fft_radix2(x, inverse=False)
        x_rec = fft_radix2(X, inverse=True)
        max_err = max(abs(x[n] - x_rec[n]) for n in range(M))
        self.assertLess(max_err, 1e-11, f"Complex FFT/IFFT error too large: {max_err}")

        # Verify Parseval's Theorem: sum |x[n]|^2 == (1/M) * sum |X[k]|^2
        energy_time = sum(abs(v) ** 2 for v in x)
        energy_freq = sum(abs(v) ** 2 for v in X) / M
        self.assertAlmostEqual(energy_time, energy_freq, places=9)

        # Verify paired 2-real-in-1-complex RFFT/IRFFT batching
        frames = [
            [rng.uniform(-1.0, 1.0) for _ in range(M)]
            for _ in range(5)
        ]
        spectra = rfft_batch(frames, M)
        frames_rec = irfft_batch(spectra, M)
        for t in range(5):
            err_t = max(abs(frames[t][n] - frames_rec[t][n]) for n in range(M))
            self.assertLess(err_t, 1e-11)

    def test_02_nola_cola_and_stft_istft_perfect_reconstruction(self) -> None:
        """Verifies Gaussian window NOLA/COLA and WOLA STFT -> ISTFT identity."""
        M = 512
        R = 128
        win, _, kappa_tf = gaussian_window(M, R)
        self.assertGreater(kappa_tf, 0.5)

        cola_info = check_nola_cola(win, M, R)
        self.assertTrue(cola_info["nola_satisfied"])
        self.assertLess(cola_info["wola_max_unity_error"], 1e-12)
        self.assertLess(cola_info["cola_relative_ripple"], 1e-2)

        # Synthesize a test signal with multiple harmonics and transients
        fs = 24000.0
        N_samples = 4096
        sig = [
            0.5 * math.sin(2.0 * math.pi * 220.0 * n / fs)
            + 0.3 * math.cos(2.0 * math.pi * 880.0 * n / fs)
            + (0.8 if 1000 <= n < 1020 else 0.0)
            for n in range(N_samples)
        ]
        S, _ = stft(sig, M=M, R=R)
        sig_rec = istft(S, M=M, R=R, original_length=N_samples)

        max_err = max(abs(sig[n] - sig_rec[n]) for n in range(N_samples))
        self.assertLess(max_err, 1e-11, f"WOLA STFT/ISTFT reconstruction error: {max_err}")

    def test_03_erb_mapping_and_partition_of_unity(self) -> None:
        """Verifies Glasberg & Moore ERB bijectivity and exact sum_{i=1}^N M_i(t,k) == 1."""
        for f_test in [0.0, 100.0, 440.0, 2500.0, 12000.0]:
            e_cam = hz_to_erb_rate(f_test)
            f_inv = erb_rate_to_hz(e_cam)
            self.assertAlmostEqual(f_test, f_inv, places=6)
            self.assertGreater(erb_bandwidth_hz(f_test), 24.0)

        M = 512
        B = 32
        N = 4
        T = 12
        erb_fb = ERBFilterbank(M=M, fs=24000.0, num_bands=B)

        # Generate random normalized band masks [N][T][B] with sum_i M_hat_i(t,b) = 1
        rng = random.Random(99)
        band_masks = [[[0.0] * B for _ in range(T)] for _ in range(N)]
        for t in range(T):
            for b in range(B):
                raw = [rng.random() + 0.05 for _ in range(N)]
                s = sum(raw)
                for i in range(N):
                    band_masks[i][t][b] = raw[i] / s

        # Project B -> K via cosine-tapered interpolation
        bin_masks = erb_fb.project_masks_b_to_k(band_masks)
        K = erb_fb.K
        for t in range(T):
            for k in range(K):
                s_k = sum(bin_masks[i][t][k] for i in range(N))
                self.assertAlmostEqual(s_k, 1.0, places=12)
                for i in range(N):
                    self.assertGreaterEqual(bin_masks[i][t][k], 0.0)
                    self.assertLessEqual(bin_masks[i][t][k], 1.0)

        # Verify 2D Spectral Feathering also preserves Partition of Unity
        feathered = feather_masks_2d(bin_masks, sigma_t=1.2, sigma_k=2.5)
        for t in range(T):
            for k in range(K):
                s_f = sum(feathered[i][t][k] for i in range(N))
                self.assertAlmostEqual(s_f, 1.0, places=12)

    def test_04_hpr_3way_decomposition(self) -> None:
        """Verifies 3-Way HPR decomposition: W_H + W_P + W_R == 1, H + P + R == |S|, and selectivity."""
        T = 24
        K = 65
        mag = [[0.05] * K for _ in range(T)]

        # Inject a steady horizontal harmonic partial at bin k=20
        for t in range(T):
            mag[t][20] = 2.5

        # Inject a sharp vertical transient click at frame t=12 across bins 35..60
        for k in range(35, 61):
            mag[12][k] = 3.0

        hpr = decompose_hpr(mag, L_t=7, L_k=7, beta_hpr=2.0, gamma_m=2.5)
        W_H, W_P, W_R = hpr["W_H"], hpr["W_P"], hpr["W_R"]
        H, P, R_c = hpr["H"], hpr["P"], hpr["R"]

        for t in range(T):
            for k in range(K):
                w_sum = W_H[t][k] + W_P[t][k] + W_R[t][k]
                self.assertAlmostEqual(w_sum, 1.0, places=11)
                mag_sum = H[t][k] + P[t][k] + R_c[t][k]
                self.assertAlmostEqual(mag_sum, mag[t][k], places=11)

        # Check horizontal tone dominance at (t=6, k=20)
        self.assertGreater(W_H[6][20], 0.85, "Steady harmonic bin should be dominated by W_H")
        # Check vertical transient dominance at (t=12, k=48)
        self.assertGreater(W_P[12][48], 0.85, "Broadband transient bin should be dominated by W_P")

    def test_05_cepstral_true_envelope_upper_bound(self) -> None:
        """Verifies Iterative Cepstral True Envelope: E_i(t,k) >= |S_i(t,k)| and F_i in [0, 1.05]."""
        K = 257
        fs = 24000.0
        M = 2 * (K - 1)
        # Construct harmonic series modulated by two vocal formants at 700 Hz and 1800 Hz
        mag_frame = [0.002] * K
        f0 = 180.0
        for h in range(1, int((0.5 * fs) // f0)):
            f_h = h * f0
            k_bin = int(round(f_h * M / fs))
            if k_bin < K:
                formant_env = math.exp(-((f_h - 700.0) / 350.0) ** 2) + 0.6 * math.exp(
                    -((f_h - 1800.0) / 450.0) ** 2
                )
                mag_frame[k_bin] = 0.05 + 1.8 * formant_env

        E_frame, F_frame = extract_true_envelope(
            mag_frame, fs=fs, f0_hz=f0, num_iters=8, delta_overshoot=0.0, lambda_tik=1e-4
        )
        for k in range(K):
            self.assertGreaterEqual(
                E_frame[k] + 1e-12,
                mag_frame[k],
                f"True Envelope violated upper bound E_i >= |S_i| at bin k={k}",
            )
            self.assertGreaterEqual(F_frame[k], 0.0)
            self.assertLessEqual(F_frame[k], 1.05)

    def test_06_genome_qr_orthonormality_and_sub150hz_band_lock(self) -> None:
        """Verifies QR orthonormal branches (U^T U = I_6) and Sub-150Hz Band Lock."""
        genome_eng = SpectralGenomeEngine(dim=24, num_branches=6)
        U = genome_eng.U_ortho
        for i in range(6):
            for j in range(6):
                dot = sum(U[i][d] * U[j][d] for d in range(24))
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(dot, expected, places=10)

        g0 = [0.5] * 24
        g_mut = genome_eng.mutate_genome(g0, rho=0.75, theta=1.1)
        self.assertEqual(len(g_mut), 24)
        for val in g_mut:
            self.assertGreater(val, 0.0)
            self.assertLess(val, 1.0)

    def test_07_mspghi_improves_spectral_convergence_over_naive_ism(self) -> None:
        """
        Verifies that SpecTile Causal MS-PGHI achieves significantly better (lower)
        Spectral Convergence (SC_dB) and higher Phase Gradient Coherence (PGC)
        than Baseline Naive Bin Switching (SoundHack ISM / FFTease ether~).
        """
        fs = 24000.0
        M = 512
        R = 128
        N_len = 3072
        erb_fb = ERBFilterbank(M=M, fs=fs, num_bands=32)

        # Synthesize 3 distinct miniature test sources
        s1 = [
            0.6 * math.sin(2.0 * math.pi * 110.0 * n / fs)
            + 0.4 * math.sin(2.0 * math.pi * 220.0 * n / fs)
            + 0.25 * math.sin(2.0 * math.pi * 330.0 * n / fs)
            for n in range(N_len)
        ]
        s2 = [
            0.5 * math.sin(2.0 * math.pi * 275.0 * n / fs + 0.8)
            + 0.45 * math.sin(2.0 * math.pi * 550.0 * n / fs + 1.4)
            + 0.35 * math.sin(2.0 * math.pi * 825.0 * n / fs + 2.1)
            for n in range(N_len)
        ]
        rng = random.Random(77)
        s3 = [
            (0.9 if (n % 512) < 40 else 0.15) * rng.uniform(-0.8, 0.8)
            for n in range(N_len)
        ]

        samples = [
            analyze_sample(s1, erb_fb, name="S1_Bass", sample_id=0, f0_hz=110.0, M=M, R=R),
            analyze_sample(s2, erb_fb, name="S2_Choir", sample_id=1, f0_hz=275.0, M=M, R=R),
            analyze_sample(s3, erb_fb, name="S3_Strike", sample_id=2, f0_hz=200.0, M=M, R=R),
        ]

        cfg_naive = RuleConfig(
            name="Baseline_Naive_ISM",
            phase_mode="naive",
            sub_lock_enabled=False,
        )
        cfg_mspghi = RuleConfig(
            name="SpecTile_Voronoi_MSPGHI",
            phase_mode="mspghi",
            beta_glue=0.95,
            sub_lock_enabled=True,
            sub_lock_hz=150.0,
            sub_lock_sample=0,
        )

        res_naive = synthesize_chimera(samples, erb_fb, cfg_naive, target_length=N_len)
        res_mspghi = synthesize_chimera(samples, erb_fb, cfg_mspghi, target_length=N_len)

        sc_naive = res_naive["metrics"]["SC_dB"]  # type: ignore
        sc_mspghi = res_mspghi["metrics"]["SC_dB"]  # type: ignore
        pgc_naive = res_naive["metrics"]["PGC"]  # type: ignore
        pgc_mspghi = res_mspghi["metrics"]["PGC"]  # type: ignore

        # Verify Sub-150Hz Lock on res_mspghi masks before feathering
        raw_masks = compute_rule_masks(samples, erb_fb, cfg_mspghi)["band_masks_fine"]
        for t in range(samples[0].T):
            for b in range(erb_fb.B):
                if erb_fb.band_freqs[b] <= 140.0:
                    self.assertGreater(raw_masks[0][t][b], 0.99)

        # MS-PGHI must improve Spectral Convergence (more negative dB) and PGC
        self.assertLess(
            sc_mspghi,
            sc_naive - 3.0,
            f"Expected MS-PGHI SC_dB ({sc_mspghi} dB) < Naive SC_dB ({sc_naive} dB) - 3 dB",
        )
        self.assertGreater(
            pgc_mspghi,
            pgc_naive,
            f"Expected MS-PGHI PGC ({pgc_mspghi}) > Naive PGC ({pgc_naive})",
        )

    def test_08_generated_golden_vectors_wav_and_svg_assets(self) -> None:
        """Verifies that all 4 WAV demos, 2 SVG plots, and golden_vectors.json exist and are valid."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(base_dir, ".."))

        expected_wavs = [
            "1_baseline_naive_bin_switching.wav",
            "2_spectile_voronoi_mspghi.wav",
            "3_spectile_harmonic_interleave.wav",
            "4_spectile_transient_formant_graft.wav",
        ]
        for wname in expected_wavs:
            wpath = os.path.join(base_dir, "output", "audio", wname)
            self.assertTrue(os.path.isfile(wpath), f"Missing WAV file: {wpath}")
            self.assertGreater(os.path.getsize(wpath), 50000)

        expected_svgs = [
            "fig1_voronoi_polychromatic_masks.svg",
            "fig2_spectral_convergence_ablation.svg",
        ]
        for sname in expected_svgs:
            spath = os.path.join(base_dir, "output", "figures", sname)
            self.assertTrue(os.path.isfile(spath), f"Missing SVG file: {spath}")
            self.assertGreater(os.path.getsize(spath), 4000)

        golden_path = os.path.join(repo_root, "tests", "golden_data", "golden_vectors.json")
        self.assertTrue(os.path.isfile(golden_path), f"Missing Golden JSON: {golden_path}")
        self.assertGreater(os.path.getsize(golden_path), 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
