# SpecTile Settore 5 — Experimental Ablation Report & Golden Reference Summary

- **Engine:** Pure Python 3.12 Golden Reference (`spectile_proto/spectile_engine.py`)
- **Total Execution Time:** `37.43 s`
- **STFT Configuration:** `fs = 24000 Hz`, `M = 1024`, `R = 256` (75% Overlap), `B = 48` ERB Bands
- **Primary Exit Gate ($\Delta\text{SC}_{\text{dB}} \ge 8\text{ dB}$):** **PASSED (`+9.81 dB` improvement over Naive Bin Switching)**

## 1. Objective Evaluation Metrics (SMC / DAFx / CIM Paper Table)

| Condition / Archetype | Phase & Mask Mode | Spectral Convergence $\text{SC}_{\text{dB}}$ ($\downarrow$) | Log-Spectral Distance $\text{LSD}_{\text{dB}}$ ($\downarrow$) | Phase Gradient Coherence $\text{PGC}$ ($\uparrow$) | Transient Preservation $\text{TPI}$ ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1_Baseline_Naive_ISM** | `Hard Binary + Raw Phase` | **-6.10 dB** | 7.75 dB | **0.606** | 0.844 |
| **1b_ERB_Feathering_Only** | `2D Feather + Raw Phase` | **-10.57 dB** | 4.96 dB | **0.770** | 0.520 |
| **2_SpecTile_Voronoi_MSPGHI** | `Voronoi + Feather + MS-PGHI` | **-15.91 dB** | 3.48 dB | **0.866** | 0.671 |
| **3_SpecTile_Harmonic_Interleave** | `von Mises Comb + Sub-150Hz Lock` | **-14.90 dB** | 3.03 dB | **0.858** | 0.784 |
| **4_SpecTile_Transient_Formant_Graft** | `Flux Split + E_i*F_j + MS-PGHI` | **-14.85 dB** | 2.65 dB | **0.852** | 0.601 |

## 2. Generated Chimeric Audio Demos (`spectile_proto/output/audio/`)

| Audio File | Sample Rate | Peak Level | RMS Level | Size (Bytes) |
| :--- | :---: | :---: | :---: | :---: |
| `1_baseline_naive_bin_switching.wav` | `24000 Hz` | `-1.00 dBFS` | `-14.13 dBFS` | `105,644 B` |
| `2_spectile_voronoi_mspghi.wav` | `24000 Hz` | `-1.00 dBFS` | `-14.73 dBFS` | `105,644 B` |
| `3_spectile_harmonic_interleave.wav` | `24000 Hz` | `-1.00 dBFS` | `-12.97 dBFS` | `105,644 B` |
| `4_spectile_transient_formant_graft.wav` | `24000 Hz` | `-1.00 dBFS` | `-10.35 dBFS` | `105,644 B` |

## 3. Generated Vector Plots & Golden Data

- `spectile_proto/output/figures/fig1_voronoi_polychromatic_masks.svg`
- `spectile_proto/output/figures/fig2_spectral_convergence_ablation.svg`
- `tests/golden_data/golden_vectors.json`
