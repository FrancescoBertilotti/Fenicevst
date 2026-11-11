#!/usr/bin/env python3
"""
SpecTile — Interactive CLI & Custom WAV Chimera Generator
Permette di provare il vero motore DSP (STFT + HPR + True Envelope + Voronoi/Markov + MS-PGHI)
sia con i campioni sintetici interni sia caricando i propri file .wav (16-bit PCM).

Uso rapido:
  python spectile_proto/run_custom_chimera.py --archetype voronoi --glue 0.9 --out mia_chimera.wav
  python spectile_proto/run_custom_chimera.py --inputs s1.wav s2.wav s3.wav --archetype graft --out custom.wav
"""

import argparse
import copy
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectile_proto.spectile_engine import (
    ARCHETYPES,
    ERBFilterbank,
    SpectralSampleData,
    analyze_sample,
    synthesize_chimera,
)
from spectile_proto.generate_demos_and_golden import (
    synthesize_source_samples,
    write_wav_16bit,
)


def load_wav_mono_float(path: str, target_len: int) -> list[float]:
    """Carica un file .wav 16-bit PCM e lo converte in float [-1, 1]."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(min(n_frames, target_len * 2))

    if sampwidth == 2:
        fmt = f"<{len(raw) // 2}h"
        ints = struct.unpack(fmt, raw)
        samples = [x / 32768.0 for x in ints[::n_channels]]
    else:
        raise ValueError(f"Formato WAV non supportato (richiesto 16-bit PCM): {path}")

    if len(samples) < target_len:
        samples = (samples * ((target_len // max(1, len(samples))) + 1))[:target_len]
    else:
        samples = samples[:target_len]
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="SpecTile DSP Chimera Generator")
    parser.add_argument(
        "--archetype",
        choices=["voronoi", "interleave", "graft", "hole_filler", "freeze", "choir"],
        default="voronoi",
        help="Archetipo di regola spettrale da applicare",
    )
    parser.add_argument("--tile-size", type=float, default=0.45, help="Macro 1: Tile Size [0.05..1.0]")
    parser.add_argument("--glue", type=float, default=0.88, help="Macro 3: Phase & Spectral Glue (MS-PGHI) [0.0..1.0]")
    parser.add_argument("--sub-lock-hz", type=float, default=150.0, help="Frequenza Sub-Lock ancorata a S1 (Hz)")
    parser.add_argument("--inputs", nargs="*", help="Lista opzionale di file .wav 16-bit da combinare (2..8 file)")
    parser.add_argument(
        "--out",
        default=os.path.join("spectile_proto", "output", "audio", "custom_chimera.wav"),
        help="Percorso del file .wav generato",
    )
    args = parser.parse_args()

    fs = 24000.0
    M = 1024
    R = 256
    B = 48
    duration_s = 2.2
    target_len = int(fs * duration_s)
    erb_fb = ERBFilterbank(M=M, fs=fs, num_bands=B)

    if args.inputs and len(args.inputs) >= 2:
        print(f"[SpecTile] Caricamento e pre-analisi di {len(args.inputs)} file WAV utente...")
        samples: list[SpectralSampleData] = []
        for idx, p in enumerate(args.inputs[:8]):
            sig = load_wav_mono_float(p, target_len)
            samples.append(
                analyze_sample(sig, erb_fb, name=os.path.basename(p), sample_id=idx, f0_hz=110.0 * (idx + 1), M=M, R=R, cepstral_iters=6)
            )
    else:
        print("[SpecTile] Generazione e pre-analisi dei 4 campioni sorgente (Sub Saw, Vocal Formant, Metal Strike, Arctic Air)...")
        raw_sources = synthesize_source_samples(fs=fs, duration_s=duration_s)
        f0_list = [110.0, 220.0, 285.0, 400.0]
        samples = [
            analyze_sample(sig, erb_fb, name=name, sample_id=idx, f0_hz=f0_list[idx], M=M, R=R, cepstral_iters=6)
            for idx, (name, sig) in enumerate(raw_sources.items())
        ]

    arch_key_map = {
        "interleave": "A1_Harmonic_Interleave",
        "graft": "A2_Transient_Formant_Graft",
        "voronoi": "A3_Voronoi_Spectral_Shatter",
        "hole_filler": "A4_Complementarity_Hole_Filler",
        "freeze": "A5_Spectral_Freeze_Morph",
        "choir": "A6_Formant_Choir_Cross",
    }
    cfg = copy.deepcopy(ARCHETYPES[arch_key_map[args.archetype]])
    cfg.sigma_t = max(0.04, args.tile_size * 0.45)
    cfg.sigma_f = max(0.04, args.tile_size * 0.40)
    cfg.beta_glue = max(0.0, min(1.0, args.glue))
    cfg.sub_lock_hz = args.sub_lock_hz

    print(f"[SpecTile] Sintesi Chimerica in corso -> Archetipo: {cfg.name} | Phase Glue: {cfg.beta_glue:.2f} | Sub-Lock: {cfg.sub_lock_hz:.0f} Hz...")
    res = synthesize_chimera(samples, erb_fb, cfg, target_length=target_len)
    wav_info = write_wav_16bit(args.out, res["audio"], fs=int(fs))
    m = res["metrics"]

    print("-" * 72)
    print(f"[OK] File audio generato : {os.path.abspath(args.out)} ({wav_info['file_size_bytes']:,} bytes)")
    print(f"     Spectral Convergence: {m['SC_dB']:.2f} dB (piu' negativo = migliore coerenza)")
    print(f"     Log-Spectral Dist.  : {m['LSD_dB']:.2f} dB")
    print(f"     Phase Grad Coherence: {m['PGC']:.3f} (1.0 = coerenza perfetta ai bordi)")
    print(f"     Transient Preserv.  : {m['TPI']:.3f}")
    print("-" * 72)


if __name__ == "__main__":
    main()
