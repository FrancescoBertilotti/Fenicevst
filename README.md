# FENICE — Spectral Chimera Synthesizer (JUCE 8 / C++20)

Repository di sviluppo per **FENICE**: combinazione del **motore DSP spettrale C++20 / JUCE 8** (target `SpecTile` VST3 / Standalone / CLI), del **prototipo di riferimento Python** (`spectile_proto/`) e del **nuovo Design System UI/UX** per l'interfaccia minimale *FENICE* (`design/`).

> **Stato attuale del repository**
> - Il backend DSP, la suite di test di parità Python↔C++ e i target JUCE (`VST3`, `Standalone`, `CLI`) compilano ed eseguono sotto il target `SpecTile`.
> - La nuova UI minimalista **FENICE** (tema scuro di default *Obsidian Ember*, selettore centrale *The Fenice Eclipse*, workflow *user-loaded samples* negli slot `I–VIII` e *Pro View* separata) è documentata e prototipata in [`design/`](design/) e pronta per essere portata in C++/JUCE nel prossimo step di sviluppo.

---

## Struttura del Repository

```text
fenice/
├── src/                        # Engine C++20 / JUCE 8
│   ├── core/                   # Tipi SoA SIMD-aligned & primitive lock-free
│   ├── dsp/                    # Mapping ERB, HPR/Cepstrum, Feathering & PGHI
│   ├── rules/                  # Spectral Genome, Geometric Tiler, Feature Competitor
│   ├── engine/                 # PluginProcessor, Voice, StaggeredHopScheduler, SampleAnalysisWorker
│   ├── ui/                     # Prototipo UI attuale (in attesa di migrazione a FENICE Design System)
│   └── cli/                    # Tool da riga di comando (spectile_cli)
├── design/                     # FENICE UI/UX Design System & prototipi interattivi
│   ├── fenice_interactive_mockup.html
│   ├── fenice_main_view_dark.svg
│   ├── fenice_pro_view_dark.svg
│   ├── fenice_main_view_light_concept.svg
│   └── FENICE_DESIGN_SYSTEM.md
├── spectile_proto/             # Prototipo DSP di riferimento in Python (NumPy/SciPy)
│   ├── spectile_engine.py
│   ├── generate_demos_and_golden.py
│   ├── run_custom_chimera.py
│   ├── test_spectile_proto.py
│   └── output/ablation_metrics_report.md
├── tests/                      # Unit test C++20 & Golden Vectors (parità Python ↔ C++)
│   ├── TestSpecTileDSP.cpp
│   └── golden_data/golden_vectors.json
├── cmake/                      # Configurazione dipendenze e flag SIMD (Dependencies.cmake)
├── docs/                       # Specifiche tecniche e piano architetturale
├── external/JUCE/              # Submodule Git ufficiale JUCE (tag 8.0.4)
├── CMakeLists.txt
└── build_spectile.bat          # Script build automatico su Windows
```

---

## Come Clonare il Repository

Il repository utilizza `external/JUCE` come **Git submodule** (`JUCE 8.0.4`). Per clonare correttamente anche JUCE:

```bash
git clone --recurse-submodules <URL_DEL_REPOSITORY>
cd fenice
```

Se il repository è già stato clonato senza `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

> **Fallback automatico senza submodule:** se `external/JUCE` risultasse vuoto e non si usa il submodule, CMake supporta `-DSPECTILE_FETCH_EXTERNAL_DEPS=ON` per scaricare JUCE `8.0.4` via `FetchContent`.

---

## Requisiti e Build (C++20 / VST3 / Standalone)

### Requisiti
- **CMake** `>= 3.22`
- Compilatore **C++20** (su Windows: Visual Studio 2022 o Build Tools con toolchain C++ MSVC)
- *(Opzionale per prototipo Python)* **Python 3.10+** con `numpy`, `scipy`, `matplotlib`

### Build rapida su Windows

Lo script [`build_spectile.bat`](build_spectile.bat) rileva automaticamente `cmake` nel `PATH` o nelle installazioni di Visual Studio 2022 e compila tutti i target in `Release`:

```bat
.\build_spectile.bat
```

### Build manuale con CMake (Windows x64)

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64 -DSPECTILE_BUILD_PLUGIN=ON -DSPECTILE_FETCH_EXTERNAL_DEPS=OFF
cmake --build build --config Release --parallel
```

> **Nota su Windows ARM64 + Visual Studio x64:** [`cmake/Dependencies.cmake`](cmake/Dependencies.cmake) gestisce automaticamente la compilazione cross-host di `juceaide` verso `x64` quando il sistema host ARM64 utilizza il toolset VS x64.

### Artefatti generati (esclusi da Git)

Al termine della compilazione in `Release`, gli output locali si trovano in `build/` (cartella esclusa da `.gitignore`):
- **VST3:** `build\SpecTile_artefacts\Release\VST3\SpecTile.vst3`
- **Standalone:** `build\SpecTile_artefacts\Release\Standalone\SpecTile.exe`
- **CLI:** `build\Release\spectile_cli.exe`
- **Unit Test C++20:** `build\Release\spectile_unit_tests.exe`

---

## Esecuzione Test e Prototipo Python

### 1. Unit Test C++20 (Golden Vector Parity)

```bat
.\build\Release\spectile_unit_tests.exe
```

I test utilizzano [`tests/golden_data/golden_vectors.json`](tests/golden_data/golden_vectors.json) (risolto automaticamente tramite definizione CMake `SPECTILE_GOLDEN_JSON_PATH` o fallback relativo).

### 2. Test Suite Python di Riferimento

```bash
python -m unittest spectile_proto.test_spectile_proto
```

### 3. Generazione Demo Audio & Report Ablation

```bash
python -m spectile_proto.generate_demos_and_golden
```

*(I file `.wav` in `spectile_proto/output/audio/` e i grafici `.svg` in `spectile_proto/output/figures/` sono esclusi da Git e rigenerabili localmente.)*

---

## Preview Design UI/UX (FENICE)

Apri direttamente nel browser [`design/fenice_interactive_mockup.html`](design/fenice_interactive_mockup.html) per provare il prototipo interattivo del design **FENICE**:
- Caricamento drag & drop o file picker di campioni audio reali dell'utente (`WAV`, `AIFF`, `FLAC`, `MP3`) negli slot `I–VIII`
- *Main View* minimale (*The Fenice Eclipse*, tema scuro *Obsidian Ember* di default)
- *Pro View* separata a 3 colonne con *Trim Inspector* (`IN` / `OUT`) e preview del concept *Light* (*Ceramic & Carbon*)

---

## Licenza

- Le dipendenze esterne mantengono le rispettive licenze: **JUCE 8** è distribuito sotto doppia licenza **GPLv3 / Commerciale** (`external/JUCE`).
- Aggiungere un file `LICENSE` dedicato nella root prima della pubblicazione pubblica se si desidera definire una licenza specifica per il codice proprietario/open-source di FENICE.
