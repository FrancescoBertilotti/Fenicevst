# FENICE — Design System & UI/UX Architecture (User-Loaded Sample Workflow)

> Documento aggiornato per riflettere l'architettura **user-loaded**: FENICE non è un synth a preset chiusi ma uno strumento in cui **l'utente carica i propri file audio** (WAV / AIFF / FLAC / MP3) negli slot `I–VIII` o direttamente sul nucleo centrale. Nessun codice C++ è stato modificato.

---

## 1. Flusso Fondamentale: Caricamento Campioni Utente

L'interazione mantiene la **Main View ultra-minimale** (un solo selettore centrale *The Fenice Eclipse* + dock inferiore `SOURCES`) rendendo il caricamento dei file immediato:

### Nella Main View (`CORE // 1.0`)
1. **Stato Iniziale / Vuoto (`0 / 8`):** se non sono caricati file, il pozzo centrale resta in stato quiescente con l'invito discreto `0 / 8 USER SAMPLES · DROP WAV HERE OR CLICK SLOTS BELOW`.
2. **Drag & Drop diretto sul Nucleo (*The Fenice Eclipse*):** trascinando da 1 a 8 file audio dalla DAW o da Explorer/Finder compare un anello tratteggiato `#FF5500` (*INGEST USER AUDIO*) che popola gli slot liberi `I–VIII`.
3. **Bottom Recessed Dock (`SOURCES` `I–VIII`):**
   - **Slot vuoto (`V · + DROP` – `VIII · + DROP`):** capsula tratteggiata cliccabile (apre il file picker) o bersaglio diretto di drag-and-drop.
   - **Slot caricato (`I · PIANO.W`, `II · VOX_CH`, …):** mostra indicatore termico, nome troncato del file dell'utente e barra di contribuzione; click per mute/unmute e mini-azioni su hover (`↻` sostituisci, `×` svuota slot).

### Nella Pro View Separata (`01 // USER SAMPLES`)
- **Rack completo `I–VIII`:** tutti gli 8 slot (`+ LOAD`, `DEMO`, `CLEAR` in header), distinguendo i file caricati dall'utente (nome file, slider hairline del peso, `↻` / `×`) dagli slot vuoti `+ DROP AUDIO OR CLICK TO BROWSE`.
- **Inspector dello Slot Selezionato:** waveform reale del campione caricato con finestra di **Trim (`IN` / `OUT`)**, selettore `F0 TRACKING` (`AUTO | LOCK | FREE`), `WHITENING`, `SNAP`, Swap `↻` ed Eject `×`.

---

## 2. Filosofia Cromatica & Temi (Zero Rainbow Clutter)

- **Tema Default — Obsidian Ember (Dark):** `#08080A` / `#15151C` / cenere + singolo accento funzionale `#FF5500` → `#FFB800`.
- **Concept Futuro — Ceramic & Carbon (Light):** `#F4F3EF` / `#141416` / rame `#E04800` (solo concept grafico, non implementato in C++).

---

## 3. File del Design Showcase

- `fenice_interactive_mockup.html` — Prototipo interattivo nel browser con upload / drag & drop reale di file audio utente, WebAudio grain resynthesis, Main View *Fenice Eclipse*, Pro View a 3 colonne con Trim Inspector e preview Dark/Light
- `fenice_main_view_dark.svg` — Mockup vettoriale Main View (Dark default)
- `fenice_pro_view_dark.svg` — Mockup vettoriale Pro View (Dark)
- `fenice_main_view_light_concept.svg` — Mockup vettoriale Light Concept (futuro)
