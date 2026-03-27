# QM2 Unified Data Analysis App

A Flask-based web application for real-time data analysis and visualization at the **QM2 (CHESS ID4B) beamline**, supporting powder diffraction, reciprocal-space slice viewing, thin-film mapping, 1D linecut extraction, and pyFAI azimuthal integration — all from a single browser interface.

---

## Features

### 🔬 Browse & Analyze (Powder Data)
- Navigate the experiment directory tree through a sidebar file browser
- Plot the averaged radial sum `(f1 + f2 + f3) / 3` from `.nxs` NeXus files
- Adjustable Q-range (X-min / X-max) with interactive Plotly charts
- Export data as CSV

### 🌡️ Temperature-Dependent Overlays
- Select multiple temperature-encoded files from a folder
- Generate multi-line overlay plots for temperature-series comparison
- Export overlay data as CSV

### 📊 NxProcess Comparison
- Side-by-side comparison of powder patterns from two different folders
- Overlay plots with per-file selection for each folder
- Export comparison data as CSV

### 🗂️ NxRefine Slice Viewer
Four sub-tabs for exploring reciprocal-space `.nxs` slice data:

| Tab | Description |
|-----|-------------|
| **Single File** | View slices from one `.nxs` file at chosen L/K/H values |
| **Compare Two** | Side-by-side comparison of slices from files A and B |
| **Compare Multiple** | Display slices from up to 4 files simultaneously |
| **Thin-Film** | Fast 2D slab extraction with rotation, scaling, and axis-range controls |

**Shared controls across all slice tabs:**
- Colormap selection
- Global vmin / vmax or percentile-based autoscale
- Skew angle input (override the default per-axis skew)
- Show/hide reciprocal lattice grid overlay
- ⬇ PNG download button on every plot

**Thin-Film tab extras:**
- Slice axis selector: fix **Ql**, **Qk**, or **Qh** — view the perpendicular 2D plane
- Display mode: **linear**, **log**, or **sqrt** intensity scaling
- X / Y axis range controls (auto or explicit min/max)
- Per-slice elapsed time display

### 📐 1D Linecut Tool
- Choose a `.nxs` file, axis, and slice value
- Draw a line cut by specifying start `(X₁, Y₁)` and end `(X₂, Y₂)` in r.l.u.
- Returns overlay image showing the cut path and a 1D intensity profile (log scale)
- Export linecut profile as CSV

### ⚡ pyFAI Integration
- Single-image or batch-folder azimuthal integration
- File browser for `.cbf` / `.tif` / `.edf` images, `.poni` calibration files, and mask files
- Integrated Plotly 1D `I(Q)` and 2D `I(2θ, χ)` output plots
- Height-scan viewer for multi-image series

---

## Requirements

```
Python >= 3.9
Flask
numpy
scipy
matplotlib
h5py
nexusformat
nxs-analysis-tools
fabio
pyFAI
pandas
```

Install dependencies:

```bash
pip install flask numpy scipy matplotlib h5py nexusformat nxs-analysis-tools fabio pyFAI pandas
```

> **Note:** `nxs-analysis-tools` provides the `plot_slice` function used for all reciprocal-space rendering. Ensure it is installed from the correct CHESS internal or public source.

---

## Configuration

Open `app.py` and adjust the following constants near the top of the file:

```python
# Root directory for experiment data (.nxs files, folder tree)
ROOT = "/nfs/chess/id4baux/2026-1"

# Root directory for raw detector images (used by pyFAI)
PYFAI_IMG_ROOT = "/nfs/chess/id4b/2026-1"

# Temporary cache directory for slice data
CACHE_DIR = "/tmp/lslice_cache"

# Max parallel workers for I/O
MAX_IO_WORKERS = 4
```

The CHESS logo is loaded at startup from `/nfs/chess/id4baux/chesslogo.png`. If that path does not exist, the app runs normally without the logo.

---

## Running the App

```bash
python app.py
```

The server starts on all interfaces at port **5000**:

```
http://localhost:5000
```

To use a different port, edit the last line of `app.py`:

```python
app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
```

---

## URL Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Home / landing page |
| `/browse` | GET | Directory browser |
| `/analyze_file` | GET | Single-file powder plot |
| `/plot` | POST | Render powder radial sum |
| `/export_csv` | POST | Download averaged data as CSV |
| `/select_temps` | GET | Temperature file selector |
| `/plot_temp` | POST | Temperature overlay plot |
| `/export_temp_csv` | POST | Download overlay CSV |
| `/nxprocess` | GET | NxProcess folder comparison |
| `/choose_folder_A` | GET | Select folder A for comparison |
| `/choose_folder_B` | POST | Select folder B for comparison |
| `/plot_temp_compare` | POST | Render comparison overlay |
| `/export_compare_csv` | POST | Download comparison CSV |
| `/slices` | GET / POST | NxRefine Slice Viewer (all tabs) |
| `/slices/thinfilm` | POST | Thin-film slab generation |
| `/slices/linecut` | GET / POST | 1D linecut tool |
| `/slices/linecut/csv` | POST | Download linecut profile CSV |
| `/pyfai` | GET | pyFAI integration page |
| `/pyfai/browse` | GET | File/folder browser for pyFAI |
| `/pyfai/pick` | POST | Confirm picked file/folder |
| `/pyfai/run` | POST | Run azimuthal integration |
| `/pyfai/height_one_ajax` | POST | AJAX call for height-scan frame |
| `/help` | GET | Help / documentation page |

---

## Architecture Notes

- **Single-file app** — all HTML, CSS, JavaScript, and Python logic live in `app.py` using Flask's `render_template_string`.
- **In-memory plot rendering** — matplotlib figures are rendered to PNG in memory and embedded as base64 `data:image/png` URIs; no static file server needed.
- **NeXus / HDF5 access** — uses `nexusformat.nexus.nxload` with slab indexing (`data[L_idx, :, :]`) for memory-safe reads of large datasets.
- **Slice caching** — slice metadata and rendered images are cached in `_sv_slice_cache` (in-process dict, thread-safe via `threading.Lock`) to avoid redundant disk reads.
- **Thin-film rotation** — `scipy.ndimage.rotate` (order=1, `cval=0.0`) with optional `scipy.ndimage.zoom` downsampling for fast rendering.
- **Dark / light theme** — CSS custom properties toggled via JavaScript; persists via `localStorage`.

---

## Project Structure

```
app.py          # Complete Flask application (all logic, templates, and styles)
README.md       # This file
```

---

## Beamline Context

This app is developed for the **QM2 beamline (ID4B)** at the Cornell High Energy Synchrotron Source (**CHESS**). It is designed to run on a beamline workstation with direct NFS access to experiment data directories.

---

## License

Internal CHESS / QM2 beamline tool. Please contact the beamline staff before redistributing.
