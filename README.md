# 🏛️ Complete Amazon Archaeological Discovery Pipeline

A modular, extensible pipeline for the automated detection and analysis of archaeological features in the Amazon using satellite imagery, LiDAR, and advanced geospatial analytics.

---

## ✅ All Files Created
- **src/__init__.py** – Package initialization with clean imports
- **src/processors.py** – Advanced image processing utilities (image enhancement, spectral analysis, preprocessing)
- **src/visualizers.py** – Interactive maps, dashboards, and analysis plots
- **requirements.txt** – Complete dependency list
- **setup.py** – Professional package setup for installation
- **.env.template** – Environment variables template with all API credentials
- **notebooks/analysis.ipynb** – Comprehensive Jupyter notebook for interactive analysis
- **src/drive_downloader.py** – Utility for downloading large GEE exports from Google Drive

---

## 🚀 Major Update: Modular Pipeline Architecture (2024)

### What's New?
- **Modular Pipeline:** Each step (download, analyze, score, report, visualize) is now a standalone class with a `run()` method.
- **Provider Abstraction:** USGS and GEE providers are abstracted behind a common interface (`BaseProvider`).
- **Feature Awareness:** The pipeline checks for required bands/features and adapts or skips gracefully if missing.
- **CLI Flag:** Use `--modular-pipeline` to run the new architecture end-to-end.
- **Backward Compatibility:** The legacy pipeline remains available as a fallback.

### Why Modular?
- **Extensibility:** Add new providers, steps, or features with minimal changes.
- **Testability:** Each step can be tested in isolation.
- **Robustness:** Handles missing data/features gracefully.
- **Clear Data Flow:** Standardized objects (`SceneData`) are passed between steps.

---

## Usage

### Legacy Pipeline (Default)
Run the full pipeline as before:
```bash
python main.py --zone negro_madeira --full-pipeline
```

### Modular Pipeline (Recommended)
Run the new modular pipeline:
```bash
python main.py --zone negro_madeira --modular-pipeline --provider gee
```
- Use `--provider usgs`, `--provider gee`, or `--provider both` as needed.
- All major outputs (report, map) are summarized at the end.

#### Example: All Zones, Both Providers
```bash
python main.py --zone all --modular-pipeline --provider both
```

---

## Pipeline Steps & Classes
| Step         | Modular Class                        |
|--------------|--------------------------------------|
| Download     | `USGSProvider`, `GEEProvider`        |
| Analyze      | `AnalysisStep`                       |
| Score        | `ScoringStep`                        |
| Report       | `ReportStep`                         |
| Visualize    | `VisualizationStep`                  |
| Orchestrator | `ModularPipeline`                    |

---

## Provider Abstraction
- All providers implement the `BaseProvider` interface.
- Add new providers by subclassing `BaseProvider` and implementing `download_data()`.

## Feature Awareness
- Each `SceneData` object lists available bands/features.
- Analysis and scoring steps check for required features and skip/adapt if missing.

---

## Migration & Transition
- **Legacy pipeline** remains available for backward compatibility.
- **Modular pipeline** is recommended for new workflows and future development.
- CLI options and outputs are consistent between both modes.

---

## Existing Usage (Legacy)
(Original instructions remain here...)

---

## 🛡️ USGS Authentication (Required)

**You must use a USGS ERS Application Token for all API access.**

1. Go to [https://ers.cr.usgs.gov/](https://ers.cr.usgs.gov/) and log in.
2. On your profile page, create an "Application Token" with the "M2M API" scope.
3. Copy the token value (it will only be shown once).
4. Set your credentials in `.env`:
   ```env
   USGS_USERNAME=your_ers_username
   USGS_TOKEN=your_application_token
   ```
5. The pipeline will use these credentials to authenticate and set the required `X-Auth-Token` header for all API requests.

---

## 🗂️ Project Structure
```
amazon-discovery/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── usgs_api.py
│   ├── detectors.py
│   ├── processors.py
│   ├── scoring.py
│   ├── visualizers.py
│   └── drive_downloader.py
├── main.py
├── requirements.txt
├── setup.py
├── .env.template
├── data/
├── results/
└── notebooks/
    └── analysis.ipynb
```

---

## ⚡ USGS Data Download & Provider Selection

- The pipeline now defaults to USGS for all satellite data operations.
- GEE (Google Earth Engine) is supported for legacy/optional use only. To use GEE, you must specify `--provider gee` and have GEE credentials set up.
- **Recommended:** Use USGS as the provider for all new projects.

### Example CLI Usage

- **List target zones:**
  ```bash
  python main.py --list-zones
  ```
- **Download data for a zone (USGS):**
  ```bash
  python main.py --zone negro_madeira --provider usgs --download
  ```
- **Run full pipeline (USGS):**
  ```bash
  python main.py --zone negro_madeira --provider usgs --full-pipeline
  ```
- **Analyze existing data:**
  ```bash
  python main.py --zone negro_madeira --provider usgs --analyze-existing --score --report --visualize
  ```

---

## 🛠️ Troubleshooting
- **USGS authentication failed:**
  - Ensure your `.env` contains a valid `USGS_USERNAME` and `USGS_TOKEN` (application token, not password).
  - The token must have the "M2M API" scope.
  - If you see `Authentication failed: User credential verification failed`, your token or username is incorrect or missing the correct scope.
- **Large data exports:**
  - USGS downloads are direct; GEE exports (if used) may require Google Drive as described below.
- **Missing data directory:**
  - Ensure you place the downloaded file in the correct `data/satellite/<zone_name>/` directory.
- **File format issues:**
  - Convert KML to GeoTIFF or GeoJSON as needed for your pipeline.

---

## 📞 Need Help?
Open an issue or contact the maintainers for further support.

---

## 🚀 Ready to Launch!

### Next Steps for You

#### 1. Setup Environment
```bash
cp .env.template .env
# Edit .env with your USGS_USERNAME and USGS_TOKEN (application token, not password)

pip install -r requirements.txt
```

#### 2. Test the System
```bash
# List target zones
python main.py --list-zones

# Download data for priority zone (USGS)
python main.py --zone negro_madeira --provider usgs --download

# Run full pipeline (USGS)
python main.py --zone negro_madeira --provider usgs --full-pipeline
```

#### 3. Interactive Analysis
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## ✨ Key Features Now Complete

### 🔍 Detection Engine
- Terra preta spectral analysis
- Geometric pattern detection (circles, earthworks, linear features)
- Advanced image processing with contrast enhancement and noise reduction

### 🧮 Scoring System
- 15-point convergent anomaly scale
- Multi-modal evidence integration
- Confidence classification and prioritization

### 📊 Visualization Suite
- Interactive Folium maps with discoveries
- Plotly dashboards for scoring analysis
- Statistical plots for method effectiveness
- Zone comparison visualizations

### 📓 Analysis Notebook
- Executive summary dashboard
- Zone-by-zone detailed analysis
- Statistical method evaluation
- Interactive exploration functions
- Export capabilities (CSV, Excel)

---

## 📝 Contributing
1. Fork the repository and create a feature branch
2. Follow the modular structure for new features
3. Write clear docstrings and comments
4. Submit a pull request with a detailed description

---

## 📚 References
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Landsat Missions](https://landsat.gsfc.nasa.gov/)
- [Geopandas Documentation](https://geopandas.org/)

---

## 📧 Contact
For questions or collaboration, please open an issue or contact the maintainer.

---

## 🛡️ Security Note
- **Never share your USGS application token.**
- If your token is compromised, revoke it immediately from your ERS profile. 