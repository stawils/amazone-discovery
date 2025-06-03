# 🏛️ Amazon Archaeological Discovery Pipeline

**AI-Enhanced Satellite Remote Sensing for Archaeological Site Discovery**

A systematic archaeological discovery framework that combines satellite imagery, space-based LiDAR, AI pattern recognition, and convergent anomaly detection to identify potential archaeological sites in the Amazon rainforest.

---

## 🎯 Competition Entry

This pipeline is designed for the **OpenAI to Z Challenge** and implements all 5 required checkpoints with OpenAI integration. 

[![Competition Status](https://img.shields.io/badge/OpenAI%20to%20Z-In%20Development-blue)](https://kaggle.com/competitions/openai-to-z-challenge)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Project Overview

### **Core Innovation: Convergent Anomaly Detection**

Our approach addresses the challenge of high false positive rates in individual remote sensing techniques by combining multiple independent evidence sources. When 3-4 different anomaly types converge at the same location, the probability of all being false positives decreases significantly.

### **Multi-Modal Data Integration**
- **🛰️ Satellite Spectral Analysis**: Terra preta detection and vegetation patterns (Sentinel-2/Landsat)
- **🚀 Space-Based LiDAR**: Canopy structure and ground elevation (GEDI from ISS)
- **📜 Historical Intelligence**: Systematic extraction of coordinates from expedition accounts
- **🧠 AI Enhancement**: OpenAI model integration for pattern analysis and text mining
- **🌍 Environmental Context**: Settlement suitability and resource availability

**Note**: While GEDI space LiDAR data is included, research has shown limitations for direct archaeological detection. We address this through convergent analysis rather than relying on any single data source.

---

## 🚀 Quick Start

### **OpenAI to Z Challenge Checkpoints**
```bash
# Setup
git clone https://github.com/stawils/amazon-discovery.git
cd amazon-discovery
pip install -r requirements.txt
cp .env.template .env
# Add your OpenAI API key to .env

# Run individual checkpoints
python main.py --checkpoint 1  # Familiarize with data
python main.py --checkpoint 3 --zone negro_madeira  # Site discovery

# Complete checkpoint sequence
python main.py --all-checkpoints
```

### **Research Pipeline**
```bash
# Analyze priority zones
python main.py --pipeline --zones negro_madeira trombetas

# Full multi-modal analysis
python main.py --pipeline --multi-modal --visualize
```

---

## 📁 System Architecture

### **Modular Pipeline Design**
```
📊 Data Sources → 🔍 Detection → 🧮 Scoring → 📈 Analysis → 🗺️ Visualization
```

### **Project Structure**
```
amazon-discovery/
├── src/
│   ├── core/                    # Core algorithms and scoring
│   │   ├── config.py           # Target zones and parameters
│   │   ├── detectors.py        # Archaeological feature detection
│   │   ├── scoring.py          # Convergent anomaly scoring
│   │   └── visualizers.py      # Interactive mapping
│   ├── providers/              # Data providers
│   │   ├── sentinel2_provider.py  # Sentinel-2 spectral analysis
│   │   ├── gedi_provider.py    # GEDI space LiDAR
│   │   ├── usgs_provider.py    # Landsat data
│   │   └── gee_provider.py     # Google Earth Engine
│   ├── checkpoints/            # Competition checkpoints
│   │   ├── checkpoint1.py      # Data familiarization
│   │   ├── checkpoint2.py      # Multi-source analysis
│   │   ├── checkpoint3.py      # Site discovery
│   │   ├── checkpoint4.py      # Impact narrative
│   │   └── checkpoint5.py      # Final submission
│   └── pipeline/               # Analysis workflows
├── main.py                     # Main entry point
├── openai_checkpoints.py       # Competition runner
└── README.md                   # This file
```

---

## 🎯 Target Zones

Based on historical research and environmental analysis, we focus on 5 priority zones:

| Zone | Coordinates | Historical Evidence | Environmental Context |
|------|-------------|-------------------|---------------------|
| **Negro-Madeira Confluence** | -3.17°, -60.00° | Orellana 1542 battle accounts | Major river confluence |
| **Trombetas River Junction** | -1.50°, -56.00° | Amazon warrior encounters | Clearwater tributary access |
| **Upper Xingu Region** | -11.72°, -54.58° | Fawcett expedition target | 81 known sites nearby |
| **Upper Napo Region** | -0.50°, -72.50° | Multiple expedition reports | Major Andean access route |
| **Marañón River System** | -4.00°, -75.00° | 60+ documented Jesuit missions | Western Amazon hub |

---

## 🔬 Detection Methodology

### **Terra Preta Spectral Analysis**
- **Method**: NIR-SWIR index calculation with vegetation analysis
- **Target**: Anthropogenic dark soils indicating ancient settlements
- **Data Sources**: Sentinel-2 (10m), Landsat 8/9 (30m)
- **Validation**: Cross-reference with published archaeological soil studies

### **GEDI Space LiDAR Analysis**
- **Method**: Canopy gap detection and elevation anomaly analysis
- **Limitations**: Research shows GEDI has constraints for direct archaeological detection
- **Integration**: Combined with spectral analysis for convergent evidence
- **Data Source**: NASA GEDI mission from International Space Station

### **Geometric Pattern Recognition**
- **Targets**: Circular earthworks (50-400m), linear causeways, rectangular compounds
- **Methods**: Hough transforms, edge detection, morphological analysis
- **Validation**: Comparison with known Amazon archaeological features

### **Historical Intelligence Mining**
- **Sources**: 16th-century expedition accounts, Jesuit mission records
- **Processing**: OpenAI GPT-4.1 extraction of geographic coordinates
- **Validation**: Cross-reference multiple historical sources

---

## 📊 Convergent Scoring System

**Evidence Integration:**
```
Total Score = Historical(2) + Spectral(3) + Geometric(3) + Environmental(1) + 
              GEDI_Analysis(2) + Convergence_Bonus(3) = 14 points max

Classification:
• 10+ points: HIGH CONFIDENCE - Multiple evidence convergence
• 7-9 points: PROBABLE FEATURE - Strong indicators present  
• 4-6 points: POSSIBLE ANOMALY - Investigate further
• 0-3 points: NATURAL VARIATION - Unlikely archaeological
```

**Quality Control:**
- Minimum 2 independent evidence sources required
- Spatial convergence within 100m radius
- Cross-validation against known non-archaeological features

---

## 📈 Usage Examples

### **Competition Checkpoints**

#### **Checkpoint 1: Data Familiarization**
```bash
python main.py --checkpoint 1
# Downloads satellite scene, runs OpenAI analysis, logs model version
```

#### **Checkpoint 3: Site Discovery**  
```bash
python main.py --checkpoint 3 --zone negro_madeira
# Algorithmic detection + historical cross-reference + comparison to known sites
```

### **Research Analysis**

#### **Multi-Zone Analysis**
```bash
# Analyze multiple priority zones
python main.py --pipeline --zones negro_madeira trombetas upper_xingu

# Generate comprehensive report
python main.py --pipeline --full --report --visualize
```

#### **Single Zone Deep Dive**
```bash
# Detailed analysis of highest priority zone
python main.py --pipeline --zone negro_madeira --provider sentinel2 --max-scenes 5
```

---

## 🔧 Installation & Configuration

### **1. Dependencies**
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `rasterio`, `geopandas` - Geospatial processing
- `scikit-learn`, `opencv-python` - Pattern recognition  
- `folium`, `plotly` - Visualization
- `openai` - AI integration
- `h5py` - GEDI data processing

### **2. Environment Setup**
```bash
cp .env.template .env
```

**Required Variables:**
```env
# OpenAI (Required for competition)
OPENAI_API_KEY=your_openai_api_key

# USGS (Recommended for Landsat)
USGS_USERNAME=your_usgs_username
USGS_TOKEN=your_usgs_application_token

# Google Earth Engine (Optional)
GEE_SERVICE_ACCOUNT_PATH=path/to/service_account.json
GEE_PROJECT_ID=your_gee_project
```

### **3. Validation**
```bash
# Test system setup
python main.py --list-zones

# Validate competition compliance
python -m src.checkpoints.validator
```

---

## 📊 Expected Results

### **Typical Analysis Output**
```
🎯 CONVERGENT ANALYSIS RESULTS:
   Zones Analyzed: 3
   Evidence Sources: 4 (Spectral, GEDI, Historical, Environmental)
   Features Detected: 18
   High Confidence Sites: 3 (scores 10+)

🏆 TOP DISCOVERY:
   Location: Negro-Madeira Confluence
   Score: 12/14 points
   Evidence: Historical + Terra Preta + Geometric + Environmental

📊 EVIDENCE BREAKDOWN:
   Terra Preta Signatures: 8 locations
   Geometric Patterns: 5 locations  
   GEDI Anomalies: 6 locations
   Convergent Sites: 3 locations (multiple evidence types)
```

### **Generated Outputs**
- **Interactive Maps**: Web-based discovery visualization
- **Analysis Reports**: JSON/Markdown documentation
- **Competition Package**: All checkpoint requirements satisfied
- **Export Data**: GeoJSON files for further analysis

---

## 🔍 Data Sources & Limitations

### **Satellite Data**
- **Sentinel-2**: 10m resolution, 5-day revisit, red-edge bands for vegetation analysis
- **Landsat 8/9**: 30m resolution, 16-day revisit, long historical archive
- **Coverage**: Good for Amazon region, limited by cloud cover

### **GEDI Space LiDAR**  
- **Capabilities**: Forest structure, canopy height, ground elevation
- **Limitations**: Research shows constraints for direct archaeological detection
- **Coverage**: 25m footprints, ~4% of Earth's surface, limited spatial density
- **Integration**: Most effective when combined with other data sources

### **Historical Sources**
- **16th Century**: Spanish expedition accounts (Orellana, Carvajal)
- **Colonial Period**: Jesuit mission records, Portuguese expeditions  
- **Processing**: AI-assisted coordinate extraction and validation

---

## 🤝 Contribution & Collaboration

### **Research Foundation**
This system builds on established archaeological research:
- Terra preta studies (Glaser et al., Lehmann et al.)
- Amazon earthwork discoveries (Schaan, Iriarte et al.)
- Remote sensing archaeology (Parcak, Canuto et al.)
- GEDI archaeological applications (Kokalj & Mast, 2021)

### **Contributing**
1. Follow existing code structure and documentation
2. Add tests for new functionality  
3. Ensure competition compliance
4. Update documentation with new features

### **Academic Collaboration**
Welcome collaboration with:
- Archaeological institutions
- Remote sensing researchers
- Amazon indigenous communities
- Conservation organizations

---

## 📚 Scientific Context

### **Methodological Innovation**
- **Convergent Analysis**: Addresses individual technique limitations through multi-modal integration
- **Historical Guidance**: Systematic coordinate extraction from primary sources
- **AI Enhancement**: Pattern recognition and text analysis for archaeological applications
- **Systematic Targeting**: Geographic focus based on environmental and historical factors

### **Research Impact**
Results suitable for publication in:
- Journal of Archaeological Science
- Remote Sensing of Environment
- Archaeological Prospection  
- Latin American Antiquity

---

## 📄 License & Citation

### **License**
MIT License - see LICENSE file for details.

### **Citation**
```bibtex
@software{amazon_archaeological_pipeline,
  title={Amazon Archaeological Discovery Pipeline: Convergent Multi-Modal Analysis},
  author={[Your Name]},
  year={2025},
  url={https://github.com/stawils/amazon-discovery},
  note={OpenAI to Z Challenge submission}
}
```

### **Data Attribution**
- **Satellite Data**: ESA (Sentinel-2), USGS (Landsat)
- **GEDI Data**: NASA Global Ecosystem Dynamics Investigation
- **Historical Sources**: Multiple expedition accounts and archaeological publications

---

## 🏆 Competition Status

### **OpenAI to Z Challenge Compliance**
✅ **All 5 checkpoints implemented**  
✅ **OpenAI integration documented**  
✅ **Multiple data sources processed**  
✅ **Reproducible methodology**  
✅ **Competition requirements met**

### **Submission Timeline**
- **Competition Period**: May 15 - June 29, 2025
- **Current Status**: Development phase
- **Approach**: Systematic, research-based methodology

---

## 🎯 Competitive Advantages

### **Systematic Approach**
- **Historical Research**: Targeted zones based on expedition accounts
- **Multi-Modal Integration**: Addresses individual data source limitations
- **Quality Control**: Convergent evidence reduces false positives
- **Scientific Rigor**: Reproducible methodology with validation

### **Technical Innovation**
- **Convergent Scoring**: Novel approach to archaeological site confidence
- **AI-Enhanced Analysis**: GPT integration for pattern interpretation
- **Modular Architecture**: Extensible framework for continued development

**Note**: This system represents one approach among multiple teams working on the OpenAI to Z Challenge. Success will depend on execution quality and the strength of discoveries rather than technological claims.

---

## 🚀 Get Started

### **Competition Participants**
```bash
git clone https://github.com/stawils/amazon-discovery.git
cd amazon-discovery
pip install -r requirements.txt
cp .env.template .env
# Add OpenAI API key
python main.py --all-checkpoints
```

### **Researchers**
```bash
# Explore target zones and methodology
python main.py --list-zones
python main.py --pipeline --zones negro_madeira --visualize
```

### **Developers**
```bash
# Test individual components
python main.py --checkpoint 1
python -m src.checkpoints.validator
```

---

**🌟 A systematic approach to Amazon archaeological discovery through convergent evidence analysis** 🏛️

*Combining historical intelligence, modern remote sensing, and AI analysis to identify potential archaeological sites in the Amazon rainforest.*