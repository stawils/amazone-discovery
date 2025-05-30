# Amazon Archaeological Discovery Pipeline - Testing Guide

## 🧪 How to Test the System

### Quick Test (5 minutes) - Synthetic Data

**Step 1: Run the synthetic test**
```bash
python run_test.py
```

This will:
- Install basic packages (numpy, matplotlib, opencv, etc.)
- Generate synthetic Amazon landscape with archaeological features
- Run the complete detection pipeline
- Create visualizations and report

**Expected Output:**
- `test_pipeline_results.png` - 6-panel visualization showing detection results
- `test_pipeline_report.json` - Detailed analysis report
- Anomaly score between 8-12 points (varies due to randomness)
- Classification: "PROBABLE ARCHAEOLOGICAL FEATURE" or "HIGH CONFIDENCE SITE"

### Full Test (30 minutes) - Real USGS Data

**Step 1: Get USGS Credentials**
1. Go to https://ers.cr.usgs.gov/profile/access
2. Create account and request M2M API access
3. Generate API token

**Step 2: Setup Environment**
```bash
# Copy template and add your credentials
cp .env.template .env

# Edit .env file:
USGS_USERNAME=your_username
USGS_TOKEN=your_m2m_token
```

**Step 3: Install Full Requirements**
```bash
pip install -r requirements.txt
```

**Step 4: Run Pipeline on Negro-Madeira Target**
```bash
# List available zones
python main.py --list-zones

# Download satellite data for highest priority target
python main.py --zone negro_madeira --download --max-scenes 2

# Run analysis
python main.py --zone negro_madeira --analyze

# Generate complete report
python main.py --zone negro_madeira --full-pipeline
```

## 📊 How the Detection Works

### 1. Satellite Data Processing
- **Input**: Landsat 8/9 Surface Reflectance (6 bands: Blue, Green, Red, NIR, SWIR1, SWIR2)
- **Processing**: Calculate spectral indices (NDVI, Terra Preta Index, Clay Minerals)
- **Enhancement**: Contrast enhancement, noise reduction, cloud masking

### 2. Archaeological Feature Detection

#### Terra Preta Detection
```python
# Key formula:
terra_preta_index = (NIR - SWIR1) / (NIR + SWIR1)
ndvi = (NIR - Red) / (NIR + Red)

# Detection criteria:
terra_preta_signature = (terra_preta_index > 0.1) & (ndvi > 0.3)
```

#### Geometric Pattern Detection
```python
# Computer vision algorithms:
circles = cv2.HoughCircles()  # Detect circular earthworks
lines = cv2.HoughLinesP()     # Detect linear causeways
contours = cv2.findContours() # Detect rectangular compounds
```

### 3. Convergent Anomaly Scoring (0-15 points)
- **Historical Evidence**: +2 points (documented expeditions)
- **Geometric Patterns**: +3 points each (max 6)
- **Terra Preta Signatures**: +2 points (anthropogenic soils)
- **Environmental Suitability**: +1 point (rivers, elevation)
- **Priority Zone Bonus**: +1 point (Priority 1 zones)
- **Convergence Bonus**: +1-3 points (multiple evidence types)

### 4. Classification Thresholds
- **10+ points**: HIGH CONFIDENCE SITE → Ground verification
- **7-9 points**: PROBABLE FEATURE → High-res imagery + reconnaissance
- **4-6 points**: POSSIBLE ANOMALY → Additional analysis
- **0-3 points**: NATURAL VARIATION → Continue monitoring

## 🎯 Expected Results for Target Zones

### Negro-Madeira Confluence (Priority 1)
- **Expected Score**: 10-13 points
- **Evidence**: Orellana 1542 battle site + river confluence + high terra preta probability
- **Features**: Large circular earthworks (200-400m), multiple terra preta patches
- **Classification**: HIGH CONFIDENCE ARCHAEOLOGICAL SITE
- **Recommendation**: Immediate ground verification expedition

### Trombetas Junction (Priority 1)  
- **Expected Score**: 9-12 points
- **Evidence**: Amazon warrior encounter + eastern Amazon optimal zone
- **Features**: Fortified settlements, 100-300m diameter earthworks
- **Classification**: PROBABLE to HIGH CONFIDENCE
- **Recommendation**: High-resolution imagery + ground reconnaissance

## 🔍 Understanding the Visualizations

### RGB Composite
- Natural color view of the landscape
- Green = dense forest, Brown = clearings, Blue = water

### False Color (NIR-Red-Green)
- Red = healthy vegetation (high NIR)
- Dark areas = water or bare soil
- Bright red = very healthy/dense vegetation

### NDVI (Normalized Difference Vegetation Index)
- Green = healthy vegetation (0.3-0.8)
- Yellow = moderate vegetation (0.1-0.3)
- Red = bare soil or stressed vegetation (<0.1)

### Terra Preta Index
- Bright areas = potential anthropogenic soils
- Dark areas = natural forest soils
- Purple/pink = highest probability zones

### Detection Overlays
- **Red highlights**: Detected terra preta patches
- **Cyan circles**: Circular earthworks/settlements
- **Yellow lines**: Linear features (causeways, roads)

## 🚨 Troubleshooting

### Common Issues

**"No scenes found"**
- Check date range (try different years)
- Increase cloud cover threshold
- Verify coordinates are correct

**"Authentication failed"**
- Verify USGS username and token
- Check M2M API access is approved
- Token may need renewal

**"No features detected"**
- Try different detection parameters
- Check if image covers archaeological zone
- Verify cloud masking isn't too aggressive

**"ImportError: No module named..."**
- Install missing packages: `pip install package_name`
- Try installing from requirements.txt again
- Check Python version (3.8+ required)

### Performance Tips

**For faster testing:**
- Use smaller image regions
- Reduce number of scenes downloaded
- Lower image resolution for initial tests

**For better results:**
- Use dry season imagery (June-September)
- Multiple scenes for temporal analysis
- High-resolution imagery when available

## 📈 Success Metrics

### Synthetic Test Success
- ✅ Pipeline runs without errors
- ✅ Detects 2-3 synthetic features
- ✅ Scores 8+ points
- ✅ Creates visualizations

### Real Data Test Success  
- ✅ Downloads Landsat scenes successfully
- ✅ Detects terra preta signatures
- ✅ Identifies geometric patterns
- ✅ Generates interactive map
- ✅ Negro-Madeira scores 10+ points

## 🎉 Next Steps After Successful Testing

1. **Expand Analysis**: Test additional zones (Trombetas, Upper Xingu)
2. **Parameter Tuning**: Adjust detection thresholds based on results
3. **Validation**: Compare results with known archaeological sites
4. **Field Planning**: Organize ground verification for high-confidence sites
5. **Scale Up**: Process all 5 target zones systematically

## 📞 Support

If you encounter issues:
1. Check the log files in `results/` directory
2. Review `test_pipeline_report.json` for detailed diagnostics
3. Verify all dependencies are installed correctly
4. Check USGS API status and credentials

The system is designed to be robust and provide detailed error messages to help diagnose any issues.