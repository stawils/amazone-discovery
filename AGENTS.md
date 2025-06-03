# Amazon Archaeological Discovery - Agent Guide

## Project Overview

The Amazon Archaeological Discovery Pipeline is an AI-enhanced satellite remote sensing system for archaeological site discovery in the Amazon rainforest. It implements a convergent anomaly detection methodology, combining multiple independent evidence sources to identify potential archaeological sites with unprecedented accuracy.

## Repository Structure

```
amazon-discovery/
├── src/
│   ├── core/                      # Core algorithms and config
│   │   ├── config.py              # Target zones and configuration
│   │   ├── data_objects.py        # SceneData and provider interfaces
│   │   ├── scoring.py             # Convergent anomaly scoring
│   │   ├── processors.py          # Image processing utilities
│   │   ├── visualizers.py         # Interactive mapping and plots
│   │   └── detectors/             # Archaeological detection algorithms
│   │       ├── gee_detectors.py   # Google Earth Engine optimized
│   │       └── sentinel2_detector.py # Sentinel-2 enhanced detection
│   ├── providers/                 # Satellite data providers
│   │   ├── gee_provider.py        # Google Earth Engine
│   │   └── sentinel2_provider.py  # Sentinel-2 AWS access
│   ├── pipeline/                  # Modular pipeline components
│   │   ├── modular_pipeline.py    # Main pipeline orchestrator
│   │   ├── analysis.py            # Feature analysis step
│   │   ├── scoring.py             # Scoring step
│   │   ├── report.py              # Report generation
│   │   └── visualization.py       # Map creation step
│   ├── checkpoints/               # OpenAI to Z Challenge checkpoints
│   │   ├── validator.py           # Competition validation
│   │   └── checkpoint[1-5].py     # Individual checkpoint classes
│   └── utils/                     # Helper utilities
├── main.py                        # Main entry point
├── openai_checkpoints.py          # Competition checkpoint runner
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup script
├── .env.template                  # Environment configuration template
├── data/                          # Downloaded satellite data
├── results/                       # Analysis results and outputs
└── notebooks/                     # Jupyter analysis notebooks
```

## Development Environment Setup

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/stawils/amazon-discovery.git
cd amazon-discovery

# Install dependencies
pip install -e .
# OR
python setup.py develop
```

### Required Environment Variables
Copy `.env.template` to `.env` and configure:
```
# OpenAI API (Required for competition)
OPENAI_API_KEY=your_openai_api_key

# Google Earth Engine (Optional but recommended)
GEE_SERVICE_ACCOUNT_PATH=path/to/service_account.json
GEE_PROJECT_ID=your_gee_project

# Copernicus/Sentinel Data (Optional)
COPERNICUS_USER=your_username
COPERNICUS_PASSWORD=your_password
```

## Testing Instructions

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_detectors.py

# Run with coverage
pytest --cov=src
```

### Validation
```bash
# Validate competition checkpoints
python openai_checkpoints.py --validate

# Run specific checkpoint
python openai_checkpoints.py --checkpoint 1
```

## Code Style Guidelines

### Python Style
- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Document classes and functions using docstrings (Google style)
- Maximum line length: 100 characters

### Linting and Formatting
```bash
# Run linting
flake8 src

# Format code
black src
```

## Core Components

### Data Providers
Data providers in `src/providers/` handle satellite imagery acquisition from different sources. When working with these:
- Ensure proper authentication handling
- Implement caching for downloaded data
- Follow the provider interface defined in `src/core/data_objects.py`

### Detection Algorithms
Detection algorithms in `src/core/detectors/` implement feature extraction from satellite imagery:
- Each detector should implement the base detector interface
- Optimize for both accuracy and performance
- Include proper documentation of detection parameters
- Return standardized feature objects

### Scoring System
The scoring system in `src/core/scoring.py` implements the 15-point convergent anomaly scale:
- Maintain the weighted scoring approach
- Document any changes to the scoring algorithm
- Ensure backward compatibility with existing results

## Pull Request Guidelines

### PR Format
Title: `[Component] Brief description of changes`

Description template:
```
## Changes
Brief description of the changes made

## Testing
How the changes were tested

## Related Issues
Links to related issues or tickets
```

### Review Process
- All PRs require at least one review
- Include test coverage for new features
- Ensure all tests pass before merging
- Update documentation as needed

## Common Tasks

### Adding a New Target Zone
Add to `src/core/config.py`:
```python
TARGET_ZONES['new_zone'] = TargetZone(
    name="New Archaeological Zone",
    center=(lat, lon),
    bbox=(min_lat, min_lon, max_lat, max_lon),
    priority=priority_level,
    min_feature_size_m=min_size,
    expected_features=["feature1", "feature2"]
)
```

### Adding a New Detector
1. Create a new file in `src/core/detectors/`
2. Implement the detector interface
3. Register in the detector factory
4. Add tests in `tests/test_detectors.py`

### Extending the Pipeline
1. Add new module in `src/pipeline/`
2. Update `src/pipeline/modular_pipeline.py` to include the new step
3. Add configuration options in `main.py`

## Troubleshooting

### Common Issues
- **Missing dependencies**: Ensure all requirements are installed
- **API authentication errors**: Check `.env` configuration
- **Memory errors**: Reduce batch size or scene dimensions
- **Missing data**: Check data provider configuration

### Debug Mode
```bash
# Run with debug logging
python main.py --debug

# Run specific zone with verbose output
python main.py --zone negro_madeira --verbose
```

## Performance Considerations

- Satellite imagery processing is memory-intensive
- Use batch processing for large areas
- Consider using cloud processing for production runs
- Cache intermediate results when possible

## Additional Resources

- [Project Documentation](https://github.com/stawils/amazon-discovery/wiki)
- [OpenAI to Z Challenge Guidelines](https://kaggle.com/competitions/openai-to-z-challenge)
- [Sentinel-2 Documentation](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)
- [Google Earth Engine API](https://developers.google.com/earth-engine/guides)
