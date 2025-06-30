
# Amazon Archaeological Discovery Pipeline - Documentation

**Technical documentation for the Amazon Archaeological Discovery Pipeline system**

This directory contains comprehensive technical documentation organized by system components and development workflows. Use this guide to navigate to the specific documentation you need.

---

## 📖 Documentation Quick Start

### For New Developers
1. **[Getting Started Guide](development/getting-started.md)** - Complete developer onboarding
2. **[System Architecture](architecture/system-overview.md)** - Understand the overall system design
3. **[Core API Reference](api/core-api-reference.md)** - Essential API documentation

### For System Administration
1. **[Configuration Guide](configuration/configuration-guide.md)** - System configuration and setup
2. **[Deployment Guide](operations/deployment-guide.md)** - Production deployment
3. **[Troubleshooting Manual](operations/troubleshooting.md)** - Problem resolution

### For Advanced Users
1. **[Performance Optimization](operations/performance-tuning.md)** - System tuning and scaling
2. **[Visualization System](visualization/visualization-system.md)** - Interactive mapping and exports
3. **[Provider Development](providers/provider-development.md)** - Custom data provider creation

---

## 📋 Directory Structure

```
repo-documentation/
├── README.md                             # Documentation navigation (this file)
├── api/                                  # API reference documentation
│   └── core-api-reference.md             # Core system API documentation
├── architecture/                         # System architecture guides
│   └── system-overview.md                # Complete architectural overview
├── configuration/                        # Configuration management
│   └── configuration-guide.md            # System configuration guide
├── detectors/                            # Detection systems
│   ├── sentinel2.md                      # Sentinel-2 detector documentation
│   └── gedi-detector.md                  # GEDI LiDAR detector documentation
├── providers/                            # Data acquisition systems
│   └── provider-development.md           # Custom data provider development
├── pipeline/                             # Pipeline orchestration
│   ├── modular-pipeline.md               # Main pipeline orchestrator
│   ├── analysis-coordination.md          # Multi-sensor analysis coordination
│   └── export-management.md              # Output and export systems
├── checkpoints/                          # OpenAI integration system  
│   └── checkpoint-system.md              # OpenAI framework and checkpoints
├── validation/                           # Quality assurance systems
│   └── validation-system-documentation.md  # Data validation and quality control
├── visualization/                        # Interactive mapping and exports
│   └── visualization-system.md           # Interactive mapping and visualization
├── development/                          # Developer resources
│   └── getting-started.md                # Developer onboarding and setup
└── operations/                           # Deployment and maintenance
    ├── deployment-guide.md               # Production deployment guide
    ├── performance-tuning.md             # Performance optimization guide
    └── troubleshooting.md                # Troubleshooting and problem resolution
```

---

## 🔧 System Components

### Core Systems
- **[Core API Reference](api/core-api-reference.md)** - Complete API documentation for config, data objects, pipeline, analysis, and export systems
- **[System Architecture](architecture/system-overview.md)** - Architectural overview with data flow diagrams and component interactions
- **[Configuration Guide](configuration/configuration-guide.md)** - System configuration, target zones, and environment setup

### Data Processing
- **[GEDI Detector](detectors/gedi-detector.md)** - Space-based LiDAR processing for archaeological feature detection
- **[Sentinel-2 Detector](detectors/sentinel2.md)** - Multispectral satellite imagery analysis for archaeological signatures
- **[Provider Development](providers/provider-development.md)** - Framework for developing custom data providers

### Pipeline Systems
- **[Modular Pipeline](pipeline/modular-pipeline.md)** - Four-stage archaeological analysis workflow orchestration
- **[Analysis Coordination](pipeline/analysis-coordination.md)** - Multi-sensor analysis coordination and detector dispatch
- **[Export Management](pipeline/export-management.md)** - Unified export system for GeoJSON and visualization outputs

### Advanced Features
- **[OpenAI Integration](checkpoints/checkpoint-system.md)** - AI-enhanced archaeological interpretation framework
- **[Visualization System](visualization/visualization-system.md)** - Interactive mapping and archaeological visualization tools
- **[Validation System](validation/validation-system-documentation.md)** - Data quality assurance and validation frameworks

---

## 🔍 Finding Information

### By Use Case
- **Setting up the system**: [Configuration Guide](configuration/configuration-guide.md) → [Getting Started](development/getting-started.md)
- **Understanding the architecture**: [System Architecture](architecture/system-overview.md) → [Pipeline Systems](#pipeline-systems)
- **Developing custom components**: [Provider Development](providers/provider-development.md) → [Core API Reference](api/core-api-reference.md)
- **Deploying to production**: [Deployment Guide](operations/deployment-guide.md) → [Performance Optimization](operations/performance-tuning.md)
- **Troubleshooting issues**: [Troubleshooting Manual](operations/troubleshooting.md) → [Configuration Guide](configuration/configuration-guide.md)

### By System Component
- **Data Acquisition**: [Provider Development](providers/provider-development.md)
- **Archaeological Detection**: [GEDI Detector](detectors/gedi-detector.md), [Sentinel-2 Detector](detectors/sentinel2.md)
- **Analysis Pipeline**: [Modular Pipeline](pipeline/modular-pipeline.md), [Analysis Coordination](pipeline/analysis-coordination.md)
- **Results and Visualization**: [Export Management](pipeline/export-management.md), [Visualization System](visualization/visualization-system.md)
- **Quality Assurance**: [Validation System](validation/validation-system-documentation.md)

---

## 🔗 External Documentation

### Project Documentation
- **[Main README](../../README.md)** - Project overview and capabilities
- **[Developer Guide](../../CLAUDE.md)** - Development workflows and system architecture

### Related Resources
- **Configuration Files**: `config/` directory in project root
- **API Code**: Source code in `src/` directory provides implementation details

---

## 🚀 Getting Started

### New to the System?
1. Start with **[Getting Started Guide](development/getting-started.md)** for developer setup
2. Read **[System Architecture](architecture/system-overview.md)** to understand the design
3. Review **[Configuration Guide](configuration/configuration-guide.md)** for system setup

### Need Specific Information?
Use the **[Finding Information](#-finding-information)** section above to navigate to the right documentation for your needs.

### Contributing to Documentation?
All documentation follows consistent patterns - review existing files for style and structure before contributing new content.