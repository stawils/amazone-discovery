# 🏗️ Proposed New Project Structure

## MAIN DIRECTORIES
```
amazon-discovery/
├── src/
│   ├── core/                    # Core functionality (keep existing)
│   │   ├── config.py
│   │   ├── data_objects.py
│   │   ├── detectors.py
│   │   ├── scoring.py
│   │   ├── visualizers.py
│   │   └── processors.py
│   ├── providers/               # Data providers (reorganized)
│   │   ├── __init__.py
│   │   ├── usgs_provider.py     # (renamed from usgs_api.py)
│   │   ├── gee_provider.py
│   │   └── sentinel2_provider.py # (future addition)
│   ├── checkpoints/             # NEW: OpenAI checkpoint system
│   │   ├── __init__.py
│   │   ├── base_checkpoint.py
│   │   ├── checkpoint1.py       # Familiarize + OpenAI test
│   │   ├── checkpoint2.py       # Early explorer
│   │   ├── checkpoint3.py       # Site discovery
│   │   ├── checkpoint4.py       # Story & impact
│   │   └── checkpoint5.py       # Final submission
│   ├── pipeline/                # Simplified pipeline (rename pipeline_steps/)
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── scoring.py
│   │   ├── reporting.py
│   │   └── visualization.py
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── drive_downloader.py
├── main.py                      # Simplified entry point
├── openai_checkpoints.py        # NEW: OpenAI checkpoint runner
├── requirements.txt
├── setup.py
├── .env.template
└── README.md
```

## KEY CHANGES

### 1. **Eliminate Dual Pipeline**
- Remove `ArchaeologicalPipeline` class from main.py
- Keep only `ModularPipeline` approach
- Simplify main.py to just CLI + checkpoint routing

### 2. **Add OpenAI Checkpoint System**
```python
# openai_checkpoints.py - NEW FILE
class CheckpointRunner:
    def run_checkpoint1(self):
        """Familiarize yourself with challenge and data"""
        # Download one LiDAR tile or Sentinel-2 scene
        # Run single OpenAI prompt
        # Print model version and dataset ID
        
    def run_checkpoint2(self):
        """Early explorer - multiple data types"""
        # Load two independent sources
        # Produce 5 anomaly footprints
        # Log all dataset IDs and prompts
        
    def run_checkpoint3(self):
        """New Site Discovery"""
        # Single best site with evidence
        # Algorithmic detection + historical cross-reference
        # Compare to known archaeological feature
        
    def run_checkpoint4(self):
        """Story & impact draft"""
        # Two-page PDF with cultural context
        # Hypotheses for function/age
        # Proposed survey effort
        
    def run_checkpoint5(self):
        """Final submission"""
        # Everything above + polish
        # Livestream-ready presentation
```

### 3. **Reorganize by Function**
- `src/core/` - Core archaeological algorithms
- `src/providers/` - Data providers (GEE, future Sentinel-2)
- `src/checkpoints/` - OpenAI competition checkpoints
- `src/pipeline/` - Modular pipeline steps
- `src/utils/` - Helper utilities

### 4. **Simplified main.py**
```python
# New simplified main.py
def main():
    parser = argparse.ArgumentParser()
    
    # OpenAI checkpoint options
    parser.add_argument('--checkpoint', type=int, choices=[1,2,3,4,5],
                       help='Run specific OpenAI checkpoint')
    
    # Regular pipeline options (simplified)
    parser.add_argument('--zones', nargs='+', help='Target zones')
    parser.add_argument('--provider', choices=['usgs', 'gee'], default='usgs')
    parser.add_argument('--full-pipeline', action='store_true')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        from openai_checkpoints import CheckpointRunner
        runner = CheckpointRunner()
        runner.run(args.checkpoint)
    else:
        from src.pipeline.modular_pipeline import ModularPipeline
        pipeline = ModularPipeline(provider=args.provider)
        pipeline.run(zones=args.zones)

if __name__ == "__main__":
    main()
```

## MIGRATION STEPS

### Step 1: Create New Structure
1. Create `src/checkpoints/` directory
2. Create `openai_checkpoints.py`
3. Rename `src/pipeline_steps/` to `src/pipeline/`
4. Move providers to `src/providers/`

### Step 2: Implement Checkpoints
1. Build checkpoint base class
2. Implement each of the 5 OpenAI checkpoints
3. Add OpenAI API integration for text analysis

### Step 3: Simplify main.py
1. Remove `ArchaeologicalPipeline` class
2. Keep only CLI parsing + routing
3. Route to either checkpoints or modular pipeline

### Step 4: Test & Validate
1. Ensure all existing functionality works
2. Test checkpoint system
3. Verify OpenAI integration

## BENEFITS

1. **Competition Ready** - Matches OpenAI checkpoint structure exactly
2. **No Duplication** - Single pipeline approach
3. **Better Organization** - Clear separation by function
4. **Easier Extension** - Add new providers/checkpoints easily
5. **Maintainable** - Smaller, focused files