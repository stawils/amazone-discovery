#!/usr/bin/env python3
"""
Amazon Archaeological Discovery - Main Entry Point
Simplified routing between modular pipeline and OpenAI checkpoints
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, str(Path(__file__).parent))  # Ensure src is in path\
    
# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import TARGET_ZONES, RESULTS_DIR

# Setup logging
def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    log_file = RESULTS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return log_file

logger = logging.getLogger(__name__)

def list_target_zones():
    """Display all configured target zones"""
    print("\n🎯 AMAZON ARCHAEOLOGICAL DISCOVERY - TARGET ZONES")
    print("=" * 60)
    
    # Sort by priority
    sorted_zones = sorted(TARGET_ZONES.items(), key=lambda x: x[1].priority)
    
    for zone_id, zone in sorted_zones:
        print(f"\n📍 {zone.name.upper()}")
        print(f"   ID: {zone_id}")
        print(f"   Coordinates: {zone.center[0]:.4f}°, {zone.center[1]:.4f}°")
        print(f"   Priority: {zone.priority} {'⭐' * (4 - zone.priority)}")
        print(f"   Expected: {zone.expected_features}")
        print(f"   Evidence: {zone.historical_evidence}")
        print(f"   Search Area: {zone.search_radius_km} km radius")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total zones: {len(TARGET_ZONES)}")
    print(f"   Priority 1 (Highest): {sum(1 for z in TARGET_ZONES.values() if z.priority == 1)}")
    print(f"   Priority 2 (High): {sum(1 for z in TARGET_ZONES.values() if z.priority == 2)}")
    print(f"   Priority 3 (Medium): {sum(1 for z in TARGET_ZONES.values() if z.priority == 3)}")

def run_checkpoint(checkpoint_num: int, **kwargs):
    """Execute specific OpenAI checkpoint"""
    
    try:
        from openai_checkpoints import CheckpointRunner
        
        runner = CheckpointRunner()
        result = runner.run(checkpoint_num, **kwargs)
        
        if result.get('success'):
            print(f"\n✅ Checkpoint {checkpoint_num} completed successfully!")
            print(f"Results saved in: {runner.checkpoint_dir}")
        else:
            print(f"\n❌ Checkpoint {checkpoint_num} failed: {result.get('error', 'Unknown error')}")
            
        return result
        
    except ImportError as e:
        print(f"❌ Error importing checkpoint system: {e}")
        print("Make sure OpenAI API key is set in .env file")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running checkpoint {checkpoint_num}: {e}")
        sys.exit(1)

def run_modular_pipeline(**kwargs):
    """Execute the modular archaeological discovery pipeline"""
    
    try:
        from src.pipeline.modular_pipeline import ModularPipeline
        
        # Extract parameters
        zones = kwargs.get('zones')
        provider = kwargs.get('provider', 'gee')
        max_scenes = kwargs.get('max_scenes', 3)
        
        print(f"\n🚀 Starting Modular Archaeological Discovery Pipeline...")
        print(f"Provider: {provider}")
        print(f"Zones: {zones or 'Priority zones'}")
        print(f"Max scenes per zone: {max_scenes}")
        
        # Initialize and run pipeline
        pipeline = ModularPipeline(provider=provider)
        results = pipeline.run(zones=zones, max_scenes=max_scenes)
        
        # Print summary
        scene_count = len(results.get('scene_data', []))
        analysis_zones = len(results.get('analysis', {}))
        scoring_zones = len(results.get('scores', {}))
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Summary:")
        print(f"   Scenes processed: {scene_count}")
        print(f"   Zones analyzed: {analysis_zones}")
        print(f"   Zones scored: {scoring_zones}")
        
        if results.get('map_path'):
            print(f"   Interactive map: {results['map_path']}")
        
        if results.get('report'):
            session_id = results['report'].get('session_info', {}).get('session_id', 'unknown')
            print(f"   Report session: {session_id}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Error importing pipeline modules: {e}")
        print("Check that all required modules are properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

def run_all_checkpoints(**kwargs):
    """Run all 5 OpenAI checkpoints in sequence"""
    
    print("\n🎯 Running All OpenAI to Z Challenge Checkpoints")
    print("=" * 60)
    
    try:
        from openai_checkpoints import CheckpointRunner
        runner = CheckpointRunner()
        
        results = {}
        
        for i in range(1, 6):
            print(f"\n{'='*40}")
            print(f"CHECKPOINT {i}")
            print(f"{'='*40}")
            
            try:
                result = runner.run(i, **kwargs)
                results[f"checkpoint_{i}"] = result
                
                if result.get('success'):
                    print(f"✅ Checkpoint {i} completed")
                else:
                    print(f"❌ Checkpoint {i} failed: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ Checkpoint {i} error: {e}")
                results[f"checkpoint_{i}"] = {'success': False, 'error': str(e)}
        
        # Summary
        successful = sum(1 for r in results.values() if r.get('success'))
        print(f"\n🏁 FINAL SUMMARY:")
        print(f"Checkpoints completed: {successful}/5")
        print(f"Results directory: {runner.checkpoint_dir}")
        
        if successful == 5:
            print("🏆 All checkpoints completed! Ready for competition submission.")
        else:
            print("⚠️ Some checkpoints failed. Check logs for details.")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running checkpoints: {e}")
        sys.exit(1)

def main():
    """Main entry point with simplified argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Amazon Archaeological Discovery Pipeline - OpenAI to Z Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available zones
  python main.py --list-zones
  
  # Run specific OpenAI checkpoint
  python main.py --checkpoint 1
  python main.py --checkpoint 3 --zone negro_madeira
  
  # Run all checkpoints in sequence
  python main.py --all-checkpoints
  
  # Run modular pipeline
  python main.py --pipeline --zones negro_madeira trombetas
  python main.py --pipeline --provider gee --full
  
  # Quick start for competition
  python main.py --competition-ready
        """
    )
    
    # Main operation modes
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--list-zones', action='store_true',
                          help='List all configured target zones')
    operation.add_argument('--checkpoint', type=int, choices=[1,2,3,4,5],
                          help='Run specific OpenAI checkpoint (1-5)')
    operation.add_argument('--all-checkpoints', action='store_true',
                          help='Run all 5 OpenAI checkpoints in sequence')
    operation.add_argument('--pipeline', action='store_true',
                          help='Run modular archaeological discovery pipeline')
    operation.add_argument('--competition-ready', action='store_true',
                          help='Run everything needed for competition submission')
    
    # Zone and provider options
    parser.add_argument('--zones', nargs='+', 
                       choices=list(TARGET_ZONES.keys()) + ['all'],
                       help='Target zones to process (default: priority zones)')
    parser.add_argument('--zone', type=str, choices=list(TARGET_ZONES.keys()),
                       help='Single target zone (for checkpoints)')
    parser.add_argument('--provider', choices=['gee'], default='gee',
                       help='Data provider (only gee is supported)')
    
    # Pipeline options
    parser.add_argument('--max-scenes', type=int, default=3,
                       help='Maximum scenes per zone (default: 3)')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline with all steps')
    
    # System options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.verbose)
    
    # Prepare keyword arguments
    kwargs = {
        'zones': args.zones,
        'zone': args.zone or 'negro_madeira',  # Default zone for checkpoints
        'provider': args.provider,
        'max_scenes': args.max_scenes,
        'verbose': args.verbose
    }
    
    # Execute based on operation mode
    try:
        if args.list_zones:
            list_target_zones()
            
        elif args.checkpoint:
            print(f"🎯 Running OpenAI to Z Challenge Checkpoint {args.checkpoint}")
            run_checkpoint(args.checkpoint, **kwargs)
            
        elif args.all_checkpoints:
            run_all_checkpoints(**kwargs)
            
        elif args.pipeline:
            if args.full:
                print("🚀 Running Full Modular Pipeline")
            else:
                print("🚀 Running Modular Pipeline")
            run_modular_pipeline(**kwargs)
            
        elif args.competition_ready:
            print("🏆 COMPETITION-READY EXECUTION")
            print("This will run all checkpoints + full pipeline")
            
            # First run all checkpoints
            print("\n📋 Phase 1: Running all OpenAI checkpoints...")
            checkpoint_results = run_all_checkpoints(**kwargs)
            
            # Then run full pipeline for additional analysis
            print("\n📋 Phase 2: Running full pipeline for comprehensive analysis...")
            pipeline_results = run_modular_pipeline(**kwargs)
            
            print("\n🎉 COMPETITION SUBMISSION READY!")
            print("All checkpoints completed and full analysis available.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    print(f"\n📋 Session log saved: {log_file}")

if __name__ == "__main__":
    main()