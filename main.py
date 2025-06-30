#!/usr/bin/env python3
"""
Amazon Archaeological Discovery - Main Entry Point
Simplified routing between modular pipeline and OpenAI checkpoints
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))  # Ensure src is in path\

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import TARGET_ZONES, RESULTS_DIR, SATELLITE_PROVIDERS, DEFAULT_PROVIDERS

# Import provider classes for instantiation
from src.providers.gedi_provider import GEDIProvider
from src.providers.sentinel2_provider import Sentinel2Provider

from src.pipeline.modular_pipeline import ModularPipeline
from src.checkpoints.validator import CheckpointValidator
from openai_checkpoints import CheckpointRunner

# Enable performance optimizations
try:
    from src.core.enable_optimizations import enable_pipeline_optimizations, check_optimization_requirements
    
    # Check what optimizations are available
    optimization_status = check_optimization_requirements()
    print(f"🔍 System capabilities: {optimization_status['estimated_speedup']}")
    print(f"   CPU cores: {optimization_status['system_info'].get('cpu_cores', 'unknown')}")
    print(f"   Available memory: {optimization_status['system_info'].get('memory', {}).get('available_gb', 'unknown'):.1f}GB")
    print(f"   GPU: {optimization_status['system_info'].get('gpu', 'unknown')}")
    
    # Enable optimizations with 16 workers for the 24-core system and GPU if available
    optimization_enabled = enable_pipeline_optimizations(use_gpu=True, max_workers=16)
    if optimization_enabled:
        print("✅ Performance optimizations enabled!")
    else:
        print("⚠️ Performance optimizations could not be enabled")
        
except ImportError as e:
    print(f"⚠️ Performance optimizations not available: {e}")
except Exception as e:
    print(f"⚠️ Error enabling optimizations: {e}")


# Setup logging
def setup_logging(verbose: bool = False, run_id: str = "unknown_run"):
    """Configure logging for the pipeline"""
    level = logging.DEBUG if verbose else logging.INFO

    # Get root logger and clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Console formatter - includes timestamps for better tracking
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File formatter - detailed with timestamps
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - clean format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler - detailed format
    run_log_dir = RESULTS_DIR / f"run_{run_id}" / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_log_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return log_file


logger = logging.getLogger(__name__)


def list_target_zones():
    """Display all configured target zones"""
    print("\n🌍 Available Target Zones:")
    print("-" * 50)
    print(f"{'Name':<20} {'Priority':<10} {'Expected Features':<30}")
    print("-" * 50)

    # Sort zones by priority
    sorted_zones = sorted(
        TARGET_ZONES.items(), key=lambda x: x[1].priority if hasattr(x[1], "priority") else 999
    )

    for zone_id, zone in sorted_zones:
        priority = getattr(zone, "priority", "Unknown")
        features = getattr(zone, "expected_features", "Unknown")
        print(f"{zone_id:<20} {priority:<10} {features:<30}")

    print("-" * 50)
    print(f"\n📊 SUMMARY:")
    print(f"   Total zones: {len(TARGET_ZONES)}")
    print(
        f"   Priority 1 (Highest): {sum(1 for z in TARGET_ZONES.values() if z.priority == 1)}"
    )
    print(
        f"   Priority 2 (High): {sum(1 for z in TARGET_ZONES.values() if z.priority == 2)}"
    )
    print(
        f"   Priority 3 (Medium): {sum(1 for z in TARGET_ZONES.values() if z.priority == 3)}"
    )


def normalize_zone_name(zone_name: str) -> str:
    """Normalize zone names for consistent directory structure."""
    normalized = zone_name.lower().replace(" ", "_").replace("-", "_")
    
    zone_mappings = {
        "upper_naporegion": "upper_napo",
        "uppernaporegion": "upper_napo", 
        "upper_napo_region": "upper_napo",
        "negro_madeira_confluence": "negro_madeira",
        "trombetas_river_junction": "trombetas",
        "upper_xingu_region": "upper_xingu",
        "maranon_river_system": "maranon"
    }
    
    return zone_mappings.get(normalized, normalized)


def run_checkpoint(checkpoint_num: int, run_id: str, **kwargs):
    """Execute specific OpenAI checkpoint"""

    try:
        runner = CheckpointRunner(run_id=run_id)
        result = runner.run(checkpoint_num, **kwargs)

        if result.get("success"):
            print(f"\n✅ Checkpoint {checkpoint_num} completed successfully!")
            print(f"Results saved in: {runner.checkpoint_dir}")
        else:
            print(
                f"\n❌ Checkpoint {checkpoint_num} failed: {result.get('error', 'Unknown error')}"
            )

        return result

    except ImportError as e:
        print(f"❌ Error importing checkpoint system: {e}")
        print("Make sure OpenAI API key is set in .env file")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running checkpoint {checkpoint_num}: {e}")
        sys.exit(1)


def run_modular_pipeline(provider_name: str, run_id: str, args: argparse.Namespace, total_providers: int = 1):
    """Runs the modular pipeline for a single specified provider."""
    logger.info(f"🔧 Configuring modular pipeline for provider: {provider_name} (Run ID: {run_id})")
    
    provider_name_lower = provider_name.lower()
    if provider_name_lower not in SATELLITE_PROVIDERS:
        logger.error(f"Unknown provider: {provider_name}. Available: {list(SATELLITE_PROVIDERS.keys())}")
        return

    try:
        provider_instance = SATELLITE_PROVIDERS[provider_name_lower]()
        logger.info(f"Instantiated provider: {provider_instance.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to instantiate provider {provider_name}: {e}", exc_info=True)
        return

    pipeline = ModularPipeline(provider_instance=provider_instance, run_id=run_id, total_providers=total_providers)

    # Handle both --zone (singular) and --zones (plural) arguments
    zones_to_process = None
    if args.zones:
        zones_to_process = args.zones
    elif args.zone:
        zones_to_process = [args.zone]  # Convert single zone to list

    if args.stage == "acquire_data":
        logger.info(f"▶️ Running stage: acquire_data for {provider_name}")
        pipeline.acquire_data(zones=zones_to_process, max_scenes=args.max_scenes)
    elif args.stage == "analyze_scenes":
        logger.info(f"▶️ Running stage: analyze_scenes for {provider_name}")
        if not args.input_scene_data:
            logger.error("--input-scene-data is required for 'analyze_scenes' stage.")
            return
        if not args.input_scene_data.is_file():
            logger.error(f"Input scene data file not found: {args.input_scene_data}")
            return
        pipeline.analyze_scenes(scene_data_input=args.input_scene_data)
    elif args.stage == "score_zones":
        logger.info(f"▶️ Running stage: score_zones for {provider_name}")
        if not args.input_analysis_results:
            logger.error("--input-analysis-results is required for 'score_zones' stage.")
            return
        if not args.input_analysis_results.is_file():
            logger.error(f"Input analysis results file not found: {args.input_analysis_results}")
            return
        pipeline.score_zones(analysis_results_input=args.input_analysis_results)
    elif args.stage == "generate_outputs":
        logger.info(f"▶️ Running stage: generate_outputs for {provider_name}")
        if not args.input_analysis_results:
            logger.error("--input-analysis-results is required for 'generate_outputs' stage.")
            return
        if not args.input_analysis_results.is_file():
            logger.error(f"Input analysis results file not found: {args.input_analysis_results}")
            return
        if not args.input_scoring_results:
            logger.error("--input-scoring-results is required for 'generate_outputs' stage.")
            return
        if not args.input_scoring_results.is_file():
            logger.error(f"Input scoring results file not found: {args.input_scoring_results}")
            return
        pipeline.generate_outputs(
            analysis_results_input=args.input_analysis_results,
            scoring_results_input=args.input_scoring_results
        )
    elif args.stage == "full":
        logger.info(f"▶️ Running full pipeline for {provider_name}")
        pipeline.run(zones=zones_to_process, max_scenes=args.max_scenes)
    else:
        logger.error(f"Unknown pipeline stage: {args.stage}")

    logger.info(f"🏁 Modular pipeline stage '{args.stage}' for provider {provider_name} completed processing.")


def run_all_checkpoints(run_id: str, **kwargs):
    """Run all 5 OpenAI checkpoints in sequence"""

    print("\n🎯 Running All OpenAI to Z Challenge Checkpoints")
    print("=" * 60)

    try:
        runner = CheckpointRunner(run_id=run_id)

        results = {}

        for i in range(1, 6):
            print(f"\n{'='*40}")
            print(f"CHECKPOINT {i}")
            print(f"{'='*40}")

            try:
                result = runner.run(i, **kwargs)
                results[f"checkpoint_{i}"] = result

                if result.get("success"):
                    print(f"✅ Checkpoint {i} completed")
                else:
                    print(f"❌ Checkpoint {i} failed: {result.get('error', 'Unknown')}")

            except Exception as e:
                print(f"❌ Checkpoint {i} error: {e}")
                results[f"checkpoint_{i}"] = {"success": False, "error": str(e)}

        # Summary
        successful = sum(1 for r in results.values() if r.get("success"))
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


def run_enhanced_checkpoints(run_id: str, model: str = "o1", temperature: float = 1.0, **kwargs):
    """Run SAAM-enhanced checkpoints with advanced AI integration"""

    print("\n🧠 Running SAAM-Enhanced OpenAI to Z Challenge Checkpoints")
    print("=" * 70)
    print(f"Model: {model} | Temperature: {temperature}")
    print("🚀 SAAM Cognitive Enhancement: ENABLED")
    print("=" * 70)

    try:
        from src.checkpoints.enhanced_checkpoint_runner import EnhancedCheckpointRunner
        
        enhanced_runner = EnhancedCheckpointRunner(
            run_id=run_id,
            model=model,
            temperature=temperature
        )

        zone_name = kwargs.get('zone')  # No default, let checkpoints handle it
        available_data_types = ["sentinel2", "gedi_lidar", "historical_records", "environmental_data"]

        results = enhanced_runner.run_all_checkpoints(
            zone_name=zone_name,
            available_data_types=available_data_types
        )

        # Summary
        successful_checkpoints = results.get('openai_interaction_summary', {}).get('successful_interactions', 0)
        total_checkpoints = len([k for k in results.get('checkpoint_results', {}).keys() if k.startswith('checkpoint_')])
        
        print(f"\n🎯 SAAM-ENHANCED SUMMARY:")
        print(f"Checkpoints completed: {total_checkpoints}/5")
        print(f"AI interactions: {results.get('openai_interaction_summary', {}).get('total_interactions', 0)}")
        print(f"Total tokens used: {results.get('openai_interaction_summary', {}).get('total_tokens_used', 0)}")
        print(f"Processing time: {results.get('overall_processing_time', 0):.2f}s")
        print(f"Results directory: {enhanced_runner.checkpoint_dir}")

        # Export competition documentation
        doc_file = enhanced_runner.export_competition_documentation()
        print(f"📋 Competition documentation: {doc_file}")

        if total_checkpoints == 5:
            print("🏆 All SAAM-enhanced checkpoints completed! Competition ready with AI documentation.")
        else:
            print("⚠️ Some enhanced checkpoints failed. Check logs for details.")

        return results

    except ImportError as e:
        print(f"❌ Error importing enhanced checkpoint system: {e}")
        print("Falling back to standard checkpoint system...")
        return run_all_checkpoints(run_id, **kwargs)
    except Exception as e:
        print(f"❌ Error running enhanced checkpoints: {e}")
        sys.exit(1)


def create_parser():
    """Create and configure the argument parser"""
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
  
  # Run SAAM-enhanced checkpoints with advanced AI
  python main.py --enhanced-checkpoints
  python main.py --enhanced-checkpoints --model o1 --temperature 0.7
  
  # Run modular pipeline
  python main.py --pipeline --zones negro_madeira trombetas
  python main.py --pipeline --provider gee --full
  
  # Skip OpenAI calls during testing
  python main.py --checkpoint 1 --no-openai
  python main.py --all-checkpoints --no-openai
  
  # Quick start for competition (uses SAAM enhancement)
  python main.py --competition-ready
        """,
    )

    # Main operation modes
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument(
        "--list-zones", action="store_true", help="List all configured target zones"
    )
    operation.add_argument(
        "--checkpoint",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific OpenAI checkpoint (1-5)",
    )
    operation.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Run all 5 OpenAI checkpoints in sequence",
    )
    operation.add_argument(
        "--enhanced-checkpoints",
        action="store_true",
        help="Run SAAM-enhanced checkpoints with advanced AI integration",
    )
    operation.add_argument(
        "--pipeline",
        action="store_true",
        help="Run modular archaeological discovery pipeline",
    )
    operation.add_argument(
        "--competition-ready",
        action="store_true",
        help="Run everything needed for competition submission",
    )

    # Zone and provider options
    parser.add_argument(
        "--zones",
        nargs="+",
        choices=list(TARGET_ZONES.keys()) + ["all"],
        help="Target zones to process (default: priority zones)",
    )
    parser.add_argument(
        "--zone",
        type=str,
        choices=list(TARGET_ZONES.keys()),
        help="Single target zone (for checkpoints)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        nargs="*",
        help="Specify provider(s) for the pipeline (e.g., gedi, sentinel2, gee). "
        "Can be one or more. If 'all' is specified or none are provided, "
        "it runs for all default providers.",
    )

    # Pipeline options
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=1,
        help="Maximum number of scenes to process per zone (default: 1 for resouces constrained testing).",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full pipeline with all steps"
    )

    # System options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    
    # Enhanced checkpoint options
    parser.add_argument(
        "--model",
        type=str,
        default="o1",
        help="OpenAI model for enhanced checkpoints (default: o1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for OpenAI model (default: 1.0)"
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Skip OpenAI model calls during testing and log accordingly"
    )

    # Arguments for staged pipeline execution
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["acquire_data", "analyze_scenes", "score_zones", "generate_outputs", "full"],
        help="Specify a single stage of the modular pipeline to run. "
             "Defaults to 'full' to run all stages. "
             "Applies when --mode is 'pipeline'.",
    )
    parser.add_argument(
        "--input-scene-data",
        type=Path,
        help="Path to the 'acquired_scene_data.json' file, "
             "required if running 'analyze_scenes' stage independently.",
    )
    parser.add_argument(
        "--input-analysis-results",
        type=Path,
        help="Path to the 'analysis_results.json' file, "
             "required if running 'score_zones' or 'generate_outputs' stages independently.",
    )
    parser.add_argument(
        "--input-scoring-results",
        type=Path,
        help="Path to the 'scoring_results.json' file, "
             "required if running 'generate_outputs' stage independently.",
    )
    parser.add_argument(
        "--checkpoint-num", type=int, help="Checkpoint number to run (if mode is 'checkpoint')."
    )
    parser.add_argument(
        "--validate-checkpoint-num",
        type=int,
        help="Checkpoint number to validate (if mode is 'validate'). Can be 'all'.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory of checkpoint results to validate (if mode is 'validate').",
    )

    return parser


def run_cross_provider_analysis(providers_completed: List[str], run_id: str, zone_name: str):
    """Run cross-provider analysis after multiple providers complete"""
    if len(providers_completed) < 2:
        logger.debug("Skipping cross-provider analysis - only one provider completed")
        return None
        
    logger.info("🏛️ Creating enhanced cross-provider archaeological analysis...")
    try:
        from src.pipeline.cross_provider_analysis import CrossProviderAnalyzer
        from src.visualization import ArchaeologicalMapGenerator
        
        # Run cross-provider convergence analysis
        analyzer = CrossProviderAnalyzer(run_id=run_id, results_dir=RESULTS_DIR)
        convergence_results = analyzer.analyze_convergence(zone_name, providers_completed)
        
        if convergence_results:
            logger.info(f"✅ Cross-provider analysis complete: {convergence_results.get('convergent_pairs', 0)} convergent pairs found")
            logger.info(f"🎯 Enhanced top candidates: {len(convergence_results.get('enhanced_top_candidates', []))} sites")
            
            # Generate final unified map with all features
            logger.info("🗺️ Generating final unified archaeological map...")
            try:
                map_generator = ArchaeologicalMapGenerator(run_id=run_id, results_dir=RESULTS_DIR)
                final_map_path = map_generator.generate_enhanced_map(
                    zone_name=zone_name,
                    theme="professional",
                    include_analysis=True,
                    interactive_features=True
                )
                
                if final_map_path:
                    logger.info(f"✅ Final unified map generated: {final_map_path}")
                    convergence_results['final_map_path'] = str(final_map_path)
                else:
                    logger.warning("❌ Failed to generate final unified map")
                    
            except Exception as e:
                logger.error(f"❌ Error generating final map: {e}")
            
            return convergence_results
        else:
            logger.warning("❌ Cross-provider analysis failed")
            return None
            
    except ImportError:
        logger.warning("Cross-provider analyzer not available - using individual provider results")
        return None
    except Exception as e:
        logger.error(f"❌ Error in cross-provider analysis: {e}")
        return None


def main():
    """Main entry point for the Amazon Archaeological Discovery pipeline"""

    parser = create_parser()
    args = parser.parse_args()

    # Generate a unique run ID at the beginning, including zone if specified
    base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add zone suffix to run ID for better organization
    zone_suffix = ""
    if args.zone:
        zone_suffix = f"_{args.zone}"
    elif args.zones and len(args.zones) == 1:
        zone_suffix = f"_{args.zones[0]}"
    elif args.zones and len(args.zones) > 1:
        zone_suffix = f"_multi_{len(args.zones)}zones"
    
    RUN_ID = base_run_id + zone_suffix

    log_file_path = setup_logging(args.verbose, run_id=RUN_ID) # Corrected to get path
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Amazon Archaeological Discovery Pipeline")
    logger.info(f"Run ID: {RUN_ID}")
    logger.info(f"Log file: {log_file_path}") # Log the actual file path
    logger.info(f"Arguments: {args}")

    # Ensure RESULTS_DIR exists (it should be created by config.py, but double-check)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Prepare keyword arguments - these might be deprecated if functions take run_id directly
    # For now, ensure run_id is passed explicitly
    pipeline_kwargs = args.__dict__.copy()
    pipeline_kwargs['run_id'] = RUN_ID

    # Execute based on operation mode
    if args.pipeline:
        # Determine which provider(s) to run
        providers_to_run = args.provider
        if not providers_to_run: # None provided, use defaults
            providers_to_run = DEFAULT_PROVIDERS
        elif "all" in [p.lower() for p in providers_to_run]: # 'all' keyword used
            providers_to_run = DEFAULT_PROVIDERS
        
        if not providers_to_run:
            logger.error("No providers specified and no default providers configured. Exiting.")
            return

        logger.info(f"🚀 Will run Modular Pipeline for provider(s): {providers_to_run}")
        for provider_name in providers_to_run:
            run_modular_pipeline(provider_name=provider_name, run_id=RUN_ID, args=args)
        
        # After all providers complete, create enhanced cross-provider analysis
        if len(providers_to_run) > 1:  # Only for multi-provider runs
            # Determine the zone being processed
            zone_to_process = None
            if args.zones and len(args.zones) == 1:
                zone_to_process = args.zones[0]
            elif args.zone:
                zone_to_process = args.zone
            else:
                # Default to first priority zone
                zone_to_process = next(iter(TARGET_ZONES.keys()))
            
            # Run shared cross-provider analysis function
            run_cross_provider_analysis(providers_to_run, RUN_ID, zone_to_process)

    elif args.checkpoint is not None:
        if args.no_openai:
            print(f"🚫 Skipping OpenAI Checkpoint {args.checkpoint} (--no-openai flag enabled)")
            print(f"🔧 Running core archaeological pipeline for zone {args.zone or 'default'} instead")
            logger.info(f"OpenAI checkpoint {args.checkpoint} skipped due to --no-openai flag - running core pipeline")
            
            # Run the core pipeline instead of checkpoint
            providers_to_run = DEFAULT_PROVIDERS or ['gedi', 'sentinel2']
            logger.info(f"🚀 Running core pipeline for provider(s): {providers_to_run}")
            
            for provider_name in providers_to_run:
                # Create a mock args object for pipeline execution
                pipeline_args = argparse.Namespace()
                pipeline_args.zones = [args.zone] if args.zone else None
                pipeline_args.zone = args.zone
                pipeline_args.max_scenes = 1
                pipeline_args.stage = "full"
                pipeline_args.input_scene_data = None
                pipeline_args.input_analysis_results = None
                pipeline_args.input_scoring_results = None
                
                run_modular_pipeline(provider_name=provider_name, run_id=RUN_ID, args=pipeline_args, total_providers=len(providers_to_run))
            
            # CRITICAL FIX: Add cross-provider analysis after all providers complete
            if len(providers_to_run) > 1:
                zone_to_process = args.zone or next(iter(TARGET_ZONES.keys()))
                run_cross_provider_analysis(providers_to_run, RUN_ID, zone_to_process)
        else:
            print(f"🎯 Running OpenAI to Z Challenge Checkpoint {args.checkpoint}")
            # Pass relevant args to run_checkpoint - handle both --zone and --zones
            if args.zones:
                checkpoint_kwargs = {'zones': args.zones}
            elif args.zone:
                checkpoint_kwargs = {'zone': args.zone}
            else:
                checkpoint_kwargs = {'zone': None}  # No zone specified
            run_checkpoint(checkpoint_num=args.checkpoint, run_id=RUN_ID, **checkpoint_kwargs)

    elif args.all_checkpoints:
        if args.no_openai:
            print("🚫 Skipping All OpenAI Checkpoints (--no-openai flag enabled)")
            print(f"🔧 Running core archaeological pipeline for zone {args.zone or 'default'} instead")
            logger.info("All OpenAI checkpoints skipped due to --no-openai flag - running core pipeline")
            
            # Run the core pipeline instead of checkpoints
            providers_to_run = DEFAULT_PROVIDERS or ['gedi', 'sentinel2']
            logger.info(f"🚀 Running core pipeline for provider(s): {providers_to_run}")
            
            for provider_name in providers_to_run:
                # Create a mock args object for pipeline execution
                pipeline_args = argparse.Namespace()
                pipeline_args.zones = [args.zone] if args.zone else args.zones
                pipeline_args.zone = args.zone
                pipeline_args.max_scenes = 1
                pipeline_args.stage = "full"
                pipeline_args.input_scene_data = None
                pipeline_args.input_analysis_results = None
                pipeline_args.input_scoring_results = None
                
                run_modular_pipeline(provider_name=provider_name, run_id=RUN_ID, args=pipeline_args, total_providers=len(providers_to_run))
            
            # CRITICAL FIX: Add cross-provider analysis after all providers complete
            if len(providers_to_run) > 1:
                zone_to_process = args.zone or next(iter(TARGET_ZONES.keys()))
                run_cross_provider_analysis(providers_to_run, RUN_ID, zone_to_process)
        else:
            print("🎯 Running All OpenAI to Z Challenge Checkpoints")
            # Pass relevant args - handle both --zone and --zones for all checkpoints
            if args.zones:
                checkpoint_kwargs = {'zones': args.zones}
            elif args.zone:
                checkpoint_kwargs = {'zone': args.zone}
            else:
                checkpoint_kwargs = {'zone': None}  # No zone specified
            run_all_checkpoints(run_id=RUN_ID, **checkpoint_kwargs)

    elif args.enhanced_checkpoints:
        if args.no_openai:
            print("🚫 Skipping SAAM-Enhanced OpenAI Checkpoints (--no-openai flag enabled)")
            print(f"🔧 Running core archaeological pipeline for zone {args.zone or 'default'} instead")
            logger.info("SAAM-Enhanced OpenAI checkpoints skipped due to --no-openai flag - running core pipeline")
            
            # Run the core pipeline instead of enhanced checkpoints
            providers_to_run = DEFAULT_PROVIDERS or ['gedi', 'sentinel2']
            logger.info(f"🚀 Running core pipeline for provider(s): {providers_to_run}")
            
            for provider_name in providers_to_run:
                # Create a mock args object for pipeline execution
                pipeline_args = argparse.Namespace()
                pipeline_args.zones = [args.zone] if args.zone else args.zones
                pipeline_args.zone = args.zone
                pipeline_args.max_scenes = 1
                pipeline_args.stage = "full"
                pipeline_args.input_scene_data = None
                pipeline_args.input_analysis_results = None
                pipeline_args.input_scoring_results = None
                
                run_modular_pipeline(provider_name=provider_name, run_id=RUN_ID, args=pipeline_args, total_providers=len(providers_to_run))
        else:
            print("🧠 Running SAAM-Enhanced OpenAI to Z Challenge Checkpoints")
            # Pass relevant args including model configuration - handle both --zone and --zones
            checkpoint_kwargs = {
                'model': args.model,
                'temperature': args.temperature
            }
            if args.zones:
                checkpoint_kwargs['zones'] = args.zones
            elif args.zone:
                checkpoint_kwargs['zone'] = args.zone
            else:
                checkpoint_kwargs['zone'] = None  # No zone specified
            run_enhanced_checkpoints(run_id=RUN_ID, **checkpoint_kwargs)

    elif args.list_zones:
        list_target_zones()

    elif args.competition_ready:
        logger.info("🏆 Preparing for competition submission...")
        if args.no_openai:
            print("🚫 Skipping Competition-Ready Checkpoints (--no-openai flag enabled)")
            print(f"🔧 Running core archaeological pipeline for zone {args.zone or 'default'} instead")
            logger.info("Competition-ready checkpoints skipped due to --no-openai flag - running core pipeline")
            
            # Run the core pipeline instead of competition checkpoints
            providers_to_run = DEFAULT_PROVIDERS or ['gedi', 'sentinel2']
            logger.info(f"🚀 Running core pipeline for provider(s): {providers_to_run}")
            
            for provider_name in providers_to_run:
                # Create a mock args object for pipeline execution
                pipeline_args = argparse.Namespace()
                pipeline_args.zones = [args.zone] if args.zone else args.zones
                pipeline_args.zone = args.zone
                pipeline_args.max_scenes = 1
                pipeline_args.stage = "full"
                pipeline_args.input_scene_data = None
                pipeline_args.input_analysis_results = None
                pipeline_args.input_scoring_results = None
                
                run_modular_pipeline(provider_name=provider_name, run_id=RUN_ID, args=pipeline_args, total_providers=len(providers_to_run))
            logger.info("🏁 Core pipeline execution complete (no OpenAI calls).")
        else:
            print("🧠 Running SAAM-Enhanced Checkpoints for Maximum Competition Impact...")
            checkpoint_kwargs = {
                'model': args.model,
                'temperature': args.temperature
            }
            if args.zones:
                checkpoint_kwargs['zones'] = args.zones
            elif args.zone:
                checkpoint_kwargs['zone'] = args.zone
            else:
                checkpoint_kwargs['zone'] = None  # No zone specified
            run_enhanced_checkpoints(run_id=RUN_ID, **checkpoint_kwargs)
            logger.info("🏁 Competition readiness preparation complete (SAAM-enhanced checkpoints).")
        
    # The following part for --validate-checkpoint-num needs to be integrated.
    # It was previously under a more generic 'checkpoints' mode.
    # Let's assume for now that checkpoint validation is part of the --checkpoint or --all-checkpoints flow,
    # or needs its own explicit flag if it's a distinct top-level operation.
    # The current parser setup makes --checkpoint a number, not a mode to then validate.
    # For simplicity, I'll assume validation is not a primary mode here based on current structure
    # and the mutually exclusive group. If validation is a separate top-level operation,
    # the parser needs adjustment.

    # Based on the original parser, --validate-checkpoint-num and --results-dir were not part of the
    # mutually_exclusive_group. Let's see where these were intended to be used.
    # The original code had:
    # elif args.mode == "checkpoints":
    #     if args.validate:  <-- This 'validate' flag is not in the current args Namespace based on parser.
    #                            'validate_checkpoint_num' and 'results_dir' are.

    # Let's add a distinct check for validation if those args are present,
    # assuming it can be run independently or alongside other checkpoint runs.
    # However, the 'mutually_exclusive_group' implies only one primary action.
    # This suggests that if --validate-checkpoint-num is used, it should be its OWN primary action.
    # The parser needs to be updated to reflect this if that's the intent.
    # For now, if args.validate_checkpoint_num is set, we'll try to validate.
    # This might conflict if another mode like --pipeline is also set (which parser should prevent).

    # Re-evaluating the parser: `validate_checkpoint_num` is NOT in the exclusive group.
    # This means it can be passed along with other flags.
    # The logic should be: if validate_checkpoint_num is present, perform validation.
    # This can happen AFTER another main operation, or as its own operation if no other main op is chosen
    # (though the group is 'required=True').

    # Let's refine the handling for validation. It's not a primary mode from the group.
    # It's an ancillary action or a mis-grouped argument.
    # Given the original code's structure (elif args.mode == "checkpoints": if args.validate:),
    # it implies checkpoint validation was a sub-mode of "checkpoints".
    # The current args.checkpoint is a number, not a mode.

    # Safest immediate fix is to handle the primary modes.
    # If validation is needed, we'll need to clarify its intended invocation.
    # For now, the AttributeError for 'mode' is the primary target.

    else:
        logger.warning(f"No recognized primary operation selected. Please use --help for options.")
        parser.print_help()


def run_modular_pipeline_for_all_providers(run_id: str, args: argparse.Namespace):
    """Helper to run pipeline for all providers specified in DEFAULT_PROVIDERS from config."""
    logger = logging.getLogger(__name__)
    
    if not DEFAULT_PROVIDERS:
        logger.warning("No default providers configured in src.core.config.py (DEFAULT_PROVIDERS list is empty). Skipping.")
        return

    logger.info(f"Running pipeline for all default providers: {DEFAULT_PROVIDERS}")
    any_provider_run = False
    for provider_key in DEFAULT_PROVIDERS:
        logger.info(f"\n===== Processing Provider: {provider_key.upper()} (Run ID: {run_id}) =====")
        
        if provider_key not in SATELLITE_PROVIDERS:
            logger.error(f"Provider key '{provider_key}' from DEFAULT_PROVIDERS not found in SATELLITE_PROVIDERS. Skipping.")
            continue

        try:
            ProviderClass = SATELLITE_PROVIDERS[provider_key]
            provider_instance = ProviderClass()
            logger.info(f"Instantiated provider: {provider_instance.__class__.__name__}")

            pipeline = ModularPipeline(provider_instance=provider_instance, run_id=run_id)
            # Handle both --zone (singular) and --zones (plural) arguments
            zones_to_process = None
            if args.zones:
                zones_to_process = args.zones
            elif args.zone:
                zones_to_process = [args.zone]  # Convert single zone to list

            if args.stage == "acquire_data":
                logger.info(f"▶️ Running stage: acquire_data for {provider_key}")
                pipeline.acquire_data(zones=zones_to_process, max_scenes=args.max_scenes)
            elif args.stage == "analyze_scenes":
                logger.info(f"▶️ Running stage: analyze_scenes for {provider_key}")
                # For "all providers" mode, input paths would need to be dynamic or generic.
                # This assumes that if running a specific stage for "all", the user intends
                # for each provider to look for its own staged files based on the run_id and provider_name.
                # ModularPipeline's _get_staged_data_path will handle this.
                # We construct the expected path to check if it exists.
                # If a generic --input-scene-data was provided, it might not apply to all providers.
                # For simplicity, we'll assume each provider's stage relies on its own prior stage output.

                # Path to the expected input file for this provider and stage
                expected_input_path = pipeline._get_staged_data_path("acquired_scene_data") # ModularPipeline method
                
                if args.input_scene_data: # User provided a specific file
                    logger.warning(f"Using user-specified --input-scene-data {args.input_scene_data} for {provider_key}. This might override provider-specific data if not intended.")
                    input_data_for_stage = args.input_scene_data
                elif expected_input_path.is_file():
                     input_data_for_stage = expected_input_path
                else:
                    logger.error(f"--input-scene-data not specified and default input {expected_input_path} not found for 'analyze_scenes' stage for provider {provider_key}. Skipping.")
                    continue
                
                if not Path(input_data_for_stage).is_file():
                    logger.error(f"Input scene data file not found: {input_data_for_stage} for provider {provider_key}. Skipping.")
                    continue
                pipeline.analyze_scenes(scene_data_input=Path(input_data_for_stage))

            elif args.stage == "score_zones":
                logger.info(f"▶️ Running stage: score_zones for {provider_key}")
                expected_input_path = pipeline._get_staged_data_path("analysis_results")
                if args.input_analysis_results:
                    input_data_for_stage = args.input_analysis_results
                elif expected_input_path.is_file():
                    input_data_for_stage = expected_input_path
                else:
                    logger.error(f"--input-analysis-results not specified and default input {expected_input_path} not found for 'score_zones' stage for provider {provider_key}. Skipping.")
                    continue

                if not Path(input_data_for_stage).is_file():
                    logger.error(f"Input analysis results file not found: {input_data_for_stage} for provider {provider_key}. Skipping.")
                    continue
                pipeline.score_zones(analysis_results_input=Path(input_data_for_stage))

            elif args.stage == "generate_outputs":
                logger.info(f"▶️ Running stage: generate_outputs for {provider_key}")
                expected_analysis_path = pipeline._get_staged_data_path("analysis_results")
                expected_scoring_path = pipeline._get_staged_data_path("scoring_results")

                if args.input_analysis_results: input_analysis = args.input_analysis_results
                elif expected_analysis_path.is_file(): input_analysis = expected_analysis_path
                else:
                    logger.error(f"Analysis results input not found for 'generate_outputs' for {provider_key}. Skipping.")
                    continue
                
                if args.input_scoring_results: input_scoring = args.input_scoring_results
                elif expected_scoring_path.is_file(): input_scoring = expected_scoring_path
                else:
                    logger.error(f"Scoring results input not found for 'generate_outputs' for {provider_key}. Skipping.")
                    continue

                if not Path(input_analysis).is_file():
                    logger.error(f"Input analysis results file not found: {input_analysis} for {provider_key}. Skipping.")
                    continue
                if not Path(input_scoring).is_file():
                    logger.error(f"Input scoring results file not found: {input_scoring} for {provider_key}. Skipping.")
                    continue
                pipeline.generate_outputs(
                    analysis_results_input=Path(input_analysis),
                    scoring_results_input=Path(input_scoring)
                )
            elif args.stage == "full":
                logger.info(f"▶️ Running full pipeline for {provider_key}")
                pipeline.run(zones=zones_to_process, max_scenes=args.max_scenes)
            else:
                logger.error(f"Unknown pipeline stage: {args.stage} for provider {provider_key}. Skipping.")
                continue
            
            logger.info(f"Modular pipeline stage '{args.stage}' for provider {provider_key} completed processing.")
            any_provider_run = True
        except ImportError as e: # Should not happen if SATELLITE_PROVIDERS is correct
            logger.error(f"Could not import or instantiate provider {provider_key}: {e}")
        except Exception as e:
            logger.error(f"Error running pipeline stage '{args.stage}' for provider {provider_key}: {e}", exc_info=True)
            
    if not any_provider_run:
        logger.error("No providers were successfully run. Check configuration and provider availability.")


if __name__ == "__main__":
    main()
