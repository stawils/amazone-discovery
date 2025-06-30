# src/checkpoints/validator.py
"""
Checkpoint Validation Utility
Tests and validates checkpoint requirements
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .checkpoint1 import Checkpoint1
from .checkpoint2 import Checkpoint2Explorer
from .checkpoint3 import Checkpoint3SiteDiscovery
from .checkpoint4 import Checkpoint4StoryImpact
from .checkpoint5 import Checkpoint5FinalSubmission
from src.core.config import RESULTS_DIR

logger = logging.getLogger(__name__)

class CheckpointValidator:
    """Validates checkpoint results against competition requirements"""
    
    # Requirements dictionary can be extensive and is kept for brevity here
    CHECKPOINT_REQUIREMENTS = {
        1: {
            "description": "Familiarize with challenge and data",
            "files": ["checkpoint_1_result.json"],
            "result_keys": ["data_downloaded", "openai_analysis_summary"],
            "custom_checks": ["check_data_downloaded_structure", "check_openai_summary_content"]
        },
        # ... other checkpoints ...
    }

    def __init__(self, results_dir_to_validate: Path, run_id: str):
        """Initialize the validator with the path to the checkpoint results directory and the run_id."""
        self.results_dir = results_dir_to_validate
        self.run_id = run_id
        logger.info(f"CheckpointValidator initialized for results directory: {self.results_dir}, Run ID: {self.run_id}")
        self.checkpoint_classes = {
            1: Checkpoint1,
            2: Checkpoint2Explorer,
            3: Checkpoint3SiteDiscovery,
            4: Checkpoint4StoryImpact,
            5: Checkpoint5FinalSubmission
        }
    
    def validate_checkpoint(self, checkpoint_num: int, result_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a specific checkpoint"""
        
        if checkpoint_num not in self.checkpoint_classes:
            return {
                'valid': False,
                'error': f"Invalid checkpoint number: {checkpoint_num}"
            }
        
        # Load result data if not provided
        if result_data is None:
            result_file = self.results_dir / f"checkpoint_{checkpoint_num}_result.json"
            if not result_file.exists():
                return {
                    'valid': False,
                    'error': f"Result file not found: {result_file}"
                }
            
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
            except Exception as e:
                return {
                    'valid': False,
                    'error': f"Error loading result file: {e}"
                }
        
        # Create checkpoint instance and validate
        checkpoint_class = self.checkpoint_classes[checkpoint_num]
        checkpoint = checkpoint_class(checkpoint_num, "validator", self.results_dir)
        
        validation = checkpoint.validate_result(result_data)
        
        return validation
    
    def validate_all_checkpoints(self) -> Dict[str, Any]:
        """Validate all available checkpoints"""
        
        overall_validation = {
            'timestamp': datetime.now().isoformat(),
            'results_dir': str(self.results_dir),
            'overall_valid': True,
            'checkpoints': {},
            'summary': {
                'total_checkpoints': 5,
                'completed_checkpoints': 0,
                'valid_checkpoints': 0,
                'invalid_checkpoints': 0,
                'missing_checkpoints': 0
            }
        }
        
        for checkpoint_num in range(1, 6):
            result_file = self.results_dir / f"checkpoint_{checkpoint_num}_result.json"
            
            if result_file.exists():
                validation = self.validate_checkpoint(checkpoint_num)
                overall_validation['checkpoints'][checkpoint_num] = validation
                
                overall_validation['summary']['completed_checkpoints'] += 1
                
                if validation.get('valid', False):
                    overall_validation['summary']['valid_checkpoints'] += 1
                else:
                    overall_validation['summary']['invalid_checkpoints'] += 1
                    overall_validation['overall_valid'] = False
            else:
                overall_validation['checkpoints'][checkpoint_num] = {
                    'valid': False,
                    'error': 'Checkpoint not completed',
                    'status': 'missing'
                }
                overall_validation['summary']['missing_checkpoints'] += 1
                overall_validation['overall_valid'] = False
        
        return overall_validation
    
    def generate_validation_report(self) -> Path:
        """Generate a comprehensive validation report for the current run_id."""
        
        validation_results = self.validate_all_checkpoints()
        
        # Define the output directory for this run's validation reports
        validation_reports_output_dir = RESULTS_DIR / f"run_{self.run_id}" / "validation_reports"
        validation_reports_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report filename is now consistent per run_id
        report_path = validation_reports_output_dir / f"validation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# OpenAI to Z Challenge - Checkpoint Validation Report\n\n")
            f.write(f"**Run ID:** {self.run_id}\n")
            f.write(f"**Validation Target Directory:** {self.results_dir}\n")
            f.write(f"**Report Generated Timestamp:** {datetime.now().isoformat()}\n\n")
            
            # Summary
            summary = validation_results['summary']
            f.write("## Summary\n\n")
            f.write(f"- **Overall Status:** {'✅ VALID' if validation_results['overall_valid'] else '❌ INVALID'}\n")
            f.write(f"- **Completed Checkpoints:** {summary['completed_checkpoints']}/5\n")
            f.write(f"- **Valid Checkpoints:** {summary['valid_checkpoints']}/5\n")
            f.write(f"- **Invalid Checkpoints:** {summary['invalid_checkpoints']}\n")
            f.write(f"- **Missing Checkpoints:** {summary['missing_checkpoints']}\n\n")
            
            # Individual checkpoint details
            f.write("## Checkpoint Details\n\n")
            for checkpoint_num, validation in validation_results.get('checkpoints', {}).items():
                f.write(f"### Checkpoint {checkpoint_num}\n")
                f.write(f"- **Status:** {'✅ VALID' if validation.get('valid') else ('❌ INVALID' if validation.get('status') != 'missing' else '⚠️ MISSING')}\n")
                if not validation.get('valid'):
                    f.write(f"- **Reason:** {validation.get('error', 'N/A')}\n")
                    if validation.get('errors'):
                        for err_detail in validation.get('errors'):
                            f.write(f"  - {err_detail}\n")
                f.write("\n")
            
            # Competition readiness assessment
            f.write("## Competition Readiness\n\n")
            
            if validation_results['overall_valid']:
                f.write("🏆 **COMPETITION READY!**\n\n")
                f.write("All checkpoints are completed and valid. Ready for submission!\n\n")
            else:
                f.write("⚠️ **NOT READY FOR COMPETITION**\n\n")
                f.write("Issues need to be resolved before submission:\n\n")
                
                if summary['missing_checkpoints'] > 0:
                    f.write(f"- Complete {summary['missing_checkpoints']} missing checkpoints\n")
                
                if summary['invalid_checkpoints'] > 0:
                    f.write(f"- Fix validation errors in {summary['invalid_checkpoints']} checkpoints\n")
                
                f.write("\n")
            
            # Next steps
            f.write("## Next Steps\n\n")
            
            if validation_results['overall_valid']:
                f.write("1. Review final submission package\n")
                f.write("2. Prepare livestream presentation\n")
                f.write("3. Submit to OpenAI to Z Challenge\n")
            else:
                f.write("1. Run missing checkpoints:\n")
                for checkpoint_num in range(1, 6):
                    if checkpoint_num not in validation_results['checkpoints'] or not validation_results['checkpoints'][checkpoint_num].get('valid'):
                        f.write(f"   - `python main.py --checkpoint {checkpoint_num}`\n")
                f.write("2. Re-run validation: `python -m src.checkpoints.validator`\n")
                f.write("3. Review and fix any remaining errors\n")
        
        logger.info(f"Validation report for Run ID {self.run_id} generated: {report_path}")
        return report_path
    
    def print_summary(self):
        """Print validation summary to console"""
        
        validation_results = self.validate_all_checkpoints()
        summary = validation_results['summary']
        
        print("\n🔍 CHECKPOINT VALIDATION SUMMARY")
        print("=" * 50)
        
        if validation_results['overall_valid']:
            print("🏆 COMPETITION READY!")
        else:
            print("⚠️  NOT READY FOR COMPETITION")
        
        print(f"\nCheckpoints Status:")
        print(f"  ✅ Valid: {summary['valid_checkpoints']}/5")
        print(f"  ❌ Invalid: {summary['invalid_checkpoints']}/5") 
        print(f"  ⏳ Missing: {summary['missing_checkpoints']}/5")
        
        print(f"\nIndividual Checkpoint Status:")
        for checkpoint_num in range(1, 6):
            if checkpoint_num in validation_results['checkpoints']:
                result = validation_results['checkpoints'][checkpoint_num]
                if result.get('valid', False):
                    print(f"  Checkpoint {checkpoint_num}: ✅ Valid")
                else:
                    print(f"  Checkpoint {checkpoint_num}: ❌ Invalid ({result.get('error', 'Unknown error')})")
            else:
                print(f"  Checkpoint {checkpoint_num}: ⏳ Not completed")
        
        if not validation_results['overall_valid']:
            print(f"\n📋 To fix:")
            print(f"  Run: python main.py --all-checkpoints")
            print(f"  Then: python -m src.checkpoints.validator")


def main():
    """Main validation entry point - for standalone execution or direct call if args are managed."""
    
    import argparse
    # RESULTS_DIR should be available from core.config
    
    parser = argparse.ArgumentParser(description="Validate OpenAI to Z checkpoints")
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory to validate (e.g., /path/to/project/results/run_YYYYMMDD_HHMMSS/checkpoints)')
    parser.add_argument('--run-id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                       help='Run ID to associate with this validation report. Defaults to current timestamp.')
    parser.add_argument('--checkpoint', type=int, choices=[1,2,3,4,5],
                       help='Validate specific checkpoint only')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed validation report')
    
    args = parser.parse_args()
    
    results_to_validate_path = Path(args.results_dir) if args.results_dir else None
    
    if not results_to_validate_path:
        # Attempt to find the latest run_.../checkpoints directory if no specific dir is given
        run_dirs = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if run_dirs:
            latest_run_dir = run_dirs[0]
            potential_checkpoints_dir = latest_run_dir / "checkpoints"
            if potential_checkpoints_dir.exists():
                results_to_validate_path = potential_checkpoints_dir
                logger.info(f"No --results-dir specified, using latest found: {results_to_validate_path}")
            else:
                logger.error("No --results-dir specified and could not find a 'checkpoints' subdirectory in the latest run directory.")
                return
        else:
            logger.error("No --results-dir specified and no 'run_...' directories found in results.")
            return

    if not results_to_validate_path.exists():
        logger.error(f"Specified results directory to validate does not exist: {results_to_validate_path}")
        return

    validator = CheckpointValidator(results_dir_to_validate=results_to_validate_path, run_id=args.run_id)
    
    if args.checkpoint:
        # Validate specific checkpoint
        validation = validator.validate_checkpoint(args.checkpoint)
        print(f"\nCheckpoint {args.checkpoint} Validation:")
        print(f"Valid: {validation.get('valid', False)}")
        
        if not validation.get('valid', False):
            print("Errors:")
            for error in validation.get('errors', []):
                print(f"  - {error}")
    
    elif args.report:
        # Generate detailed report
        report_path = validator.generate_validation_report()
        print(f"✅ Validation report generated: {report_path}")
    
    else:
        # Print summary
        validator.print_summary()


if __name__ == "__main__":
    # This allows running the validator standalone. 
    # It will try to find the latest run's checkpoints or use --results-dir.
    # The run_id for the report will be a new timestamp unless --run-id is specified.
    setup_logging_for_validator() # A simple logger for standalone use
    main()

def setup_logging_for_validator(level=logging.INFO):
    """Basic logging setup for standalone validator script execution."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )
    logger.info("Logging setup for CheckpointValidator standalone execution.")