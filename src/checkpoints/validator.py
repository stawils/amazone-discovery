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

from .checkpoint1 import Checkpoint1Familiarize
from .checkpoint2 import Checkpoint2Explorer
from .checkpoint3 import Checkpoint3Discovery
from .checkpoint4 import Checkpoint4Story
from .checkpoint5 import Checkpoint5Final

logger = logging.getLogger(__name__)

class CheckpointValidator:
    """Validates checkpoint results against competition requirements"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.checkpoint_classes = {
            1: Checkpoint1Familiarize,
            2: Checkpoint2Explorer,
            3: Checkpoint3Discovery,
            4: Checkpoint4Story,
            5: Checkpoint5Final
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
        """Generate a comprehensive validation report"""
        
        validation_results = self.validate_all_checkpoints()
        
        # Create readable report
        report_path = self.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# OpenAI to Z Challenge - Checkpoint Validation Report\n\n")
            f.write(f"**Generated:** {validation_results['timestamp']}\n")
            f.write(f"**Results Directory:** {validation_results['results_dir']}\n\n")
            
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
            
            for checkpoint_num in range(1, 6):
                f.write(f"### Checkpoint {checkpoint_num}\n\n")
                
                if checkpoint_num in validation_results['checkpoints']:
                    checkpoint_result = validation_results['checkpoints'][checkpoint_num]
                    
                    if checkpoint_result.get('valid', False):
                        f.write("**Status:** ✅ VALID\n\n")
                        
                        if 'requirements_met' in checkpoint_result:
                            f.write("**Requirements:**\n")
                            for req_name, req_met in checkpoint_result['requirements_met'].items():
                                status = "✅" if req_met else "❌"
                                f.write(f"- {status} {req_name}\n")
                            f.write("\n")
                    else:
                        f.write("**Status:** ❌ INVALID\n\n")
                        
                        if 'errors' in checkpoint_result:
                            f.write("**Errors:**\n")
                            for error in checkpoint_result['errors']:
                                f.write(f"- ❌ {error}\n")
                            f.write("\n")
                        
                        if 'requirements_met' in checkpoint_result:
                            f.write("**Requirements Status:**\n")
                            for req_name, req_met in checkpoint_result['requirements_met'].items():
                                status = "✅" if req_met else "❌"
                                f.write(f"- {status} {req_name}\n")
                            f.write("\n")
                else:
                    f.write("**Status:** ❌ NOT COMPLETED\n\n")
            
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
        
        logger.info(f"Validation report generated: {report_path}")
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
    """Main validation entry point"""
    
    import argparse
    from src.core.config import RESULTS_DIR
    
    parser = argparse.ArgumentParser(description="Validate OpenAI to Z checkpoints")
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory to validate')
    parser.add_argument('--checkpoint', type=int, choices=[1,2,3,4,5],
                       help='Validate specific checkpoint only')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed validation report')
    
    args = parser.parse_args()
    
    # Find most recent results directory if not specified
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Find most recent checkpoints directory
        checkpoint_dirs = list(RESULTS_DIR.glob("checkpoints_*"))
        if not checkpoint_dirs:
            print("❌ No checkpoint results found. Run checkpoints first.")
            return
        
        results_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results: {results_dir}")
    
    validator = CheckpointValidator(results_dir)
    
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
    main()