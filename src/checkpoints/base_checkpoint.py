# src/checkpoints/base_checkpoint.py
"""
Base checkpoint class for OpenAI to Z Challenge
Provides common functionality for all checkpoints
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class BaseCheckpoint(ABC):
    """Abstract base class for OpenAI challenge checkpoints"""
    
    def __init__(self, checkpoint_num: int, session_id: str, results_dir: Path):
        self.checkpoint_num = checkpoint_num
        self.session_id = session_id
        self.results_dir = results_dir
        self.start_time = None
        self.end_time = None
        
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the checkpoint logic"""
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Return checkpoint requirements and validation criteria"""
        pass
    
    def start_checkpoint(self):
        """Mark checkpoint start time"""
        self.start_time = datetime.now()
        logger.info(f"🎯 Starting Checkpoint {self.checkpoint_num}")
    
    def end_checkpoint(self):
        """Mark checkpoint end time"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"✅ Checkpoint {self.checkpoint_num} completed in {duration:.1f}s")
    
    def save_result(self, result: Dict[str, Any]) -> Path:
        """Save checkpoint result to file"""
        
        # Add timing information
        if self.start_time and self.end_time:
            result['execution_time'] = {
                'start': self.start_time.isoformat(),
                'end': self.end_time.isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds()
            }
        
        # Save to JSON file
        result_file = self.results_dir / f"checkpoint_{self.checkpoint_num}_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"📄 Results saved: {result_file}")
        
        # Save separate validation report if validation exists
        if 'validation' in result:
            validation_file = self.results_dir / f"checkpoint_{self.checkpoint_num}_validation.json"
            validation_report = {
                'checkpoint': self.checkpoint_num,
                'timestamp': datetime.now().isoformat(),
                'validation': result['validation'],
                'requirements_summary': self._create_validation_summary(result['validation'])
            }
            
            with open(validation_file, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            logger.info(f"📋 Validation report saved: {validation_file}")
        
        return result_file
    
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate checkpoint result against requirements"""
        
        requirements = self.get_requirements()
        validation = {
            'checkpoint': self.checkpoint_num,
            'valid': True,
            'errors': [],
            'warnings': [],
            'requirements_met': {}
        }
        
        # Check each requirement
        for req_name, req_spec in requirements.items():
            try:
                if self._check_requirement(result, req_name, req_spec):
                    validation['requirements_met'][req_name] = True
                else:
                    validation['requirements_met'][req_name] = False
                    validation['errors'].append(f"Requirement '{req_name}' not met")
                    validation['valid'] = False
            except Exception as e:
                validation['requirements_met'][req_name] = False
                validation['errors'].append(f"Error checking '{req_name}': {str(e)}")
                validation['valid'] = False
        
        return validation
    
    def _check_requirement(self, result: Dict[str, Any], req_name: str, req_spec: Dict[str, Any]) -> bool:
        """Check if a specific requirement is met"""
        
        req_type = req_spec.get('type', 'exists')
        req_path = req_spec.get('path', req_name)
        
        # Navigate to the required field
        current = result
        for key in req_path.split('.'):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        # Check requirement type
        if req_type == 'exists':
            return current is not None
        elif req_type == 'not_empty':
            return current is not None and len(current) > 0
        elif req_type == 'min_count':
            min_count = req_spec.get('min_count', 1)
            return isinstance(current, (list, dict)) and len(current) >= min_count
        elif req_type == 'min_value':
            min_value = req_spec.get('min_value', 0)
            return isinstance(current, (int, float)) and current >= min_value
        elif req_type == 'contains':
            required_items = req_spec.get('items', [])
            return all(item in current for item in required_items)
        
        return True
    
    def _create_validation_summary(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Create a human-readable validation summary"""
        requirements = self.get_requirements()
        
        summary = {
            'overall_status': 'PASSED' if validation['valid'] else 'FAILED',
            'requirements_total': len(requirements),
            'requirements_passed': sum(1 for passed in validation['requirements_met'].values() if passed),
            'requirements_failed': sum(1 for passed in validation['requirements_met'].values() if not passed),
            'detailed_results': [],
            'next_steps': []
        }
        
        # Create detailed results for each requirement
        for req_name, req_spec in requirements.items():
            req_passed = validation['requirements_met'].get(req_name, False)
            req_detail = {
                'requirement': req_name,
                'description': req_spec.get('description', 'No description'),
                'status': 'PASSED' if req_passed else 'FAILED',
                'type': req_spec.get('type', 'exists'),
                'path': req_spec.get('path', req_name)
            }
            summary['detailed_results'].append(req_detail)
        
        # Add next steps based on failures
        if not validation['valid']:
            for error in validation.get('errors', []):
                if 'five_anomalies' in error:
                    summary['next_steps'].append('Increase anomaly detection sensitivity or lower confidence thresholds')
                elif 'dataset_ids_logged' in error:
                    summary['next_steps'].append('Ensure all scene/granule IDs are properly logged')
                elif 'wkt_footprints' in error:
                    summary['next_steps'].append('Generate WKT polygon footprints for detected anomalies')
                elif 'reproducibility' in error:
                    summary['next_steps'].append('Implement reproducibility verification system')
        else:
            summary['next_steps'].append('All requirements met - checkpoint ready for submission')
            
        return summary
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Main execution method with timing and validation"""
        
        self.start_checkpoint()
        
        try:
            # Execute checkpoint logic
            result = self.execute(**kwargs)
            
            # Add metadata
            result.update({
                'checkpoint': self.checkpoint_num,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            # Validate result
            validation = self.validate_result(result)
            result['validation'] = validation
            
            if not validation['valid']:
                logger.warning(f"Checkpoint {self.checkpoint_num} validation failed")
                for error in validation['errors']:
                    logger.warning(f"  - {error}")
            
            self.end_checkpoint()
            
            # Save result
            self.save_result(result)
            
            return result
            
        except Exception as e:
            self.end_checkpoint()
            
            error_result = {
                'checkpoint': self.checkpoint_num,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            logger.error(f"Checkpoint {self.checkpoint_num} failed: {e}")
            self.save_result(error_result)
            
            return error_result


# src/checkpoints/__init__.py
"""
OpenAI to Z Challenge Checkpoint System
"""

from .base_checkpoint import BaseCheckpoint

__all__ = ['BaseCheckpoint']


# src/checkpoints/checkpoint1.py
"""
Checkpoint 1: Familiarize yourself with the challenge and data
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Checkpoint1(BaseCheckpoint):
    """Checkpoint 1: Familiarize with challenge and data"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 1"""
        return {
            'data_downloaded': {
                'type': 'exists',
                'path': 'data_downloaded'
            },
            'openai_analysis': {
                'type': 'exists', 
                'path': 'openai_analysis'
            },
            'model_version': {
                'type': 'exists',
                'path': 'openai_analysis.model'
            },
            'dataset_id': {
                'type': 'exists',
                'path': 'data_downloaded.scene_id'
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 1 logic"""
        
        # This will be called by the main CheckpointRunner
        # The actual implementation is in openai_checkpoints.py
        return {
            'title': 'Familiarize with Challenge and Data',
            'description': 'Download data and run OpenAI analysis',
            'implemented_in': 'openai_checkpoints.py'
        }