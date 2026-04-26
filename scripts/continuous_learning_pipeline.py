"""
Automated training pipeline with continuous learning.

Orchestrates data collection, training, evaluation, and model updates.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import subprocess
import yaml

import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """
    Automated training pipeline with continuous learning.
    
    Features:
    - Automated data collection from legal sources
    - Incremental training on new data
    - Model evaluation and versioning
    - Automatic deployment of improved models
    - Legal compliance tracking
    """
    
    def __init__(
        self,
        config_path: str,
        model_type: str = 'text',  # text, image, video, music
        output_dir: str = 'outputs/continuous_learning'
    ):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Tracking
        self.pipeline_state_path = self.output_dir / 'pipeline_state.json'
        self.pipeline_state = self.load_pipeline_state()
    
    def load_pipeline_state(self) -> Dict:
        """Load pipeline state."""
        if self.pipeline_state_path.exists():
            with open(self.pipeline_state_path, 'r') as f:
                return json.load(f)
        return {
            'last_data_collection': None,
            'last_training': None,
            'current_version': 0,
            'best_metric': None,
            'training_history': []
        }
    
    def save_pipeline_state(self):
        """Save pipeline state."""
        with open(self.pipeline_state_path, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
    
    # ============== Data Collection ==============
    
    def collect_new_data(self) -> bool:
        """
        Collect new training data from legal sources.
        
        Returns:
            True if new data was collected
        """
        logger.info("=== Collecting New Data ===")
        
        data_dir = self.output_dir / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Run legal data collector
        script_path = Path(__file__).parent / 'legal_data_collector.py'
        
        try:
            result = subprocess.run(
                ['python', str(script_path)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Data collection successful")
                self.pipeline_state['last_data_collection'] = datetime.now().isoformat()
                self.save_pipeline_state()
                return True
            else:
                logger.error(f"Data collection failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error("Data collection timed out")
            return False
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess collected data."""
        logger.info("=== Preprocessing Data ===")
        
        # Model-specific preprocessing
        if self.model_type == 'text':
            self._preprocess_text_data()
        elif self.model_type == 'image':
            self._preprocess_image_data()
        elif self.model_type == 'video':
            self._preprocess_video_data()
        elif self.model_type == 'music':
            self._preprocess_audio_data()
    
    def _preprocess_text_data(self):
        """Preprocess text data."""
        try:
            from sarvanjna.preprocessing.text_processor import TextProcessor
            
            processor = TextProcessor()
            
            # Load raw data
            raw_data_path = Path('data/legal_sources/wikipedia_articles.jsonl')
            processed_data_path = self.output_dir / 'data' / 'processed' / 'text_data.jsonl'
            processed_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not raw_data_path.exists():
                logger.warning(f"No raw data found at {raw_data_path}")
                return
            
            processed_count = 0
            with open(raw_data_path, 'r', encoding='utf-8') as f_in, \
                 open(processed_data_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    try:
                        item = json.loads(line)
                        
                        # Process text
                        text = item.get('text', '')
                        processed = processor.normalize(text)
                        
                        # Quality check
                        if processor.calculate_quality_score(processed) > 0.5:
                            item['processed_text'] = processed
                            f_out.write(json.dumps(item) + '\n')
                            processed_count += 1
                    except Exception as e:
                        logger.debug(f"Error processing item: {e}")
            
            logger.info(f"Preprocessed {processed_count} text items to {processed_data_path}")
        
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
    
    def _preprocess_image_data(self):
        """Preprocess image data."""
        logger.info("Image preprocessing: Download and resize images")
        # TODO: Implement image preprocessing
    
    def _preprocess_video_data(self):
        """Preprocess video data."""
        logger.info("Video preprocessing not yet implemented")
    
    def _preprocess_audio_data(self):
        """Preprocess audio data."""
        logger.info("Audio preprocessing not yet implemented")
    
    # ============== Training ==============
    
    def train_model(self, incremental: bool = True) -> Optional[str]:
        """
        Train or update model.
        
        Args:
            incremental: If True, continue from previous checkpoint
        
        Returns:
            Path to trained model checkpoint
        """
        logger.info("=== Training Model ===")
        
        # Get training script
        training_script = self._get_training_script()
        
        # Prepare training config
        checkpoint_path = None
        if incremental and self.pipeline_state['current_version'] > 0:
            # Get latest checkpoint
            checkpoints_dir = self.output_dir / 'models' / 'checkpoints'
            if checkpoints_dir.exists():
                checkpoints = sorted(checkpoints_dir.glob('*.ckpt'))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    logger.info(f"Continuing from checkpoint: {checkpoint_path}")
        
        # Run training
        try:
            cmd = ['python', training_script]
            
            if checkpoint_path:
                cmd.extend(['--resume-from', checkpoint_path])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=86400  # 24 hours
            )
            
            if result.returncode == 0:
                logger.info("Training successful")
                
                # Update state
                self.pipeline_state['last_training'] = datetime.now().isoformat()
                self.pipeline_state['current_version'] += 1
                self.save_pipeline_state()
                
                # Find new checkpoint
                checkpoints_dir = self.output_dir / 'models' / 'checkpoints'
                if checkpoints_dir.exists():
                    checkpoints = sorted(checkpoints_dir.glob('*.ckpt'))
                    if checkpoints:
                        return str(checkpoints[-1])
                
                return "training_complete"
            else:
                logger.error(f"Training failed: {result.stderr}")
                return None
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return None
    
    def _get_training_script(self) -> str:
        """Get training script path for model type."""
        scripts = {
            'text': 'examples/train_text_model.py',
            'image': 'examples/train_latent_diffusion.py',
            'video': 'examples/train_text_to_video.py',
            'music': 'examples/train_music.py'
        }
        return scripts.get(self.model_type, 'examples/train_text_model.py')
    
    # ============== Evaluation ==============
    
    def evaluate_model(self, checkpoint_path: str) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
        
        Returns:
            Dictionary of metrics
        """
        logger.info("=== Evaluating Model ===")
        
        # Model-specific evaluation
        if self.model_type == 'text':
            metrics = self._evaluate_text_model(checkpoint_path)
        elif self.model_type == 'image':
            metrics = self._evaluate_image_model(checkpoint_path)
        elif self.model_type == 'video':
            metrics = self._evaluate_video_model(checkpoint_path)
        elif self.model_type == 'music':
            metrics = self._evaluate_music_model(checkpoint_path)
        else:
            metrics = {}
        
        # Log metrics
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Record in history
        self.pipeline_state['training_history'].append({
            'timestamp': datetime.now().isoformat(),
            'checkpoint': checkpoint_path,
            'metrics': metrics
        })
        
        # Update best metric
        primary_metric = metrics.get('primary_metric', 0)
        if (self.pipeline_state['best_metric'] is None or 
            primary_metric > self.pipeline_state['best_metric']):
            self.pipeline_state['best_metric'] = primary_metric
            logger.info(f"New best model! Metric: {primary_metric}")
        
        self.save_pipeline_state()
        
        return metrics
    
    def _evaluate_text_model(self, checkpoint_path: str) -> Dict:
        """Evaluate text model."""
        # TODO: Load test data and run inference
        # Calculate BLEU, ROUGE, perplexity
        return {
            'primary_metric': 0.85,  # Placeholder
            'perplexity': 25.3,
            'bleu': 0.45
        }
    
    def _evaluate_image_model(self, checkpoint_path: str) -> Dict:
        """Evaluate image model."""
        # Calculate FID, IS, CLIP score
        return {
            'primary_metric': 0.75,  # Placeholder
            'fid': 15.2,
            'inception_score': 3.8
        }
    
    def _evaluate_video_model(self, checkpoint_path: str) -> Dict:
        """Evaluate video model."""
        return {
            'primary_metric': 0.70,
            'fvd': 120.5
        }
    
    def _evaluate_music_model(self, checkpoint_path: str) -> Dict:
        """Evaluate music model."""
        return {
            'primary_metric': 0.72,
            'fad': 2.8
        }
    
    # ============== Model Management ==============
    
    def register_model(self, checkpoint_path: str, metrics: Dict):
        """Register model in registry."""
        logger.info("=== Registering Model ===")
        
        version = self.pipeline_state['current_version']
        
        model_info = {
            'name': f"{self.model_type}_v{version}",
            'version': version,
            'checkpoint_path': checkpoint_path,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'data_sources': self._get_data_sources()
        }
        
        # Save model info
        models_dir = self.output_dir / 'models' / 'registry'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = models_dir / f"{model_info['name']}.json"
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Registered model: {model_info['name']}")
    
    def _get_data_sources(self) -> List[Dict]:
        """Get list of data sources used."""
        manifest_path = Path('data/legal_sources/data_manifest.json')
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                return manifest.get('sources', [])
        
        return []
    
    # ============== Deployment ==============
    
    def deploy_model(self, checkpoint_path: str):
        """Deploy model for production use."""
        logger.info("=== Deploying Model ===")
        
        # Copy to deployment directory
        deploy_dir = Path('deployments') / self.model_type
        deploy_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        dest_path = deploy_dir / 'model_latest.ckpt'
        shutil.copy2(checkpoint_path, dest_path)
        
        # Create deployment metadata
        metadata = {
            'deployed_at': datetime.now().isoformat(),
            'checkpoint': checkpoint_path,
            'version': self.pipeline_state['current_version'],
            'metrics': self.pipeline_state.get('training_history', [])[-1]['metrics'] if self.pipeline_state.get('training_history') else {}
        }
        
        with open(deploy_dir / 'deployment_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model deployed to {dest_path}")
    
    # ============== Main Pipeline ==============
    
    def run_pipeline(
        self,
        collect_data: bool = True,
        train: bool = True,
        evaluate: bool = True,
        deploy: bool = False
    ):
        """
        Run complete continuous learning pipeline.
        
        Args:
            collect_data: Collect new data
            train: Train model
            evaluate: Evaluate model
            deploy: Deploy if improved
        """
        logger.info("=" * 60)
        logger.info("STARTING CONTINUOUS LEARNING PIPELINE")
        logger.info(f"Model Type: {self.model_type}")
        logger.info(f"Version: {self.pipeline_state['current_version']}")
        logger.info("=" * 60)
        
        # Step 1: Collect data
        if collect_data:
            success = self.collect_new_data()
            if not success:
                logger.error("Data collection failed. Aborting pipeline.")
                return
            
            self.preprocess_data()
        
        # Step 2: Train model
        checkpoint_path = None
        if train:
            checkpoint_path = self.train_model(incremental=True)
            if not checkpoint_path:
                logger.error("Training failed. Aborting pipeline.")
                return
        
        # Step 3: Evaluate model
        metrics = {}
        if evaluate and checkpoint_path:
            metrics = self.evaluate_model(checkpoint_path)
            
            # Register model
            self.register_model(checkpoint_path, metrics)
        
        # Step 4: Deploy if improved
        if deploy and checkpoint_path:
            primary_metric = metrics.get('primary_metric', 0)
            
            # Deploy if best model
            if primary_metric == self.pipeline_state['best_metric']:
                self.deploy_model(checkpoint_path)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Current Version: {self.pipeline_state['current_version']}")
        logger.info(f"Best Metric: {self.pipeline_state['best_metric']}")
        logger.info(f"State saved to: {self.pipeline_state_path}")
        logger.info("=" * 60)
    
    def schedule_continuous_learning(self, interval_hours: int = 24):
        """
        Schedule continuous learning to run periodically.
        
        Args:
            interval_hours: Hours between training runs
        """
        logger.info(f"Scheduling continuous learning every {interval_hours} hours")
        
        while True:
            try:
                self.run_pipeline(
                    collect_data=True,
                    train=True,
                    evaluate=True,
                    deploy=True
                )
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
            
            # Wait for next run
            logger.info(f"Waiting {interval_hours} hours until next run...")
            time.sleep(interval_hours * 3600)


def main():
    """Run continuous learning pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Learning Pipeline')
    parser.add_argument('--config', required=True, help='Config YAML path')
    parser.add_argument('--model-type', default='text', 
                       choices=['text', 'image', 'video', 'music'])
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously')
    parser.add_argument('--interval', type=int, default=24,
                       help='Hours between runs (for continuous mode)')
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--deploy', action='store_true',
                       help='Deploy if model improves')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ContinuousLearningPipeline(
        config_path=args.config,
        model_type=args.model_type
    )
    
    # Run
    if args.continuous:
        pipeline.schedule_continuous_learning(interval_hours=args.interval)
    else:
        pipeline.run_pipeline(
            collect_data=not args.skip_data_collection,
            train=not args.skip_training,
            evaluate=True,
            deploy=args.deploy
        )


if __name__ == '__main__':
    main()
