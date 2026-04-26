"""
Model registry for managing trained models and their metadata.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import torch
from datetime import datetime


class ModelRegistry:
    """Central registry for managing models, checkpoints, and metadata."""
    
    def __init__(self, registry_path: Path = Path("models/registry.json")):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, Any]] = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.models, f, indent=2, default=str)
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        checkpoint_path: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_type: Type (text, image, video, audio)
            checkpoint_path: Path to model checkpoint
            config: Model configuration
            metrics: Evaluation metrics
            metadata: Additional metadata
            
        Returns:
            Model ID (versioned name)
        """
        # Create versioned ID
        version = self._get_next_version(model_name)
        model_id = f"{model_name}_v{version}"
        
        self.models[model_id] = {
            "name": model_name,
            "type": model_type,
            "version": version,
            "checkpoint_path": checkpoint_path,
            "config": config,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
        }
        
        self._save_registry()
        return model_id
    
    def _get_next_version(self, model_name: str) -> int:
        """Get next version number for a model."""
        versions = [
            info["version"]
            for name, info in self.models.items()
            if info["name"] == model_name
        ]
        return max(versions, default=0) + 1
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model."""
        return self.models.get(model_id)
    
    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a model."""
        matching_models = [
            (model_id, info)
            for model_id, info in self.models.items()
            if info["name"] == model_name
        ]
        if not matching_models:
            return None
        
        latest = max(matching_models, key=lambda x: x[1]["version"])
        return {"model_id": latest[0], **latest[1]}
    
    def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """List all registered model IDs, optionally filtered by type."""
        if model_type:
            return [
                model_id
                for model_id, info in self.models.items()
                if info["type"] == model_type
            ]
        return list(self.models.keys())
    
    def load_model(self, model_id: str, device: str = "cpu") -> torch.nn.Module:
        """
        Load a model from the registry.
        
        Args:
            model_id: Model ID to load
            device: Device to load model onto
            
        Returns:
            Loaded model
        """
        info = self.get_model_info(model_id)
        if not info:
            raise ValueError(f"Model {model_id} not found in registry")
        
        checkpoint_path = Path(info["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # TODO: Instantiate model based on type and config
        # This will be implemented when we build specific model classes
        
        return checkpoint
    
    def delete_model(self, model_id: str, delete_checkpoint: bool = False):
        """Delete a model from the registry."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if delete_checkpoint:
            checkpoint_path = Path(self.models[model_id]["checkpoint_path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        
        del self.models[model_id]
        self._save_registry()
