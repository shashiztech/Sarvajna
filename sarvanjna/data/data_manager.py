"""
Data manager for versioning, lineage tracking, and dataset management.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime
import shutil


class DataManager:
    """
    Central data management system for versioning and lineage tracking.
    
    Tracks:
    - Dataset versions
    - Data provenance
    - Preprocessing lineage
    - Quality metrics
    """
    
    def __init__(self, data_root: Path = Path("data")):
        self.data_root = Path(data_root)
        self.metadata_path = self.data_root / "metadata.json"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {
            "datasets": {},
            "versions": {},
            "lineage": {},
        }
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def register_dataset(
        self,
        name: str,
        modality: str,
        source_path: Path,
        description: str = "",
        license: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new dataset.
        
        Args:
            name: Dataset name
            modality: text, image, video, audio
            source_path: Path to raw data
            description: Dataset description
            license: License information
            metadata: Additional metadata
            
        Returns:
            Dataset ID
        """
        # Compute dataset hash for versioning
        dataset_hash = self._compute_dataset_hash(source_path)
        dataset_id = f"{name}_{dataset_hash[:8]}"
        
        # Copy to managed location
        managed_path = self.data_root / "raw" / modality / dataset_id
        managed_path.mkdir(parents=True, exist_ok=True)
        
        if source_path.is_file():
            shutil.copy2(source_path, managed_path / source_path.name)
        else:
            shutil.copytree(source_path, managed_path, dirs_exist_ok=True)
        
        # Register in metadata
        self.metadata["datasets"][dataset_id] = {
            "name": name,
            "modality": modality,
            "path": str(managed_path),
            "description": description,
            "license": license,
            "hash": dataset_hash,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        self._save_metadata()
        return dataset_id
    
    def create_version(
        self,
        dataset_id: str,
        preprocessing_steps: List[str],
        output_path: Path,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a preprocessed version of a dataset.
        
        Args:
            dataset_id: Source dataset ID
            preprocessing_steps: List of preprocessing operations
            output_path: Path to processed data
            stats: Quality statistics
            
        Returns:
            Version ID
        """
        if dataset_id not in self.metadata["datasets"]:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Create version ID
        version_num = len(self.metadata["versions"].get(dataset_id, [])) + 1
        version_id = f"{dataset_id}_v{version_num}"
        
        # Record version
        version_info = {
            "version_id": version_id,
            "dataset_id": dataset_id,
            "preprocessing_steps": preprocessing_steps,
            "output_path": str(output_path),
            "stats": stats or {},
            "created_at": datetime.now().isoformat(),
        }
        
        if dataset_id not in self.metadata["versions"]:
            self.metadata["versions"][dataset_id] = []
        
        self.metadata["versions"][dataset_id].append(version_info)
        
        # Track lineage
        self.metadata["lineage"][version_id] = {
            "source": dataset_id,
            "transforms": preprocessing_steps,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._save_metadata()
        return version_id
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset."""
        return self.metadata["datasets"].get(dataset_id)
    
    def get_latest_version(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a dataset."""
        versions = self.metadata["versions"].get(dataset_id, [])
        return versions[-1] if versions else None
    
    def list_datasets(self, modality: Optional[str] = None) -> List[str]:
        """List all datasets, optionally filtered by modality."""
        datasets = self.metadata["datasets"]
        if modality:
            return [
                ds_id
                for ds_id, info in datasets.items()
                if info["modality"] == modality
            ]
        return list(datasets.keys())
    
    def get_lineage(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get the full lineage of a dataset version."""
        return self.metadata["lineage"].get(version_id)
    
    def _compute_dataset_hash(self, path: Path) -> str:
        """Compute hash of dataset for versioning."""
        hasher = hashlib.sha256()
        
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        else:
            # For directories, hash the directory structure and file names
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    hasher.update(str(file_path.relative_to(path)).encode())
        
        return hasher.hexdigest()
    
    def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Validate dataset integrity and quality.
        
        Returns:
            Dictionary with validation results
        """
        info = self.get_dataset_info(dataset_id)
        if not info:
            return {"valid": False, "error": "Dataset not found"}
        
        dataset_path = Path(info["path"])
        if not dataset_path.exists():
            return {"valid": False, "error": "Dataset path does not exist"}
        
        # Basic validation
        validation = {
            "valid": True,
            "exists": dataset_path.exists(),
            "modality": info["modality"],
            "file_count": len(list(dataset_path.rglob("*"))),
        }
        
        return validation
