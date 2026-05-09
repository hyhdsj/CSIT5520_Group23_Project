"""
Data downloader module - Download datasets from Hugging Face and save locally
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset


class DataDownloader:
    """
    Download and save datasets locally for future use
    
    This class handles:
    1. Downloading datasets from Hugging Face
    2. Saving them to local JSON files
    3. Tracking download metadata (version, date, size)
    4. Verifying download integrity
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data downloader
        
        Args:
            data_dir: Root directory for storing data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load download metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "datasets": {},
            "last_updated": None,
            "total_size_bytes": 0
        }
    
    def _save_metadata(self):
        """Save download metadata to file"""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _calculate_size(self, data: List[Dict]) -> int:
        """Calculate size of data in bytes"""
        return len(json.dumps(data).encode('utf-8'))
    
    def _generate_hash(self, data: List[Dict]) -> str:
        """Generate SHA256 hash for data verification"""
        content = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def download_gsm8k(self, config: str = "main", force_download: bool = False) -> Dict[str, Any]:
        """
        Download GSM8K dataset and save locally
        
        GSM8K: Grade School Math 8K - Mathematical reasoning problems
        
        Args:
            config: Dataset configuration ("main" or "socratic")
            force_download: Force re-download even if exists
            
        Returns:
            Dictionary with download information
        """
        print(f"\n{'='*60}")
        print(f"Downloading GSM8K dataset (config: {config})")
        print(f"{'='*60}")
        
        # Check if already downloaded
        output_file = self.processed_dir / f"gsm8k_{config}.json"
        if output_file.exists() and not force_download:
            print(f"✓ GSM8K already exists at {output_file}")
            with open(output_file, 'r') as f:
                data = json.load(f)
            return {
                "success": True,
                "file": str(output_file),
                "num_samples": len(data),
                "cached": True
            }
        
        try:
            # Download from Hugging Face
            print("Downloading from Hugging Face...")
            dataset = load_dataset("openai/gsm8k", config, split="train")
            
            # Convert to list of dictionaries
            data = []
            for item in dataset:
                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": "math",
                    "dataset": "gsm8k",
                    "config": config
                })
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            size_bytes = self._calculate_size(data)
            file_hash = self._generate_hash(data)
            
            self.metadata["datasets"]["gsm8k"] = {
                "config": config,
                "file": str(output_file),
                "num_samples": len(data),
                "size_bytes": size_bytes,
                "hash": file_hash,
                "downloaded_at": datetime.now().isoformat(),
                "source": "openai/gsm8k"
            }
            self._save_metadata()
            
            print(f"✓ Successfully downloaded {len(data)} samples")
            print(f"✓ Saved to: {output_file}")
            print(f"✓ File size: {size_bytes / 1024:.2f} KB")
            
            return {
                "success": True,
                "file": str(output_file),
                "num_samples": len(data),
                "size_bytes": size_bytes,
                "cached": False
            }
            
        except Exception as e:
            print(f"✗ Failed to download GSM8K: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_commonsenseqa(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Download CommonsenseQA dataset and save locally
        
        CommonsenseQA: Commonsense reasoning questions
        
        Args:
            force_download: Force re-download even if exists
            
        Returns:
            Dictionary with download information
        """
        print(f"\n{'='*60}")
        print(f"Downloading CommonsenseQA dataset")
        print(f"{'='*60}")
        
        output_file = self.processed_dir / "commonsense_qa.json"
        
        if output_file.exists() and not force_download:
            print(f"✓ CommonsenseQA already exists at {output_file}")
            with open(output_file, 'r') as f:
                data = json.load(f)
            return {
                "success": True,
                "file": str(output_file),
                "num_samples": len(data),
                "cached": True
            }
        
        try:
            print("Downloading from Hugging Face...")
            dataset = load_dataset("tau/commonsense_qa", split="validation")
            
            data = []
            for item in dataset:
                data.append({
                    "question": item["question"],
                    "choices": item["choices"]["text"],
                    "choices_labels": item["choices"]["label"],
                    "answer_key": item["answerKey"],
                    "type": "commonsense",
                    "dataset": "commonsense_qa"
                })
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            size_bytes = self._calculate_size(data)
            file_hash = self._generate_hash(data)
            
            self.metadata["datasets"]["commonsense_qa"] = {
                "file": str(output_file),
                "num_samples": len(data),
                "size_bytes": size_bytes,
                "hash": file_hash,
                "downloaded_at": datetime.now().isoformat(),
                "source": "tau/commonsense_qa",
                "split": "validation"
            }
            self._save_metadata()
            
            print(f"✓ Successfully downloaded {len(data)} samples")
            print(f"✓ Saved to: {output_file}")
            print(f"✓ File size: {size_bytes / 1024:.2f} KB")
            
            return {
                "success": True,
                "file": str(output_file),
                "num_samples": len(data),
                "size_bytes": size_bytes,
                "cached": False
            }
            
        except Exception as e:
            print(f"✗ Failed to download CommonsenseQA: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_hotpotqa(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Download HotpotQA dataset and save locally
        
        HotpotQA: Multi-hop question answering
        
        Args:
            force_download: Force re-download even if exists
            
        Returns:
            Dictionary with download information
        """
        print(f"\n{'='*60}")
        print(f"Downloading HotpotQA dataset")
        print(f"{'='*60}")
        
        output_file = self.processed_dir / "hotpotqa.json"
        
        if output_file.exists() and not force_download:
            print(f"✓ HotpotQA already exists at {output_file}")
            with open(output_file, 'r') as f:
                data = json.load(f)
            return {
                "success": True,
                "file": str(output_file),
                "num_samples": len(data),
                "cached": True
            }
        
        try:
            print("Downloading from Hugging Face...")
            dataset = load_dataset("hotpotqa/hotpotqa", split="train")
            
            # Sample first 5000 to keep file size manageable
            # HotpotQA has ~90k samples, which is very large
            sample_size = min(5000, len(dataset))
            print(f"Sampling {sample_size} out of {len(dataset)} samples...")
            
            data = []
            for i in range(sample_size):
                item = dataset[i]
                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": "multihop",
                    "dataset": "hotpotqa"
                })
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            size_bytes = self._calculate_size(data)
            file_hash = self._generate_hash(data)
            
            self.metadata["datasets"]["hotpotqa"] = {
                "file": str(output_file),
                "num_samples": len(data),
                "size_bytes": size_bytes,
                "hash": file_hash,
                "downloaded_at": datetime.now().isoformat(),
                "source": "hotpotqa/hotpotqa",
                "split": "train",
                "original_size": len(dataset),
                "sampled": sample_size
            }
            self._save_metadata()
            
            print(f"✓ Successfully downloaded {len(data)} samples")
            print(f"✓ Saved to: {output_file}")
            print(f"✓ File size: {size_bytes / 1024:.2f} KB")
            
            return {
                "success": True,
                "file": str(output_file),
                "num_samples": len(data),
                "size_bytes": size_bytes,
                "cached": False
            }
            
        except Exception as e:
            print(f"✗ Failed to download HotpotQA: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_all_datasets(self, force_download: bool = False) -> Dict[str, Any]:
        """
        Download all three datasets
        
        Args:
            force_download: Force re-download all datasets
            
        Returns:
            Dictionary with download results for each dataset
        """
        print("\n" + "="*60)
        print("DOWNLOADING ALL DATASETS")
        print("="*60)
        
        results = {
            "gsm8k_main": self.download_gsm8k(config="main", force_download=force_download),
            "gsm8k_socratic": self.download_gsm8k(config="socratic", force_download=force_download),
            "commonsense_qa": self.download_commonsenseqa(force_download=force_download),
            "hotpotqa": self.download_hotpotqa(force_download=force_download)
        }
        
        # Print summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        success_count = sum(1 for r in results.values() if r.get("success", False))
        total_count = len(results)
        
        for name, result in results.items():
            status = "✓" if result.get("success") else "✗"
            if result.get("success"):
                cached = " (cached)" if result.get("cached") else ""
                print(f"{status} {name}: {result.get('num_samples', 0)} samples{cached}")
            else:
                print(f"{status} {name}: {result.get('error', 'Unknown error')}")
        
        print(f"\nTotal: {success_count}/{total_count} datasets downloaded successfully")
        
        # Save summary
        summary_file = self.data_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Download summary saved to: {summary_file}")
        
        return results
    
    def verify_downloads(self) -> Dict[str, bool]:
        """
        Verify integrity of downloaded datasets
        
        Returns:
            Dictionary with verification results
        """
        print("\n" + "="*60)
        print("VERIFYING DATASET INTEGRITY")
        print("="*60)
        
        verification = {}
        
        for dataset_name, info in self.metadata["datasets"].items():
            file_path = Path(info["file"])
            
            if not file_path.exists():
                verification[dataset_name] = False
                print(f"✗ {dataset_name}: File not found")
                continue
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Verify hash
                current_hash = self._generate_hash(data)
                expected_hash = info["hash"]
                
                if current_hash == expected_hash:
                    verification[dataset_name] = True
                    print(f"✓ {dataset_name}: Verified ({len(data)} samples)")
                else:
                    verification[dataset_name] = False
                    print(f"✗ {dataset_name}: Hash mismatch")
                    
            except Exception as e:
                verification[dataset_name] = False
                print(f"✗ {dataset_name}: Verification failed - {e}")
        
        return verification


def main():
    """Main function to run data download"""
    print("CSIT5520 - Data Download Utility")
    print("This script will download all required datasets locally")
    print("-"*60)
    
    downloader = DataDownloader()
    
    # Check if user wants to force re-download
    force = input("\nForce re-download existing datasets? (y/n, default n): ").lower().strip() == 'y'
    
    # Download all datasets
    results = downloader.download_all_datasets(force_download=force)
    
    # Verify downloads
    print("\n")
    verification = downloader.verify_downloads()
    
    # Final status
    all_verified = all(verification.values())
    if all_verified:
        print("\n✅ All datasets downloaded and verified successfully!")
        print("You can now run experiments using local data files.")
    else:
        print("\n⚠️ Some datasets failed verification. Please check the errors above.")
        print("You can still use local sample data for testing.")


if __name__ == "__main__":
    main()