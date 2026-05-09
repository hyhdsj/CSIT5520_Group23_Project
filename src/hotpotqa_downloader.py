"""
HotpotQA Data Downloader and Processor
Downloads HotpotQA dataset from official source and converts to JSON format
Official dataset: https://hotpotqa.github.io/
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from tqdm import tqdm
import hashlib
import random


class HotpotQADownloader:
    """
    Download and process HotpotQA dataset from official source
    
    Official files:
    - Training set: hotpot_train_v1.1.json (90k samples)
    - Dev distractor: hotpot_dev_distractor_v1.json (7k samples)
    - Dev fullwiki: hotpot_dev_fullwiki_v1.json (7k samples)
    - Test fullwiki: hotpot_test_fullwiki_v1.json (7k samples, no answers)
    """
    
    # Official download URLs
    URLS = {
        "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "dev_distractor": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "dev_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
        "test_fullwiki": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize HotpotQA downloader
        
        Args:
            data_dir: Root directory for storing data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download status
        self.download_status = {}
    
    def download_file(self, file_type: str, force_download: bool = False) -> Optional[str]:
        """
        Download a specific HotpotQA file
        
        Args:
            file_type: One of 'train', 'dev_distractor', 'dev_fullwiki', 'test_fullwiki'
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded file or None if failed
        """
        if file_type not in self.URLS:
            print(f"Unknown file type: {file_type}")
            return None
        
        url = self.URLS[file_type]
        filename = url.split('/')[-1]
        filepath = self.raw_dir / filename
        
        # Check if file already exists
        if filepath.exists() and not force_download:
            print(f"✓ File already exists: {filename}")
            self.download_status[file_type] = {"status": "cached", "path": str(filepath)}
            return str(filepath)
        
        print(f"Downloading {filename}...")
        print(f"From: {url}")
        print(f"To: {filepath}")
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify file size
            file_size = filepath.stat().st_size
            print(f"✓ Download complete: {file_size / 1024 / 1024:.2f} MB")
            
            self.download_status[file_type] = {"status": "downloaded", "path": str(filepath)}
            return str(filepath)
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            self.download_status[file_type] = {"status": "failed", "error": str(e)}
            return None
    
    def download_all(self, force_download: bool = False) -> Dict:
        """
        Download all HotpotQA datasets
        
        Args:
            force_download: Force re-download all files
            
        Returns:
            Dictionary with download status
        """
        print("\n" + "="*60)
        print("Downloading HotpotQA Datasets")
        print("="*60)
        
        for file_type in self.URLS.keys():
            self.download_file(file_type, force_download)
            print()
        
        # Print summary
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        for file_type, status in self.download_status.items():
            status_icon = "✓" if status["status"] in ["downloaded", "cached"] else "✗"
            print(f"{status_icon} {file_type}: {status['status']}")
        
        return self.download_status
    
    def process_hotpotqa_data(self, file_type: str, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Process raw HotpotQA JSON into simplified format for experiments
        
        Args:
            file_type: Which dataset to process ('train', 'dev_distractor', 'dev_fullwiki')
            sample_size: Number of samples to extract (None for all)
            
        Returns:
            List of processed samples in simplified format
        """
        # Get the raw file path
        if file_type not in self.URLS:
            print(f"Unknown file type: {file_type}")
            return []
        
        filename = self.URLS[file_type].split('/')[-1]
        raw_file = self.raw_dir / filename
        
        if not raw_file.exists():
            print(f"Raw file not found: {raw_file}")
            print("Please download the dataset first using download_file()")
            return []
        
        print(f"\nProcessing {filename}...")
        
        # Load raw JSON
        with open(raw_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"Loaded {len(raw_data)} raw samples")
        
        # Process each sample
        processed_samples = []
        
        for idx, item in enumerate(tqdm(raw_data, desc="Processing samples")):
            # Extract basic information
            processed = {
                "id": item.get("_id", f"hotpot_{idx}"),
                "question": item.get("question", ""),
                "type": "multihop",
                "dataset": "hotpotqa",
                "split": file_type
            }
            
            # Add answer if available (test set doesn't have answers)
            if "answer" in item:
                processed["answer"] = item["answer"]
            else:
                processed["answer"] = None  # Test set has no answers
            
            # Add supporting facts if available
            if "supporting_facts" in item:
                processed["supporting_facts"] = item["supporting_facts"]
            
            # Process context (paragraphs)
            if "context" in item:
                processed["context"] = []
                for title, sentences in item["context"]:
                    processed["context"].append({
                        "title": title,
                        "sentences": sentences,
                        "num_sentences": len(sentences)
                    })
            
            # Add question type and level if available
            if "type" in item:
                processed["question_type"] = item["type"]  # "bridge" or "comparison"
            if "level" in item:
                processed["level"] = item["level"]  # "easy", "medium", "hard"
            
            processed_samples.append(processed)
            
            # Stop if we've reached sample_size
            if sample_size and len(processed_samples) >= sample_size:
                break
        
        # Save processed data
        output_file = self.processed_dir / f"hotpotqa_{file_type}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_samples, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(processed_samples)} processed samples to {output_file}")
        
        return processed_samples
    
    def create_experiment_dataset(self, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Create a simplified dataset suitable for CoT experiments
        
        MODIFIED: Now saves the FULL processed dataset, not just 50 samples.
        Sampling will be handled by DataLoader's dynamic sampling.
        
        Args:
            sample_size: If provided, only extract this many samples (legacy support)
            
        Returns:
            List of samples in experiment-ready format (FULL dataset)
        """
        print("\n" + "="*60)
        print("Creating HotpotQA Dataset")
        print("="*60)
        
        # Ensure dev distractor is downloaded and processed
        dev_file = self.processed_dir / "hotpotqa_dev_distractor_processed.json"
        
        if not dev_file.exists():
            print("Processing dev_distractor dataset...")
            self.download_file("dev_distractor")
            samples = self.process_hotpotqa_data("dev_distractor", sample_size=None)  # Process ALL samples
        else:
            print(f"Loading from cache: {dev_file}")
            with open(dev_file, 'r') as f:
                samples = json.load(f)
        
        # Convert to experiment format (consistent with other datasets)
        experiment_samples = []
        for item in samples:
            experiment_samples.append({
                "question": item["question"],
                "answer": item["answer"],
                "type": "multihop",
                "dataset": "hotpotqa",
                "id": item["id"],
                "question_type": item.get("question_type", "unknown"),
                "level": item.get("level", "unknown")
            })
        
        # Save FULL experiment-ready format (NOT sampled)
        exp_file = self.processed_dir / f"hotpot_qa.json"
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Created full HotpotQA dataset with {len(experiment_samples)} samples")
        print(f"✓ Saved to: {exp_file}")
        
        # Also save a sampled version for legacy support (if sample_size provided)
        if sample_size and sample_size < len(experiment_samples):
            random.seed(42)
            sampled = random.sample(experiment_samples, sample_size)
            sampled_file = self.processed_dir / f"hotpotqa_experiment_{sample_size}.json"
            with open(sampled_file, 'w', encoding='utf-8') as f:
                json.dump(sampled, f, indent=2, ensure_ascii=False)
            print(f"✓ Also saved sampled version ({sample_size} samples) to: {sampled_file}")
        
        # Print statistics
        if experiment_samples:
            print("\nDataset Statistics:")
            question_types = {}
            levels = {}
            for s in experiment_samples:
                qt = s.get("question_type", "unknown")
                question_types[qt] = question_types.get(qt, 0) + 1
                lvl = s.get("level", "unknown")
                levels[lvl] = levels.get(lvl, 0) + 1
            
            print(f"  Total samples: {len(experiment_samples)}")
            print(f"  Question types: {question_types}")
            print(f"  Difficulty levels: {levels}")
        
        return experiment_samples


def create_simplified_hotpotqa_loader():
    """
    Factory function to create a HotpotQA loader compatible with DataLoader interface
    """
    downloader = HotpotQADownloader()
    
    def load_hotpotqa(sample_size: int = 50, use_local_cache: bool = True) -> List[Dict]:
        """
        Load HotpotQA dataset in experiment format
        
        Args:
            sample_size: Number of samples to load (NOT used if loading full dataset)
            use_local_cache: Use cached processed data if available
            
        Returns:
            List of samples in experiment format (FULL dataset for dynamic sampling)
        """
        # Try to load the FULL dataset first
        full_file = downloader.processed_dir / "hotpot_qa.json"
        
        if use_local_cache and full_file.exists():
            print(f"✓ Loading full HotpotQA dataset from cache: {full_file}")
            with open(full_file, 'r') as f:
                samples = json.load(f)
            print(f"✓ Loaded {len(samples)} total samples (sampling will be done by DataLoader)")
            return samples
        
        # If full dataset not found, create it
        print("Full HotpotQA dataset not found. Creating...")
        samples = downloader.create_experiment_dataset(sample_size=None)
        return samples
    
    return load_hotpotqa


def main():
    """Main function to download and process HotpotQA"""
    print("HotpotQA Data Download and Processing Tool")
    print("="*60)
    
    downloader = HotpotQADownloader()
    
    # Ask what to do
    print("\nOptions:")
    print("1. Download all HotpotQA datasets")
    print("2. Download and process dev set (create FULL dataset)")
    print("3. Create sampled experiment dataset only (if already downloaded)")
    print("4. Show download status")
    
    choice = input("\nSelect option (1-4, default 2): ").strip() or "2"
    
    if choice == "1":
        # Download all datasets
        force = input("Force re-download? (y/n, default n): ").lower().strip() == 'y'
        downloader.download_all(force_download=force)
        
    elif choice == "2":
        # Download and process dev set - create FULL dataset
        print("\nDownloading and processing dev_distractor dataset...")
        downloader.download_file("dev_distractor")
        samples = downloader.process_hotpotqa_data("dev_distractor", sample_size=None)
        print(f"\n✓ Processed {len(samples)} samples")
        
        # Create FULL experiment dataset (not sampled)
        exp_samples = downloader.create_experiment_dataset(sample_size=None)
        
        # Show sample
        if exp_samples:
            print("\nSample from dataset:")
            print("-" * 40)
            sample = exp_samples[0]
            print(f"Question: {sample['question'][:200]}...")
            print(f"Answer: {sample['answer']}")
            print(f"Type: {sample.get('question_type', 'N/A')}")
            print(f"Level: {sample.get('level', 'N/A')}")
        
    elif choice == "3":
        # Create sampled experiment dataset only
        sample_size = int(input("Sample size (default 50): ") or "50")
        samples = downloader.create_experiment_dataset(sample_size=sample_size)
        
    elif choice == "4":
        # Show status
        for file_type in downloader.URLS.keys():
            filename = downloader.URLS[file_type].split('/')[-1]
            filepath = downloader.raw_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / 1024 / 1024
                print(f"✓ {file_type}: {filename} ({size_mb:.2f} MB)")
            else:
                print(f"✗ {file_type}: Not downloaded")
    
    print("\n" + "="*60)
    print("HotpotQA data is ready for experiments!")
    print(f"Processed files are in: {downloader.processed_dir}")


if __name__ == "__main__":
    main()