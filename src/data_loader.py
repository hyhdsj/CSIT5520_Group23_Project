"""
Data loading module - Load datasets from local files
"""
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Union


class DataLoader:
    """
    Load datasets from local JSON files
    
    This class prioritizes loading from local files first.
    If local files don't exist, it provides fallback options.
    """
    
    def __init__(self, data_dir: str = "data", sample_size: int = 50, seed: int = 42):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing downloaded data
            sample_size: Number of samples to load
            seed: Random seed for sampling
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.sample_size = sample_size
        self.seed = seed
        random.seed(seed)
    
    def check_data_availability(self) -> Dict[str, bool]:
        """
        Check which datasets are available locally
        
        Returns:
            Dictionary with dataset availability status
        """
        return {
            "gsm8k_main": (self.processed_dir / "gsm8k_main.json").exists(),
            "gsm8k_socratic": (self.processed_dir / "gsm8k_socratic.json").exists(),
            "commonsense_qa": (self.processed_dir / "commonsense_qa.json").exists(),
            "hotpotqa": (self.processed_dir / "hotpotqa.json").exists()
        }
    
    def load_gsm8k(self, config: str = "main", use_local_cache: bool = True) -> List[Dict]:
        """
        Load GSM8K dataset from local file or Hugging Face
        
        Args:
            config: Dataset configuration ("main" or "socratic")
            use_local_cache: If True, load from local file first
            
        Returns:
            List of samples
        """
        print(f"Loading GSM8K dataset (config: {config})...")
        
        # Try loading from local file first
        if use_local_cache:
            local_file = self.processed_dir / f"gsm8k_{config}.json"
            if local_file.exists():
                print(f"✓ Loading from local cache: {local_file}")
                with open(local_file, 'r') as f:
                    data = json.load(f)
                
                # Sample if needed
                if len(data) > self.sample_size:
                    indices = random.sample(range(len(data)), self.sample_size)
                    data = [data[i] for i in indices]
                
                # Extract answers
                for item in data:
                    item["answer"] = self._extract_gsm8k_answer(item["answer"])
                
                print(f"✓ Loaded {len(data)} samples from local cache")
                return data
        
        # Fallback to downloading from Hugging Face
        print("Local cache not found. Downloading from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", config, split="train")
        
        # Sample
        indices = random.sample(range(len(dataset)), min(self.sample_size, len(dataset)))
        samples = []
        
        for idx in indices:
            item = dataset[idx]
            samples.append({
                "question": item["question"],
                "answer": self._extract_gsm8k_answer(item["answer"]),
                "full_answer": item["answer"],
                "type": "math",
                "dataset": "gsm8k",
                "config": config
            })
        
        print(f"✓ Loaded {len(samples)} samples from Hugging Face")
        return samples
    
    def load_commonsenseqa(self, use_local_cache: bool = True) -> List[Dict]:
        """
        Load CommonsenseQA dataset from local file or Hugging Face
        
        Args:
            use_local_cache: If True, load from local file first
            
        Returns:
            List of samples with formatted questions and choices
        """
        print("Loading CommonsenseQA dataset...")
        
        # Try loading from local file first
        if use_local_cache:
            local_file = self.processed_dir / "commonsense_qa.json"
            if local_file.exists():
                print(f"✓ Loading from local cache: {local_file}")
                with open(local_file, 'r') as f:
                    data = json.load(f)
                
                # Sample if needed
                if len(data) > self.sample_size:
                    indices = random.sample(range(len(data)), self.sample_size)
                    data = [data[i] for i in indices]
                
                # Format questions with choices
                samples = []
                for item in data:
                    choices_text = "\n".join([
                        f"{item['choices_labels'][i]}. {choice}"
                        for i, choice in enumerate(item['choices'])
                    ])
                    question_with_choices = f"{item['question']}\n\n{choices_text}"
                    
                    # Get correct answer
                    answer_idx = item['choices_labels'].index(item['answer_key'])
                    answer_text = item['choices'][answer_idx]
                    
                    samples.append({
                        "question": question_with_choices,
                        "answer": answer_text,
                        "answer_key": item['answer_key'],
                        "choices": item['choices'],  
                        "type": "commonsense",
                        "dataset": "commonsense_qa"
                    })
                
                print(f"✓ Loaded {len(samples)} samples from local cache")
                return samples
        
        # Fallback to downloading from Hugging Face
        print("Local cache not found. Downloading from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset("tau/commonsense_qa", split="validation")
        
        indices = random.sample(range(len(dataset)), min(self.sample_size, len(dataset)))
        samples = []
        
        for idx in indices:
            item = dataset[idx]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            
            choices_text = "\n".join([f"{labels[i]}. {choice}" for i, choice in enumerate(choices)])
            question_with_choices = f"{item['question']}\n\n{choices_text}"
            
            answer_idx = labels.index(item["answerKey"])
            answer_text = choices[answer_idx]
            
            samples.append({
                "question": question_with_choices,
                "answer": answer_text,
                "answer_key": item["answerKey"],
                "choices": choices,  
                "type": "commonsense",
                "dataset": "commonsense_qa"
            })
        
        print(f"✓ Loaded {len(samples)} samples from Hugging Face")
        return samples
    
    def load_hotpotqa(self, use_local_cache: bool = True) -> List[Dict]:
        """
        Load HotpotQA dataset from local files
        
        Uses the official HotpotQA dataset downloaded from:
        http://curtis.ml.cmu.edu/datasets/hotpot/
        
        Args:
            use_local_cache: If True, load from local file first
            
        Returns:
            List of samples in experiment format (FULL dataset for dynamic sampling)
        """
        print("Loading HotpotQA dataset...")
        
        # Try loading the FULL dataset from cache first
        if use_local_cache:
            # Look for the full dataset file (not the sampled one)
            full_file = self.processed_dir / "hotpot_qa.json"
            
            if full_file.exists():
                print(f"✓ Loading full HotpotQA dataset from: {full_file}")
                with open(full_file, 'r') as f:
                    data = json.load(f)
                
                # Dynamic sampling: sample if needed
                if len(data) > self.sample_size:
                    indices = random.sample(range(len(data)), self.sample_size)
                    data = [data[i] for i in indices]
                    print(f"✓ Loaded {len(data)} samples (sampled from {len(data) if len(data) <= self.sample_size else 'full'})")
                else:
                    print(f"✓ Loaded all {len(data)} samples")
                
                return data
        
        # If full dataset not found, use the HotpotQA downloader
        print("Full HotpotQA dataset not found. Running HotpotQA downloader...")
        try:
            # Try relative import first
            from .hotpotqa_downloader import HotpotQADownloader
        except ImportError:
            # Fallback to absolute import
            from hotpotqa_downloader import HotpotQADownloader
        
        # Use the same data directory
        downloader = HotpotQADownloader(data_dir=str(self.data_dir))
        
        # This now returns the FULL dataset (not sampled)
        full_samples = downloader.create_experiment_dataset(sample_size=None)
        
        # Apply dynamic sampling
        if len(full_samples) > self.sample_size:
            indices = random.sample(range(len(full_samples)), self.sample_size)
            full_samples = [full_samples[i] for i in indices]
            print(f"✓ Loaded {len(full_samples)} samples (sampled from full dataset)")
        else:
            print(f"✓ Loaded all {len(full_samples)} samples")
        
        return full_samples
    
    def _extract_gsm8k_answer(self, full_answer: str) -> str:
        """Extract numerical answer from GSM8K format"""
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', full_answer)
        if match:
            return match.group(1).replace(',', '')
        
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', full_answer)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return full_answer.strip()
    
    def _get_hotpotqa_fallback(self) -> List[Dict]:
        """Provide fallback samples for HotpotQA"""
        fallback_samples = [
            {
                "question": "Which film studio produced the movie directed by Christopher Nolan that won the Academy Award for Best Visual Effects?",
                "answer": "Warner Bros.",
                "type": "multihop",
                "dataset": "hotpotqa_fallback"
            },
            {
                "question": "What is the name of the university that Albert Einstein attended after finishing his secondary education in Switzerland?",
                "answer": "ETH Zurich",
                "type": "multihop",
                "dataset": "hotpotqa_fallback"
            },
            {
                "question": "The city that hosted the 2016 Summer Olympics is located in which country?",
                "answer": "Brazil",
                "type": "multihop",
                "dataset": "hotpotqa_fallback"
            }
        ]
        return fallback_samples[:self.sample_size]
    
    def load_all_datasets(self, use_local_cache: bool = True) -> Dict[str, List[Dict]]:
        """
        Load all three datasets
        
        Args:
            use_local_cache: If True, load from local files first
            
        Returns:
            Dictionary with dataset names as keys
        """
        datasets = {}
        
        # Check availability
        availability = self.check_data_availability()
        print("\n" + "="*60)
        print("DATA AVAILABILITY CHECK")
        print("="*60)
        for name, available in availability.items():
            status = "✓" if available else "✗"
            print(f"{status} {name}")
        
        # Load datasets
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        datasets["math"] = self.load_gsm8k(config="main", use_local_cache=use_local_cache)
        datasets["commonsense"] = self.load_commonsenseqa(use_local_cache=use_local_cache)
        datasets["multihop"] = self.load_hotpotqa(use_local_cache=use_local_cache)
        
        return datasets
    
    def get_data_info(self) -> Dict:
        """
        Get information about loaded data
        
        Returns:
            Dictionary with data statistics
        """
        availability = self.check_data_availability()
        
        info = {
            "data_directory": str(self.data_dir),
            "processed_directory": str(self.processed_dir),
            "sample_size": self.sample_size,
            "available_datasets": availability,
            "files": []
        }
        
        # List all files in processed directory
        if self.processed_dir.exists():
            for file in self.processed_dir.glob("*.json"):
                size_kb = file.stat().st_size / 1024
                info["files"].append({
                    "name": file.name,
                    "size_kb": round(size_kb, 2),
                    "path": str(file)
                })
        
        return info