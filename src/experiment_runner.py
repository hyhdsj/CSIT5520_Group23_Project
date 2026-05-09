"""
Experiment runner - Coordinates the entire experiment pipeline
Runs all prompt strategies across all datasets and collects results
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.config import Config
from src.data_loader import DataLoader
from src.prompt_builder import PromptBuilder
from src.model_client import ModelClient
from src.evaluator import Evaluator
from src.reasoning_analyzer import ReasoningAnalyzer


class ExperimentRunner:
    """
    Main experiment orchestrator
    
    Runs experiments with:
    - 3 datasets (math, commonsense, multihop)
    - 3 prompt strategies (standard, zero-shot CoT, few-shot CoT)
    - Collects accuracy, F1, and reasoning quality metrics
    """
    
    def __init__(self):
        """Initialize experiment runner with all components"""
        self.config = Config()
        self.data_loader = DataLoader(sample_size=Config.SAMPLE_SIZE)
        self.model_client = ModelClient()
        self.evaluator = Evaluator()
        self.analyzer = ReasoningAnalyzer()
        
        # Create results directory
        self.results_dir = Path(Config.RESULTS_DIR) if hasattr(Config, 'RESULTS_DIR') else Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Store all results
        self.all_results = []
    
    def run_single_experiment(
        self, 
        dataset_name: str, 
        dataset: List[Dict], 
        prompt_type: str
    ) -> pd.DataFrame:
        """
        Run experiment for one dataset and one prompt type
        
        Args:
            dataset_name: Name of the dataset (math, commonsense, multihop)
            dataset: List of samples with questions and answers
            prompt_type: Type of prompt to use
            
        Returns:
            DataFrame with results
        """
        print(f"\n{'='*60}")
        print(f"Running: {dataset_name} - {prompt_type}")
        print(f"Samples: {len(dataset)}")
        print(f"{'='*60}")
        
        results = []
        
        for i, item in enumerate(dataset):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(dataset)}")
            
            # Build prompt
            prompt = PromptBuilder.get_prompt(prompt_type, item["question"])
            
            try:
                # Generate response
                response = self.model_client.generate(prompt)

                # Always evaluate WITHOUT choices - let the extractor handle everything
                # The extractor now works for all formats regardless of choices
                f1_score, exact_match = self.evaluator.evaluate_response(
                    response, item["answer"]
                )
                
                # Analyze reasoning
                reasoning_steps = self.analyzer.count_reasoning_steps(response)
                reasoning_structure = self.analyzer.has_reasoning_structure(response)
                
                # Store result
                result = {
                    "dataset": dataset_name,
                    "prompt_type": prompt_type,
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "model_response": response,
                    "f1_score": f1_score,
                    "exact_match": exact_match,
                    "reasoning_steps": reasoning_steps,
                    **{f"has_{k}": v for k, v in reasoning_structure.items()}
                }
                results.append(result)
                
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{dataset_name}_{prompt_type}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        return df
    
    def run_all_experiments(self) -> Dict:
        """
        Run all experiments across all datasets and prompt types
        
        Returns:
            Dictionary with all results and summary metrics
        """
        print("\n" + "="*70)
        print("STARTING COMPLETE EXPERIMENT")
        print(f"Model: {self.config.AZURE_DEPLOYMENT}")
        print(f"Sample size per dataset: {self.config.SAMPLE_SIZE}")
        print(f"Temperature: {self.config.TEMPERATURE}")
        print("="*70)
        
        # Load all datasets
        datasets = self.data_loader.load_all_datasets()
        
        # Prompt strategies to test
        prompt_types = ["standard", "zero_shot_cot", "few_shot_cot"]
        
        # Run experiments for each combination
        all_results = {}
        
        for dataset_name, dataset_data in datasets.items():
            if not dataset_data:
                print(f"\n⚠️ No data for {dataset_name}, skipping...")
                continue
            
            all_results[dataset_name] = {}
            
            for prompt_type in prompt_types:
                df = self.run_single_experiment(dataset_name, dataset_data, prompt_type)
                all_results[dataset_name][prompt_type] = df
                self.all_results.append(df)
        
        # Generate comprehensive report
        self.generate_report(all_results)
        
        return all_results
    
    def generate_report(self, all_results: Dict) -> None:
        """
        Generate comprehensive experiment report
        
        Args:
            all_results: Dictionary containing all experiment DataFrames
        """
        print("\n" + "="*70)
        print("GENERATING EXPERIMENT REPORT")
        print("="*70)
        
        # Collect all metrics
        summary_data = []
        
        for dataset_name, prompt_results in all_results.items():
            for prompt_type, df in prompt_results.items():
                if not df.empty:
                    metrics = self.evaluator.calculate_metrics(df.to_dict('records'))
                    reasoning_stats = self.analyzer.analyze_reasoning_quality(
                        df['model_response'].tolist()
                    )
                    
                    summary_data.append({
                        "dataset": dataset_name,
                        "prompt_type": prompt_type,
                        "exact_match": metrics["exact_match"],
                        "f1_score": metrics["f1_score"],
                        "total_samples": metrics["total_samples"],
                        "avg_reasoning_steps": reasoning_stats["avg_steps"],
                        "has_step_keywords_pct": reasoning_stats["structure_stats"].get("has_step_keywords", 0),
                        "has_conclusion_pct": reasoning_stats["structure_stats"].get("has_conclusion", 0)
                    })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"experiment_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to: {summary_file}")
        
        # Print summary table
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
        
        # Save detailed report
        report_file = self.results_dir / f"report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CHAIN-OF-THOUGHT PROMPTING EXPERIMENT REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {self.config.AZURE_DEPLOYMENT}\n")
            f.write(f"Sample size per dataset: {self.config.SAMPLE_SIZE}\n")
            f.write(f"Temperature: {self.config.TEMPERATURE}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("RESULTS SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            # Add per-dataset detailed analysis
            for dataset_name, prompt_results in all_results.items():
                f.write(f"\n{'='*70}\n")
                f.write(f"DATASET: {dataset_name.upper()}\n")
                f.write(f"{'='*70}\n")
                
                for prompt_type, df in prompt_results.items():
                    if not df.empty:
                        f.write(f"\n{prompt_type.upper()}:\n")
                        f.write(f"  Exact Match: {df['exact_match'].mean():.4f}\n")
                        f.write(f"  F1 Score: {df['f1_score'].mean():.4f}\n")
                        f.write(f"  Avg Reasoning Steps: {df['reasoning_steps'].mean():.2f}\n")
                        
                        # Show example response
                        if len(df) > 0:
                            f.write(f"\n  Example Response:\n")
                            f.write(f"  Question: {df.iloc[0]['question'][:150]}...\n")
                            f.write(f"  Response: {df.iloc[0]['model_response'][:200]}...\n")
        
        print(f"\n✓ Detailed report saved to: {report_file}")