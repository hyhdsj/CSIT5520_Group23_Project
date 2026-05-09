#!/usr/bin/env python
"""
Main entry point for running the Chain-of-Thought prompting experiment
"""
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.experiment_runner import ExperimentRunner


def main():
    """Main function to run the experiment"""
    print("="*70)
    print("CSIT5520 - Chain-of-Thought Prompting Experiment")
    print("Exploring In-Context Learning for Complex Reasoning Tasks")
    print("="*70)
    
    # Validate configuration
    if not Config.validate():
        print("\n❌ Configuration Error: Please check your .env file")
        print("Make sure AZURE_OPENAI_API_KEY is set correctly")
        return
    
    print(f"\n✓ Configuration validated")
    print(f"  Endpoint: {Config.AZURE_ENDPOINT}")
    print(f"  Deployment: {Config.AZURE_DEPLOYMENT}")
    print(f"  Sample size: {Config.SAMPLE_SIZE}")
    
    # Run experiments
    runner = ExperimentRunner()
    
    try:
        results = runner.run_all_experiments()
        
        print("\n" + "="*70)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nResults saved in the 'results/' directory")
        print("Check the experiment summary CSV and report text file for details")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Experiment interrupted by user")
        print("Partial results may have been saved")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()