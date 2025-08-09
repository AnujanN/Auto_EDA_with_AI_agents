#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crew import EDACrew
from tools import clear_analysis_results

def run_eda_analysis(dataset_path: str, target_column: str = None):
    """
    Run the automated EDA analysis using CrewAI
    
    Args:
        dataset_path (str): Path to the dataset file
        target_column (str, optional): Name of the target column
    
    Returns:
        dict: Results from the EDA analysis
    """
    
    # Clear any previous analysis results
    clear_analysis_results()
    
    # Initialize the crew
    eda_crew = EDACrew()
    
    # Prepare inputs for the crew
    inputs = {
        'dataset_path': dataset_path,
        'target_column': target_column or '',
        'eda_results': ''  # Will be filled by the first task
    }
    
    try:
        # Run the crew
        result = eda_crew.crew().kickoff(inputs=inputs)
        
        return {
            'success': True,
            'result': result,
            'message': 'EDA analysis completed successfully!'
        }
        
    except Exception as e:
        return {
            'success': False,
            'result': None,
            'message': f'Error during EDA analysis: {str(e)}'
        }

def main():
    """Main function for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset_path> [target_column]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    target_column = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        sys.exit(1)
    
    print("🚀 Starting automated EDA analysis...")
    print(f"📁 Dataset: {dataset_path}")
    if target_column:
        print(f"🎯 Target column: {target_column}")
    
    # Run the analysis
    result = run_eda_analysis(dataset_path, target_column)
    
    if result['success']:
        print("\n✅ EDA Analysis completed successfully!")
        print("\n📋 Results:")
        print(result['result'])
    else:
        print(f"\n❌ Analysis failed: {result['message']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
