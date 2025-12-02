"""
Main Execution Script for Surrogate Model-Based Building Energy Optimization

This script runs the complete pipeline:
1. Data curation and preprocessing (Phase 1)
2. Surrogate model training (Phase 2)  
3. GA-based optimization (Phase 3)
4. Results analysis and visualization (Phase 4 & 5)

Usage:
    python main.py [--phase PHASE] [--building BUILDING_ID]

Arguments:
    --phase: Run specific phase (1-5) or 'all' (default: 'all')
    --building: Specific building ID to process (default: all buildings)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_processing import main as run_phase1
from surrogate_models import main as run_phase2
from optimization import main as run_phase3
from visualization import main as run_visualization


def main():
    """Run complete pipeline or specific phases."""
    
    parser = argparse.ArgumentParser(
        description='Surrogate Model-Based Building Energy Optimization Pipeline'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='all',
        choices=['1', '2', '3', '4', '5', 'all'],
        help='Phase to run (1: data, 2: models, 3: optimization, 4: analysis, 5: visualization, all: complete pipeline)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SURROGATE MODEL-BASED BUILDING ENERGY OPTIMIZATION")
    print("Using BDG2 Dataset with GA-Based Multi-Objective Optimization")
    print("="*80)
    
    phases_to_run = []
    
    if args.phase == 'all':
        phases_to_run = ['1', '2', '3', '5']
    else:
        phases_to_run = [args.phase]
    
    try:
        for phase in phases_to_run:
            if phase == '1':
                print("\n" + "="*80)
                print("PHASE 1: DATA CURATION AND PRE-PROCESSING")
                print("="*80)
                run_phase1()
                
            elif phase == '2':
                print("\n" + "="*80)
                print("PHASE 2: SURROGATE MODEL DEVELOPMENT")
                print("="*80)
                run_phase2()
                
            elif phase == '3':
                print("\n" + "="*80)
                print("PHASE 3: GA-BASED OPTIMIZATION")
                print("="*80)
                run_phase3()
                
            elif phase == '4':
                print("\n" + "="*80)
                print("PHASE 4: RESULTS ANALYSIS")
                print("="*80)
                print("(Analysis integrated in Phase 3)")
                
            elif phase == '5':
                print("\n" + "="*80)
                print("PHASE 5: PUBLICATION-QUALITY VISUALIZATIONS")
                print("="*80)
                run_visualization()
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*80)
        print("\nResults Summary:")
        print("  Data: /workspace/data/")
        print("  Models: /workspace/models/")
        print("  Results: /workspace/results/")
        print("  Figures: /workspace/figures/")
        print("\nFor detailed results, see:")
        print("  - /workspace/results/optimization_results.json")
        print("  - /workspace/results/model_training_results.json")
        print("  - /workspace/figures/*.png")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
