#!/usr/bin/env python3
"""
Main pipeline script to run the complete analysis workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_phase1():
    """Run Phase 1: Data Curation and Pre-Processing."""
    print("\n" + "=" * 80)
    print("PHASE 1: DATA CURATION AND PRE-PROCESSING")
    print("=" * 80)
    
    from phase1_data_curation import building_selection, data_integration, data_cleaning
    
    print("\n1.1 Building Selection...")
    building_selection.main()
    
    print("\n1.2 Data Integration...")
    data_integration.main()
    
    print("\n1.3 Data Cleaning...")
    data_cleaning.main()
    
    print("\nPhase 1 completed!")

def run_phase2():
    """Run Phase 2: Surrogate Model Development."""
    print("\n" + "=" * 80)
    print("PHASE 2: SURROGATE MODEL DEVELOPMENT")
    print("=" * 80)
    
    from phase2_surrogate import train_lstm, train_xgboost
    
    print("\n2.1 Training LSTM Model...")
    try:
        train_lstm.main()
    except Exception as e:
        print(f"Warning: LSTM training failed ({e}). Continuing with XGBoost...")
    
    print("\n2.2 Training XGBoost Model...")
    try:
        train_xgboost.main()
    except Exception as e:
        print(f"Warning: XGBoost training failed ({e})")

def run_phase3():
    """Run Phase 3: Optimization Framework."""
    print("\n" + "=" * 80)
    print("PHASE 3: OPTIMIZATION FRAMEWORK")
    print("=" * 80)
    
    from phase3_optimization import ga_optimizer
    
    print("\n3.1 Running GA Optimization...")
    ga_optimizer.main()

def run_phase4():
    """Run Phase 4: Results and Analysis."""
    print("\n" + "=" * 80)
    print("PHASE 4: RESULTS AND ANALYSIS")
    print("=" * 80)
    
    from phase4_results import comparative_analysis, generate_figures
    
    print("\n4.1 Comparative Analysis...")
    comparative_analysis.main()
    
    print("\n4.2 Generating Figures...")
    generate_figures.main()
    
    print("\nPhase 4 completed!")

def main():
    """Run complete pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run BDG2 Surrogate-Assisted Optimization Pipeline')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                       help='Run specific phase (1-4)')
    parser.add_argument('--all', action='store_true',
                       help='Run all phases')
    
    args = parser.parse_args()
    
    if args.all or args.phase is None:
        # Run all phases
        run_phase1()
        run_phase2()
        run_phase3()
        run_phase4()
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    elif args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2()
    elif args.phase == 3:
        run_phase3()
    elif args.phase == 4:
        run_phase4()

if __name__ == "__main__":
    main()
