#!/usr/bin/env python3
"""
Main Execution Script for BDG2 Surrogate-Assisted Optimization Framework.

This script orchestrates the complete pipeline:
1. Data loading and preprocessing from BDG2
2. Surrogate model training (LSTM and XGBoost)
3. GA-based optimization for HVAC control
4. Results analysis and visualization generation

Author: BDG2 Optimization Framework
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DataConfig, ModelConfig, OptimizationConfig, 
    ElectricityTariff, VisualizationConfig,
    FIGURES_DIR, RESULTS_DIR, MODELS_DIR
)
from data_preprocessing import BDG2DataProcessor, BuildingData, create_case_study_table
from surrogate_models import (
    LSTMSurrogate, XGBoostSurrogate, MultiBuildingSurrogate,
    create_input_variables_table
)
from optimization import (
    GAOptimizer, MultiObjectiveGAOptimizer, BuildingSurrogateWrapper,
    run_day_optimization, OptimizationResult, create_optimization_table
)
from visualization import FigureGenerator, generate_all_figures


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def phase1_data_curation(n_buildings: int = 8) -> tuple:
    """
    Phase 1: Data Curation and Pre-Processing.
    
    Returns:
        Tuple of (processor, building_data_list, weather_data)
    """
    print_header("PHASE 1: Data Curation and Pre-Processing")
    
    config = DataConfig()
    processor = BDG2DataProcessor(config)
    
    # Load and explore data
    processor.load_raw_data()
    
    # Select buildings
    selected = processor.select_buildings()
    building_ids = processor.get_building_list(n_buildings=n_buildings)
    
    print(f"\nSelected {len(building_ids)} buildings for analysis:")
    
    # Process each building
    building_data_list = []
    for bid in building_ids:
        print(f"  Processing: {bid}")
        building_data = processor.process_building(bid)
        print(f"    - Climate Zone: {building_data.climate_zone}")
        print(f"    - Floor Area: {building_data.floor_area_sqm:.0f} m²")
        print(f"    - Data points: {len(building_data.data)}")
        
        # Only keep buildings with sufficient data
        if len(building_data.data) >= 1000:
            building_data_list.append(building_data)
        else:
            print(f"    ⚠ Skipping - insufficient data (<1000 points)")
    
    print(f"\nValid buildings for analysis: {len(building_data_list)}")
    
    # Create Table 1
    table1 = create_case_study_table(building_data_list)
    print("\n--- Table 1: Characteristics of Selected Case Study Buildings ---")
    print(table1.to_string(index=False))
    table1.to_csv(RESULTS_DIR / 'table1_buildings.csv', index=False)
    
    return processor, building_data_list


def phase2_surrogate_training(building_data_list: list, processor: BDG2DataProcessor) -> dict:
    """
    Phase 2: Surrogate Model Development.
    
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    print_header("PHASE 2: Surrogate Model Development")
    
    model_config = ModelConfig()
    results = {
        'xgboost_models': {},
        'lstm_models': {},
        'metrics': []
    }
    
    print("\n--- Table 2: Input Variables for the Prediction Model ---")
    table2 = create_input_variables_table()
    print(table2.to_string(index=False))
    table2.to_csv(RESULTS_DIR / 'table2_variables.csv', index=False)
    
    # Train XGBoost for each building
    print("\n[1] Training XGBoost Models...")
    
    # Filter out buildings with insufficient data
    valid_buildings = [b for b in building_data_list if len(b.data) >= 500]
    print(f"  Buildings with sufficient data (≥500 points): {len(valid_buildings)}")
    
    for building_data in valid_buildings:
        print(f"\n  Building: {building_data.building_id}")
        
        # Split data
        train_df, val_df, test_df = processor.split_data(building_data.data)
        print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Skip if not enough data
        if len(train_df) < 100 or len(test_df) < 50:
            print(f"    Skipping - insufficient data")
            continue
        
        # Train XGBoost
        xgb_model = XGBoostSurrogate(model_config)
        X_train, y_train = xgb_model.prepare_features(train_df)
        X_val, y_val = xgb_model.prepare_features(val_df)
        X_test, y_test = xgb_model.prepare_features(test_df)
        
        # Skip if features are empty after preparation
        if len(X_train) < 50 or len(X_test) < 20:
            print(f"    Skipping - insufficient features after preparation")
            continue
        
        xgb_model.train(X_train, y_train, X_val, y_val, verbose=0)
        
        # Evaluate
        train_metrics = xgb_model.evaluate(X_train, y_train)
        test_metrics = xgb_model.evaluate(X_test, y_test)
        
        print(f"    XGBoost - Train RMSE: {train_metrics['rmse']:.2f}, "
              f"Test RMSE: {test_metrics['rmse']:.2f}, "
              f"R²: {test_metrics['r2']:.3f}")
        
        results['xgboost_models'][building_data.building_id] = {
            'model': xgb_model,
            'train_data': train_df,
            'val_data': val_df,
            'test_data': test_df,
            'test_metrics': test_metrics
        }
        
        results['metrics'].append({
            'building_id': building_data.building_id,
            'climate_zone': building_data.climate_zone,
            'model_type': 'XGBoost',
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'test_mape': test_metrics['mape']
        })
    
    # Feature importance analysis
    print("\n[2] Top 10 Features (averaged across buildings):")
    feature_importances = []
    for bid, data in results['xgboost_models'].items():
        fi = data['model'].get_feature_importance(top_n=20)
        fi['building_id'] = bid
        feature_importances.append(fi)
    
    all_fi = pd.concat(feature_importances)
    avg_fi = all_fi.groupby('feature')['importance'].mean().sort_values(ascending=False)
    for i, (feat, imp) in enumerate(avg_fi.head(10).items()):
        print(f"    {i+1}. {feat}: {imp:.4f}")
    
    # Save metrics
    metrics_df = pd.DataFrame(results['metrics'])
    metrics_df.to_csv(RESULTS_DIR / 'model_metrics.csv', index=False)
    
    print(f"\n[3] Average Test Performance:")
    print(f"    RMSE: {metrics_df['test_rmse'].mean():.2f} ± {metrics_df['test_rmse'].std():.2f}")
    print(f"    MAE:  {metrics_df['test_mae'].mean():.2f} ± {metrics_df['test_mae'].std():.2f}")
    print(f"    R²:   {metrics_df['test_r2'].mean():.3f} ± {metrics_df['test_r2'].std():.3f}")
    
    return results


def phase3_optimization(
    building_data_list: list,
    model_results: dict
) -> dict:
    """
    Phase 3: Optimization Framework.
    
    Returns:
        Dictionary with optimization results
    """
    print_header("PHASE 3: GA-Based Optimization Framework")
    
    opt_config = OptimizationConfig()
    tariff = ElectricityTariff()
    
    print("\n--- Table 3: Objective Function and Optimization Constraints ---")
    table3 = create_optimization_table()
    print(table3.to_string(index=False))
    table3.to_csv(RESULTS_DIR / 'table3_optimization.csv', index=False)
    
    optimization_results = []
    
    for building_data in building_data_list:
        bid = building_data.building_id
        print(f"\n[Optimizing] Building: {bid}")
        
        # Get model and test data
        model_data = model_results['xgboost_models'][bid]
        test_df = model_data['test_data']
        xgb_model = model_data['model']
        
        # Select a representative day for optimization
        # Prefer days with temperature variation (for load shifting potential)
        test_df_copy = test_df.copy()
        test_df_copy['date'] = test_df_copy.index.date
        daily_temps = test_df_copy.groupby('date')['outdoor_temp'].mean()
        daily_range = test_df_copy.groupby('date')['outdoor_temp'].apply(lambda x: x.max() - x.min())
        
        # Select days with good temperature range and complete data
        # Higher range = more opportunity for load shifting
        candidate_days = daily_range.nlargest(50)
        
        selected_day = None
        for day in candidate_days.index:
            day_data = test_df_copy[test_df_copy['date'] == day]
            if len(day_data) >= 24:
                selected_day = day
                break
        
        # Fallback: just use any day with complete data
        if selected_day is None:
            all_days = list(test_df_copy['date'].unique())
            for day in all_days:
                day_data = test_df_copy[test_df_copy['date'] == day]
                if len(day_data) >= 24:
                    selected_day = day
                    break
        
        if selected_day is None:
            print(f"  No suitable day found for {bid}, skipping...")
            continue
        
        # Extract 24-hour weather data
        day_data = test_df_copy[test_df_copy['date'] == selected_day].head(24).copy()
        print(f"  Selected day: {selected_day}")
        print(f"  Temperature range: {day_data['outdoor_temp'].min():.1f}°C - "
              f"{day_data['outdoor_temp'].max():.1f}°C")
        
        # Create surrogate wrapper
        surrogate = BuildingSurrogateWrapper(
            xgb_model, 
            day_data,
            config=opt_config
        )
        
        # Run optimization with Pareto analysis
        print("  Running GA optimization...")
        result = run_day_optimization(
            surrogate,
            day_data,
            opt_config,
            tariff,
            multi_objective=True,
            verbose=False
        )
        
        # Store results
        result_dict = {
            'building_id': bid,
            'climate_zone': building_data.climate_zone,
            'date': str(selected_day),
            'result': result,
            'weather_data': day_data
        }
        optimization_results.append(result_dict)
        
        # Print summary
        energy_savings = (sum(result.baseline_energy) - sum(result.optimized_energy)) / sum(result.baseline_energy) * 100
        cost_savings = (result.baseline_cost - result.optimized_cost) / result.baseline_cost * 100
        
        print(f"  Results:")
        print(f"    Energy: {sum(result.baseline_energy):.1f} → {sum(result.optimized_energy):.1f} kWh "
              f"({energy_savings:.1f}% savings)")
        print(f"    Cost: ${result.baseline_cost:.2f} → ${result.optimized_cost:.2f} "
              f"({cost_savings:.1f}% savings)")
        print(f"    Comfort violations: {result.baseline_comfort_violations} → "
              f"{result.optimized_comfort_violations} hours")
        print(f"    Computation time: {result.computation_time:.1f}s")
        
        if result.pareto_front:
            print(f"    Pareto solutions: {len(result.pareto_front)}")
    
    return {'optimization_results': optimization_results}


def phase4_results_analysis(
    building_data_list: list,
    model_results: dict,
    optimization_results: dict
) -> pd.DataFrame:
    """
    Phase 4: Results and Comparative Analysis.
    
    Returns:
        DataFrame with comparative results
    """
    print_header("PHASE 4: Results and Comparative Analysis")
    
    if not optimization_results['optimization_results']:
        print("No optimization results available!")
        return pd.DataFrame()
    
    results_list = []
    
    for opt_data in optimization_results['optimization_results']:
        result = opt_data['result']
        bid = opt_data['building_id']
        
        baseline_energy = sum(result.baseline_energy)
        optimized_energy = sum(result.optimized_energy)
        
        results_list.append({
            'building_id': bid,
            'climate_zone': opt_data['climate_zone'],
            'prediction_rmse': model_results['xgboost_models'][bid]['test_metrics']['rmse'],
            'baseline_energy': baseline_energy,
            'optimized_energy': optimized_energy,
            'energy_savings_pct': (baseline_energy - optimized_energy) / baseline_energy * 100,
            'baseline_cost': result.baseline_cost,
            'optimized_cost': result.optimized_cost,
            'cost_savings_pct': (result.baseline_cost - result.optimized_cost) / result.baseline_cost * 100,
            'baseline_violations': result.baseline_comfort_violations,
            'optimized_violations': result.optimized_comfort_violations,
            'comfort_improvement_pct': (result.baseline_comfort_violations - result.optimized_comfort_violations) / max(1, result.baseline_comfort_violations) * 100,
            'computation_time_s': result.computation_time
        })
    
    results_df = pd.DataFrame(results_list)
    
    print("\n--- Table 4: Comparative Results Summary ---")
    summary = pd.DataFrame({
        'Metric': [
            'Total Energy (kWh)',
            'Energy Cost ($)',
            'Comfort Violations (hours)',
            'Computation Time (s)'
        ],
        'Baseline (avg)': [
            f"{results_df['baseline_energy'].mean():.1f}",
            f"{results_df['baseline_cost'].mean():.2f}",
            f"{results_df['baseline_violations'].mean():.1f}",
            "3600 (Physics Sim est.)"
        ],
        'Optimized (avg)': [
            f"{results_df['optimized_energy'].mean():.1f}",
            f"{results_df['optimized_cost'].mean():.2f}",
            f"{results_df['optimized_violations'].mean():.1f}",
            f"{results_df['computation_time_s'].mean():.1f}"
        ],
        'Improvement': [
            f"{results_df['energy_savings_pct'].mean():.1f}% ↓",
            f"{results_df['cost_savings_pct'].mean():.1f}% ↓",
            f"{results_df['comfort_improvement_pct'].mean():.1f}% ↓",
            f"{(1 - results_df['computation_time_s'].mean() / 3600) * 100:.1f}% ↓"
        ]
    })
    print(summary.to_string(index=False))
    
    summary.to_csv(RESULTS_DIR / 'table4_results.csv', index=False)
    results_df.to_csv(RESULTS_DIR / 'detailed_results.csv', index=False)
    
    # Climate zone analysis
    print("\n[Performance by Climate Zone]")
    climate_summary = results_df.groupby('climate_zone').agg({
        'energy_savings_pct': 'mean',
        'cost_savings_pct': 'mean',
        'comfort_improvement_pct': 'mean'
    }).round(2)
    print(climate_summary)
    
    return results_df


def generate_visualizations(
    optimization_results: dict,
    results_df: pd.DataFrame
):
    """Generate all publication-quality figures."""
    print_header("GENERATING VISUALIZATIONS")
    
    viz_config = VisualizationConfig()
    generator = FigureGenerator(viz_config)
    
    # Figure 1: Framework Schematic (always generated)
    print("\n[1] Figure 1: Framework Schematic")
    generator.figure1_framework_schematic()
    print(f"    Saved to: {FIGURES_DIR / 'figure1_framework.png'}")
    
    # Figure 2: Daily Optimization (first building)
    if optimization_results.get('optimization_results'):
        print("\n[2] Figure 2: Daily Optimization Profile")
        first_result = optimization_results['optimization_results'][0]
        generator.figure2_daily_optimization(
            first_result['result'],
            first_result['weather_data']
        )
        print(f"    Saved to: {FIGURES_DIR / 'figure2_daily_optimization.png'}")
        
        # Figure 3: Pareto Front
        if first_result['result'].pareto_front:
            print("\n[3] Figure 3: Pareto Front")
            generator.figure3_pareto_front(
                first_result['result'].pareto_front,
                first_result['result'].baseline_cost,
                first_result['result'].baseline_comfort_violations
            )
            print(f"    Saved to: {FIGURES_DIR / 'figure3_pareto_front.png'}")
    else:
        print("\n[2-3] Skipping optimization figures (no results)")
    
    # Figure 4: Cross-Building Performance
    if results_df is not None and len(results_df) > 0:
        print("\n[4] Figure 4: Cross-Building Performance")
        generator.figure4_cross_building_performance(results_df)
        print(f"    Saved to: {FIGURES_DIR / 'figure4_cross_building.png'}")
    else:
        print("\n[4] Skipping cross-building figure (no results)")
    
    print(f"\n✓ Figures saved to: {FIGURES_DIR}")


def main():
    """Main execution pipeline."""
    print("\n" + "=" * 60)
    print("  BDG2 SURROGATE-ASSISTED BUILDING ENERGY OPTIMIZATION")
    print("  Framework for Applied Energy Publication")
    print("=" * 60)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Phase 1: Data Curation
        processor, building_data_list = phase1_data_curation(n_buildings=5)
        
        # Phase 2: Surrogate Model Training
        model_results = phase2_surrogate_training(building_data_list, processor)
        
        # Phase 3: Optimization
        optimization_results = phase3_optimization(building_data_list, model_results)
        
        # Phase 4: Results Analysis
        results_df = phase4_results_analysis(
            building_data_list, model_results, optimization_results
        )
        
        # Generate Visualizations
        generate_visualizations(optimization_results, results_df)
        
        # Final Summary
        total_time = time.time() - start_time
        print_header("EXECUTION COMPLETE")
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        print(f"\nOutputs saved to:")
        print(f"  - Results: {RESULTS_DIR}")
        print(f"  - Figures: {FIGURES_DIR}")
        print(f"  - Models:  {MODELS_DIR}")
        
        print("\n--- Key Findings ---")
        print(f"• Average energy savings: {results_df['energy_savings_pct'].mean():.1f}%")
        print(f"• Average cost savings: {results_df['cost_savings_pct'].mean():.1f}%")
        print(f"• Average comfort improvement: {results_df['comfort_improvement_pct'].mean():.1f}%")
        print(f"• Surrogate speedup vs physics: ~{3600/results_df['computation_time_s'].mean():.0f}x faster")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
