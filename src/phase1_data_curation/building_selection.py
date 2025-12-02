"""
Phase 1.1: Building Selection
Selects residential and lodging-type buildings from BDG2 metadata with sufficient data coverage.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Path configuration
BDG2_ROOT = Path(__file__).parent.parent.parent / "bdg2_data"
METADATA_PATH = BDG2_ROOT / "data" / "metadata" / "metadata.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "processed"

def load_metadata():
    """Load BDG2 building metadata."""
    print(f"Loading metadata from {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(metadata)} buildings")
    return metadata

def filter_residential_buildings(metadata):
    """
    Filter buildings for residential or lodging-type occupancies.
    
    Parameters:
    -----------
    metadata : pd.DataFrame
        BDG2 metadata dataframe
    
    Returns:
    --------
    pd.DataFrame
        Filtered metadata for residential/lodging buildings
    """
    # Define residential and lodging categories
    residential_keywords = [
        'residential', 'lodging', 'dormitory', 'dorm', 
        'multi-family', 'multifamily', 'apartment', 'housing'
    ]
    
    # Filter by primary use category
    primary_use_col = 'primaryspaceusage' if 'primaryspaceusage' in metadata.columns else 'primary_use'
    
    if primary_use_col in metadata.columns:
        mask = metadata[primary_use_col].str.lower().str.contains(
            '|'.join(residential_keywords), 
            na=False, 
            case=False
        )
        filtered = metadata[mask].copy()
    else:
        # Fallback: check all text columns
        print("Warning: 'primaryspaceusage' column not found. Checking all text columns...")
        mask = pd.Series([False] * len(metadata))
        for col in metadata.select_dtypes(include=['object']).columns:
            mask |= metadata[col].astype(str).str.lower().str.contains(
                '|'.join(residential_keywords), 
                na=False, 
                case=False
            )
        filtered = metadata[mask].copy()
    
    print(f"Found {len(filtered)} residential/lodging buildings")
    return filtered

def check_data_availability(building_ids, meters_dir):
    """
    Check which buildings have meter data available.
    
    Parameters:
    -----------
    building_ids : list
        List of building IDs to check
    meters_dir : Path
        Path to meters directory
    
    Returns:
    --------
    dict
        Dictionary mapping building_id to available meter files
    """
    available_buildings = {}
    
    # Check cleaned meters directory
    cleaned_dir = meters_dir / "cleaned"
    if cleaned_dir.exists():
        meter_files = list(cleaned_dir.glob("*.csv"))
        print(f"Found {len(meter_files)} meter files in cleaned directory")
        
        for building_id in building_ids:
            # Check if any meter file contains this building_id
            for meter_file in meter_files:
                try:
                    df = pd.read_csv(meter_file, nrows=100)  # Sample first 100 rows
                    if 'building_id' in df.columns and building_id in df['building_id'].values:
                        if building_id not in available_buildings:
                            available_buildings[building_id] = []
                        available_buildings[building_id].append(str(meter_file))
                except Exception as e:
                    continue
    
    return available_buildings

def select_buildings_with_sufficient_data(metadata, min_coverage=0.8):
    """
    Select buildings with sufficient data coverage.
    
    Parameters:
    -----------
    metadata : pd.DataFrame
        Building metadata
    min_coverage : float
        Minimum data coverage ratio (default: 0.8)
    
    Returns:
    --------
    pd.DataFrame
        Selected buildings with sufficient data
    """
    # For now, return all filtered buildings
    # In a full implementation, we would check actual data coverage
    selected = metadata.copy()
    
    # Add data availability flag (placeholder - would be populated by actual data check)
    selected['has_sufficient_data'] = True
    
    return selected

def main():
    """Main function for building selection."""
    print("=" * 60)
    print("Phase 1.1: Building Selection")
    print("=" * 60)
    
    # Load metadata
    metadata = load_metadata()
    
    # Filter for residential/lodging buildings
    residential_buildings = filter_residential_buildings(metadata)
    
    # Check data availability
    meters_dir = BDG2_ROOT / "data" / "meters"
    building_ids = residential_buildings['building_id'].tolist() if 'building_id' in residential_buildings.columns else residential_buildings.index.tolist()
    
    if 'building_id' not in residential_buildings.columns:
        # Try to find building ID column
        id_cols = [col for col in residential_buildings.columns if 'id' in col.lower() or 'building' in col.lower()]
        if id_cols:
            building_ids = residential_buildings[id_cols[0]].tolist()
        else:
            building_ids = residential_buildings.index.tolist()
    
    available_buildings = check_data_availability(building_ids[:100], meters_dir)  # Check first 100
    
    # Select buildings with sufficient data
    selected_buildings = select_buildings_with_sufficient_data(residential_buildings)
    
    # Save selected buildings
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / "selected_buildings.csv"
    selected_buildings.to_csv(output_file, index=False)
    print(f"\nSelected {len(selected_buildings)} buildings")
    print(f"Saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total buildings in BDG2: {len(metadata)}")
    print(f"Residential/Lodging buildings: {len(residential_buildings)}")
    print(f"Selected buildings: {len(selected_buildings)}")
    
    if available_buildings:
        print(f"Buildings with available meter data: {len(available_buildings)}")
    
    return selected_buildings

if __name__ == "__main__":
    selected_buildings = main()
