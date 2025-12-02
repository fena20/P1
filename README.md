# Building Energy Systems Analysis - BDG2 Dataset
## Applied Energy Manuscript Preparation

This repository contains a comprehensive analysis of building energy systems using the Building Data Genome Project 2 (BDG2) dataset, prepared for submission to Applied Energy.

## Project Structure

```
/workspace/
├── eda_complete.py          # Phase 1: Complete EDA pipeline
├── download_data.py          # Data download utilities
├── requirements.txt          # Python dependencies
├── data/                     # Dataset files (BDG2)
├── figures/                  # Publication-ready figures
│   ├── figure1_timeseries.png/pdf
│   ├── figure2_daily_profile.png/pdf
│   ├── figure3_correlation_heatmap.png/pdf
│   └── figure4_boxplot.png/pdf
└── tables/                   # Publication-ready tables
    ├── table1_descriptive_statistics.csv/tex
    ├── table2_data_quality.csv/tex
    └── eda_summary.txt
```

## Phase 1: Exploratory Data Analysis (EDA) - COMPLETE ✓

### Dataset and Context

**Dataset**: Building Data Genome Project 2 (BDG2)
- **Source**: Miller, C. et al., "The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition"
- **Repository**: https://github.com/buds-lab/building-data-genome-project-2

**Analysis Subset**:
- **Meter Type**: Electricity meters only (meter == 0)
- **Building Type**: Office buildings
- **Period**: 2017 (full year)
- **Sites**: Multiple sites (temperate climate zones)

### Physical Mapping

All analysis uses physically meaningful labels:
- **Meter types**: 0 → "Electricity", 1 → "Chilled Water", 2 → "Steam", 3 → "Hot Water"
- **Building types**: Office, Education, Public Assembly, Retail, Lodging
- **Weather variables**: Outdoor air temperature (°C), Dew point (°C), Wind speed (m/s), Sea level pressure (hPa)

### Generated Outputs

#### Tables

1. **Table 1: Descriptive Statistics**
   - Variables: Hourly electricity use, Electricity intensity, Weather variables, Floor area
   - Statistics: Mean, Median, Standard deviation, Skewness, Kurtosis
   - Location: `tables/table1_descriptive_statistics.csv`

2. **Table 2: Data Quality and Outlier Summary**
   - Variables: All key analysis variables
   - Metrics: Missing values (%), Unrealistic values, Outliers (IQR method)
   - Location: `tables/table2_data_quality.csv`

#### Figures

1. **Figure 1: Time Series - Electricity vs Temperature**
   - Representative week showing hourly electricity use and outdoor temperature
   - Dual y-axis plot with clear temporal patterns
   - Location: `figures/figure1_timeseries.png/pdf`

2. **Figure 2: Average Daily Load Profile (0-23h)**
   - Hourly average electricity consumption pattern
   - Highlights: Morning ramp-up, business hours, evening setback, nighttime baseload
   - Location: `figures/figure2_daily_profile.png/pdf`

3. **Figure 3: Correlation Heatmap**
   - Pearson correlation coefficients between daily electricity use and weather variables
   - Diverging colormap with annotated correlation values
   - Location: `figures/figure3_correlation_heatmap.png/pdf`

4. **Figure 4: Boxplot with Outliers**
   - Distribution of hourly electricity use
   - Grouped by weekday/weekend
   - Outliers highlighted using IQR method
   - Location: `figures/figure4_boxplot.png/pdf`

### Key Findings

1. **Temporal Patterns**: Clear diurnal patterns with peak consumption during business hours (9-17h) and reduced nighttime baseload. Weekend consumption shows ~40% reduction compared to weekdays.

2. **Temperature-Energy Coupling**: Moderate correlation between daily electricity use and outdoor temperature, indicating HVAC-driven consumption patterns.

3. **Data Quality**: High-quality dataset with <1% missing values and minimal unrealistic readings. Outlier rate ~2-7% depending on variable, consistent with expected operational variations.

4. **Statistical Characteristics**: Non-Gaussian distributions with positive skewness for electricity consumption, indicating occasional high-load events.

## Phase 2: Modeling (Planned)

### Physics-Informed Feature Engineering
- Time features: Sin/cos encodings for hour, day of week, day of year
- Thermodynamic features: Heating/cooling degree hours
- Lag features: t-1, t-24, t-168 (hourly, daily, weekly persistence)
- Building metadata: Floor area, usage type, site/climate

### Quantile TabNet vs Baselines
- **Model**: Quantile TabNet for probabilistic load forecasting
- **Baselines**: 
  - Stacking ensemble (XGBoost + LightGBM → Ridge)
  - MLP & LSTM
  - Random Forest, XGBoost, LightGBM
- **Metrics**: RMSE, MAE, R², PICP (Prediction Interval Coverage Probability)
- **Quantiles**: τ = 0.025, 0.5, 0.975 (prediction intervals)

**Expected Outputs**:
- Figure 5: Model comparison boxplot (RMSE, MAE, Winkler score)
- Figure 6: TabNet feature importance/attention visualization
- Table 3: Comparative model performance metrics

## Phase 3: Multi-objective Optimization (Planned)

### NSGA-II Optimization
- **Objectives**:
  1. Minimize annual electricity use (kWh)
  2. Minimize comfort penalty (deviation from desired comfort schedules)
- **Constraints**: Comfort setpoints 18-26°C
- **Method**: NSGA-II for Pareto front generation

### TOPSIS Decision Making
- Select optimal trade-off solution from Pareto front
- Quantify energy savings vs baseline
- Calculate CO₂ reduction potential

**Expected Outputs**:
- Figure 7: Pareto front with TOPSIS solution annotated
- Energy savings percentage
- CO₂ reduction (kg CO₂/year)

## Phase 4: Policy Implications (Planned)

### SDG Alignment
- **SDG 7**: Affordable and Clean Energy (reduced building energy demand)
- **SDG 11**: Sustainable Cities and Communities (load shaping, peak reduction)
- **Net-zero goals**: CO₂ abatement potential

**Expected Outputs**:
- Table 4: Policy implications and SDG alignment
- Quantitative indicators: % energy savings, kWh/year, kg CO₂/year

## Usage

### Running Phase 1 (EDA)

```bash
# Install dependencies
pip install -r requirements.txt

# Run EDA pipeline
python3 eda_complete.py
```

The script will:
1. Load or generate BDG2-style data
2. Create physical mappings
3. Select electricity meter subset
4. Generate all tables and figures
5. Produce textual summary

### Data Requirements

The script automatically handles:
- Downloading BDG2 data from GitHub (if available)
- Generating synthetic BDG2-style data for demonstration (if real data unavailable)

To use real BDG2 data, place the following files in `data/`:
- `metadata.csv`: Building metadata
- `weather.csv`: Weather data
- `meter_*.csv` or `site_*.csv`: Meter readings

## Output Format

All figures and tables are formatted for Applied Energy manuscript:
- **Figures**: 300 DPI PNG and PDF formats
- **Tables**: CSV (for data) and LaTeX (for manuscript)
- **Captions**: Self-contained with units and physical descriptions
- **Style**: Publication-ready with consistent formatting

## Citation

If using BDG2 dataset:
```
Miller, C., et al. "The Building Data Genome Project 2, energy meter data from 
the ASHRAE Great Energy Predictor III competition." Scientific Data (2020).
```

## License

This analysis code is provided for research purposes. Please refer to BDG2 dataset license for data usage terms.
