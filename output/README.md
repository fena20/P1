# Output Folder Contents

This folder contains all generated figures and tables for the research project.

## Folder Structure

```
output/
├── figures/          # Publication-quality figures (300 DPI)
└── tables/           # Formatted tables (CSV, Markdown, LaTeX)
```

## Figures

All figures are generated at **300 DPI** resolution for publication quality and saved in both PNG and PDF formats.

### Figure 1: Framework Schematic
- **Files**: `figure1_framework_schematic.png`, `figure1_framework_schematic.pdf`
- **Size**: 122 KB (PNG), 25 KB (PDF)
- **Content**: Schematic diagram showing:
  - (a) BDG2 Data Sources (weather, meters, metadata)
  - (b) Surrogate Model Training (LSTM/XGBoost)
  - (c) GA Optimization Loop

### Figure 2: Daily Optimization Profile
- **Files**: `figure2_daily_optimization_profile.png`, `figure2_daily_optimization_profile.pdf`
- **Size**: 223 KB (PNG), 30 KB (PDF)
- **Content**: Three-panel figure showing:
  - (a) Outdoor temperature profile (24-hour cycle)
  - (b) Baseline vs. optimized setpoint schedules
  - (c) Energy consumption comparison

### Figure 3: Pareto Front
- **Files**: `figure3_pareto_front.png`, `figure3_pareto_front.pdf`
- **Size**: 116 KB (PNG), 23 KB (PDF)
- **Content**: Cost vs. Comfort trade-off showing:
  - Pareto-optimal solutions
  - Cost-optimal solution (red square)
  - Comfort-optimal solution (green triangle)

### Figure 4: Cross-Building Performance
- **Files**: `figure4_cross_building_performance.png`, `figure4_cross_building_performance.pdf`
- **Size**: 119 KB (PNG), 27 KB (PDF)
- **Content**: Two-panel figure showing:
  - (a) Energy savings by building (bar chart)
  - (b) Model performance by climate zone (MAE/RMSE)

## Tables

All tables are provided in three formats: **CSV**, **Markdown**, and **LaTeX**.

### Table 1: Building Characteristics
- **Files**: `table1_building_characteristics.csv`, `.md`, `.tex`
- **Content**: Characteristics of selected case study buildings
- **Columns**: Building ID, Primary Use, Floor Area (m²), Climate Zone, Year Built, Data Resolution
- **Rows**: 5 sample buildings

### Table 2: Input Variables
- **Files**: `table2_input_variables.csv`, `.md`, `.tex`
- **Content**: Input variables for the prediction model
- **Columns**: Variable Category, Feature Name, Unit, Source, Relevance
- **Rows**: 6 input variables (environmental, temporal, control)

### Table 3: Optimization Constraints
- **Files**: `table3_optimization_constraints.csv`, `.md`, `.tex`
- **Content**: Objective function and optimization constraints
- **Columns**: Parameter, Description, Value/Constraint
- **Rows**: 5 parameters (objective function, decision variables, constraints, algorithm, time horizon)

### Table 4: Comparative Results
- **Files**: `table4_comparative_results.csv`, `.md`, `.tex`
- **Content**: Baseline vs. optimized controller performance comparison
- **Columns**: Performance Metric, Baseline Controller, Proposed AI Optimizer, Improvement (%)
- **Rows**: 4 metrics (energy, cost, comfort violations, computational time)

### Table 5: Pareto Solutions Summary
- **Files**: `table5_pareto_solutions_summary.csv`, `.md`, `.tex`
- **Content**: Summary of Pareto-optimal solutions
- **Columns**: Solution ID, Energy Cost ($), Comfort Penalty (hours), Total Cost ($), Avg Setpoint (°C), Setpoint Range (°C)
- **Rows**: 5 representative solutions

## Usage

### For Publications (LaTeX)
Use the `.tex` files directly in your LaTeX document:
```latex
\input{output/tables/table1_building_characteristics.tex}
```

### For Reports (Markdown)
Include the `.md` files in Markdown documents or convert to HTML/PDF.

### For Analysis (CSV)
Load the `.csv` files in Python, R, Excel, or any data analysis tool:
```python
import pandas as pd
df = pd.read_csv('output/tables/table1_building_characteristics.csv')
```

### For Presentations
Use the PNG figures directly or convert PDFs to your preferred format.

## File Sizes Summary

**Figures:**
- Total PNG: ~580 KB
- Total PDF: ~105 KB

**Tables:**
- Total CSV: ~2 KB
- Total Markdown: ~2.5 KB
- Total LaTeX: ~3 KB

## Notes

- All figures meet journal requirements (Applied Energy format, 300 DPI)
- Tables are formatted for publication-ready use
- LaTeX tables can be customized for specific journal requirements
- All data is based on BDG2 dataset analysis
