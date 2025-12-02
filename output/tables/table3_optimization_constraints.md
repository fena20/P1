# Table 3: Objective Function and Optimization Constraints

| Parameter | Description | Value / Constraint |
|---|---|---|
| Objective Function | $J = C_{\text{energy}} + w \cdot D_{\text{comfort}}$ | Trade-off between energy cost and discomfort |
| Decision Variable | HVAC Setpoint $T_{\text{set}}$ | $19^\circ\text{C} \le T_{\text{set}} \le 26^\circ\text{C}$ |
| Comfort Metric | PMV-based comfort band | $-0.5 \le \text{PMV} \le +0.5$ |
| Algorithm | Genetic Algorithm | Population = 50, Generations = 100 |
| Time Horizon | Prediction Window | 24 hours (day-ahead optimization) |
