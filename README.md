# Premier League Position Prediction Model

This project predicts the final standings of Premier League teams for the 2024 season using historical data (1993–2023). The system features multiple advanced prediction models with statistical confidence intervals and comprehensive analysis capabilities.

**Data Source:** Compiled by Evan Gower: https://www.kaggle.com/datasets/evangower/english-premier-league-standings

## 🚀 Key Features

### Multiple Prediction Models
- **Ensemble Model**: Combines Random Forest, Gradient Boosting, and Linear Regression with performance-weighted voting
- **Recent Form Model**: Enhanced Random Forest with sophisticated momentum analysis
- **Basic Model**: Single Random Forest baseline implementation

### Advanced Analytics
- **Bootstrap Confidence Intervals**: Statistical uncertainty quantification using 95% confidence intervals
- **Recent Form Analysis**: Captures team momentum using polynomial trend analysis over last 3 seasons
- **Cross-Validation**: Time-series validation to prevent data leakage
- **Comprehensive Visualizations**: Multi-panel analysis charts with error distributions and trend analysis

### Statistical Rigor
- **Weighted Historical Data**: Exponential decay prioritizes recent seasons (λ = 1.05)
- **Error Handling**: Robust polynomial fitting with singular matrix protection
- **Performance Metrics**: R² scores, prediction accuracy, and confidence interval coverage analysis

## 📁 File Structure

### Core Models
- `app.py` - Basic Random Forest implementation
- `app_ensemble.py` - Multi-model ensemble approach
- `app_tuned_recent_form.py` - Optimized recent form model

### Advanced Implementations
- `app_ensemble_with_confidence.py` - Ensemble model with bootstrap confidence intervals
- `app_tuned_recent_form_with_confidence.py` - Recent form model with confidence intervals
- `app_tuned_recent_form_with_viz.py` - Recent form model with visualizations and CI

### Visualization Versions
- `app_ensemble_with_viz.py` - Ensemble model with comprehensive charts
- `app_with_recent_form.py` - Enhanced model with momentum analysis
- `app_with_ensemble.py` - Multi-model comparison implementation

## 🛠️ Prerequisites

Install required libraries:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- warnings

## 📊 Dataset

The dataset `pl-tables-1993-2023.csv` includes comprehensive team performance metrics:
- Points, Goals For/Against, Goal Difference
- Wins, Draws, Losses, Matches Played
- Final league position by season
- Team names and season end years

## 🎯 Usage Examples

### Basic Prediction
```python
python app.py
```

### Ensemble Model with Confidence Intervals
```python
python app_ensemble_with_confidence.py
```

### Recent Form Model with Visualizations
```python
python app_tuned_recent_form_with_viz.py
```

### Function Call Example
```python
compare_predictions_with_confidence(teams_2024, actual_standings_2024, 2024, confidence_level=0.95)
```

## 📈 Output Files

### CSV Results
- `2024_comparison_predictions_with_confidence.csv` - Main results with confidence intervals
- `2024_recent_form_predictions_with_confidence.csv` - Recent form model results
- `2024_comparison_predictions.csv` - Basic prediction results

### Visualizations
- `2024_ensemble_analysis.png` - Comprehensive ensemble model analysis
- `2024_form_trends.png` - Recent form trend analysis
- `2024_recent_form_analysis.png` - Multi-panel recent form visualization

## 🔬 Model Performance

### Ensemble Model (2024 Results)
- **R² Score**: 0.693
- **Cross-Validation Score**: 0.379
- **CI Coverage**: 68.4% (95% confidence intervals)
- **Average CI Width**: 9.6 positions

### Recent Form Model (2024 Results)
- **R² Score**: 0.706
- **Cross-Validation Score**: 0.355
- **CI Coverage**: 63.2% (95% confidence intervals)
- **Average CI Width**: 7.1 positions

## 🧮 Technical Implementation

### Confidence Intervals
- **Bootstrap Sampling**: 100-500 iterations per team
- **Resampling Strategy**: With replacement from historical data
- **CI Calculation**: Percentile method (2.5th, 97.5th percentiles)
- **Performance Optimization**: Reduced sample sizes and model complexity for speed

### Recent Form Scoring
- **Trend Analysis**: Polynomial fitting on points and position trends
- **Lookback Window**: Last 3 seasons
- **Scaling**: Balanced weighting (60% points, 40% position trends)
- **Adjustment Factor**: Conservative 0.05x multiplier

### Ensemble Weighting
- **Dynamic Weights**: Based on validation set performance
- **Model Combination**: Weighted average of RF, GB, and LR predictions
- **Variance Analysis**: Identifies cases where models disagree significantly

## 🎨 Visualization Features

### Comprehensive Charts
- Prediction accuracy by team (color-coded by error magnitude)
- Recent form impact analysis with directional indicators
- Cross-validation performance over time periods
- Error distribution histograms
- Actual vs predicted scatter plots with perfect prediction lines
- Form score vs prediction error correlation analysis

### Interactive Analysis
- Team-specific trend plots showing historical performance
- Confidence interval width analysis
- Model disagreement identification
- Coverage rate statistics

## 🔍 Key Insights

### Model Strengths
- Excellent performance on top-tier teams (Manchester City, Arsenal, Liverpool)
- Effective capture of recent form momentum
- Robust handling of newly promoted teams
- Strong cross-validation stability

### Areas for Improvement
- Mid-table prediction accuracy
- Confidence interval calibration
- Handling of extreme form changes
- Seasonal adjustment factors

This advanced prediction system provides both accurate forecasts and statistical rigor through confidence intervals, making it suitable for both analytical and practical applications in football analytics.
