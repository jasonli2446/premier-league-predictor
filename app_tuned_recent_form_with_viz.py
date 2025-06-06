import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset
data = pd.read_csv('pl-tables-1993-2023.csv')

# Define features and target
features = ['points', 'gf', 'ga', 'gd', 'won', 'drawn', 'lost']
target = 'position'

# Extended features will include recent_form_score when making predictions
extended_features = features + ['recent_form_score']

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

# Cross-validation function for time-series validation
def cross_validate_time_series(data, features, target, prediction_year):
    """
    Perform time-series cross-validation using chronological splits.
    Returns average CV score across multiple time periods and detailed results.
    """
    # Define validation periods (train on earlier data, test on later)
    cv_periods = [
        {'train_end': 2013, 'test_start': 2014, 'test_end': 2016, 'name': '2014-2016'},
        {'train_end': 2016, 'test_start': 2017, 'test_end': 2019, 'name': '2017-2019'},
        {'train_end': 2019, 'test_start': 2020, 'test_end': 2023, 'name': '2020-2023'}
    ]
    
    cv_scores = []
    cv_details = []
    
    for period in cv_periods:
        # Create training and testing sets based on time periods
        train_data = data[data['season_end_year'] <= period['train_end']]
        test_data = data[(data['season_end_year'] >= period['test_start']) & 
                        (data['season_end_year'] <= period['test_end'])]
        
        if len(train_data) == 0 or len(test_data) == 0:
            continue
            
        # Prepare training data
        X_train_cv = train_data[features]
        y_train_cv = train_data[target]
        
        # Train model on this fold
        rf_cv = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_cv.fit(X_train_cv, y_train_cv)
        
        # Get unique teams from test period
        test_teams = test_data['team'].unique()
        
        # Make predictions for each team in test period
        fold_predictions = []
        fold_actuals = []
        
        for team in test_teams:
            team_test_data = test_data[test_data['team'] == team]
            
            for _, row in team_test_data.iterrows():
                # Get historical data for this team (before the test year)
                team_historical = train_data[train_data['team'] == team]
                
                if len(team_historical) == 0:
                    continue
                
                # Calculate weighted average for this team
                trend_features = weighted_average(team_historical, features, row['season_end_year'])
                X_team = trend_features[features].values.reshape(1, -1)
                
                # Make prediction
                predicted_pos = rf_cv.predict(X_team)[0]
                
                # Apply recent form adjustment
                form_score = trend_features['recent_form_score']
                form_adjustment = -form_score * 0.05
                adjusted_prediction = predicted_pos + form_adjustment
                adjusted_prediction = max(1, min(20, adjusted_prediction))
                
                fold_predictions.append(adjusted_prediction)
                fold_actuals.append(row[target])
        
        if len(fold_predictions) > 0:
            fold_score = r2_score(fold_actuals, fold_predictions)
            cv_scores.append(fold_score)
            cv_details.append({
                'period': period['name'],
                'score': fold_score,
                'predictions': len(fold_predictions)
            })
    
    return np.mean(cv_scores) if cv_scores else 0.0, cv_details

# Train final model on all historical data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Function to calculate weighted average with recent form
def weighted_average(team_data, features, current_year):
    # Create a copy to avoid SettingWithCopyWarning
    team_data = team_data.copy()
    
    # Assign weights inversely proportional to the difference from the current year
    team_data['year_diff'] = current_year - team_data['season_end_year']
    
    # Calculate weights, giving more weight to recent seasons (e.g., exponentially decay)
    team_data['weight'] = np.exp(-team_data['year_diff'] / 1.05)  # Adjust the denominator for stronger/weaker decay
    
    # Calculate weighted average for each feature
    weighted_avg = {}
    for feature in features:
        weighted_avg[feature] = np.average(team_data[feature], weights=team_data['weight'])
    
    # Add recent form score as a new feature
    recent_form = calculate_recent_form_score(team_data, current_year)
    weighted_avg['recent_form_score'] = recent_form
    
    return pd.Series(weighted_avg)

# Function to calculate recent form analysis (improved)
def calculate_recent_form_score(team_data, current_year, last_n_seasons=3):
    """
    Calculate performance trend in most recent seasons to capture momentum.
    Returns a form score indicating if team is improving (+) or declining (-).
    Improved version with better scaling and normalization.
    Handles polyfit errors gracefully for bootstrap robustness.
    """
    # Get data from recent seasons only
    recent_data = team_data[team_data['season_end_year'] >= (current_year - last_n_seasons)]
    if len(recent_data) < 2:
        return 0  # Not enough data to calculate trend
    
    # Sort by year to ensure proper trend calculation
    recent_data = recent_data.sort_values('season_end_year')
    
    # Calculate trend in points over recent seasons (slope of improvement)
    if len(recent_data) >= 2:
        years_normalized = recent_data['season_end_year'] - recent_data['season_end_year'].min()
        try:
            # If all years or points are constant, polyfit will fail; catch and return 0
            if np.all(years_normalized == 0) or np.all(recent_data['points'] == recent_data['points'].iloc[0]) or np.all(recent_data['position'] == recent_data['position'].iloc[0]):
                return 0
            points_trend = np.polyfit(years_normalized, recent_data['points'], 1)[0]
            position_trend = -np.polyfit(years_normalized, recent_data['position'], 1)[0]  # Negative because lower position = better
        except Exception:
            return 0
        
        # Scale the trends to reasonable ranges
        # Points typically range 0-100, positions 1-20
        points_trend_scaled = points_trend / 5  # Scale down points trend
        position_trend_scaled = position_trend / 2  # Scale down position trend
        
        # Combine both trends with balanced weighting
        form_score = (points_trend_scaled * 0.6) + (position_trend_scaled * 0.4)
        
        # Cap the form score to prevent extreme adjustments
        form_score = max(-3, min(3, form_score))
        
        return form_score
    
    return 0

# Enhanced visualization functions
def create_visualizations(comparison_df, cv_details, recent_form_details, prediction_year):
    """
    Create comprehensive visualizations for recent form model analysis
    """
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Prediction Accuracy by Team (sorted by error)
    plt.subplot(2, 3, 1)
    comparison_sorted = comparison_df.reindex(comparison_df['Position Difference'].abs().sort_values(ascending=True).index)
    colors = ['green' if abs(diff) <= 1 else 'orange' if abs(diff) <= 2 else 'red' 
              for diff in comparison_sorted['Position Difference']]
    
    bars = plt.barh(range(len(comparison_sorted)), comparison_sorted['Position Difference'].abs(), color=colors)
    plt.yticks(range(len(comparison_sorted)), comparison_sorted['Team'], fontsize=8)
    plt.xlabel('Absolute Position Error')
    plt.title(f'Prediction Accuracy by Team ({prediction_year})')
    plt.grid(axis='x', alpha=0.3)
    
    # 2. Recent Form Impact Analysis
    plt.subplot(2, 3, 2)
    if recent_form_details:
        teams = [detail['team'] for detail in recent_form_details]
        form_scores = [detail['form_score'] for detail in recent_form_details]
        colors = ['green' if score > 0 else 'red' for score in form_scores]
        
        bars = plt.barh(range(len(teams)), form_scores, color=colors, alpha=0.7)
        plt.yticks(range(len(teams)), teams, fontsize=8)
        plt.xlabel('Form Score (+ Improving, - Declining)')
        plt.title('Recent Form Analysis (Significant Changes)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)
    
    # 3. Cross-Validation Performance Over Time
    plt.subplot(2, 3, 3)
    if cv_details:
        periods = [detail['period'] for detail in cv_details]
        scores = [detail['score'] for detail in cv_details]
        
        bars = plt.bar(periods, scores, color='lightcoral', alpha=0.8)
        plt.ylim(0, 1)
        plt.ylabel('R² Score')
        plt.title('Cross-Validation Performance by Period')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    
    # 4. Prediction Error Distribution
    plt.subplot(2, 3, 4)
    errors = comparison_df['Position Difference']
    plt.hist(errors, bins=range(-10, 11), alpha=0.7, color='lightblue', edgecolor='black')
    plt.xlabel('Position Difference (Predicted - Actual)')
    plt.ylabel('Number of Teams')
    plt.title('Distribution of Prediction Errors')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Prediction')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 5. Actual vs Predicted Scatter Plot
    plt.subplot(2, 3, 5)
    plt.scatter(comparison_df['Actual Position'], comparison_df['Predicted Position'], 
               alpha=0.7, s=50, color='purple')
    
    # Add perfect prediction line
    min_pos, max_pos = 1, 20
    plt.plot([min_pos, max_pos], [min_pos, max_pos], 'r--', alpha=0.7, label='Perfect Prediction')
      # Add team labels for worst predictions
    worst_errors = comparison_df.loc[comparison_df['Position Difference'].abs().nlargest(3).index]
    for _, row in worst_errors.iterrows():
        plt.annotate(row['Team'], (row['Actual Position'], row['Predicted Position']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.xlabel('Actual Position')
    plt.ylabel('Predicted Position')
    plt.title('Actual vs Predicted Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Form Score vs Prediction Error Analysis
    plt.subplot(2, 3, 6)
    if recent_form_details:
        # Merge form details with comparison data
        form_df = pd.DataFrame(recent_form_details)
        merged = comparison_df.merge(form_df, left_on='Team', right_on='team', how='inner')
        
        if not merged.empty:
            plt.scatter(merged['form_score'], merged['Position Difference'], 
                       alpha=0.7, s=50, color='orange')
            
            # Add trend line
            if len(merged) > 1:
                z = np.polyfit(merged['form_score'], merged['Position Difference'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(merged['form_score'].min(), merged['form_score'].max(), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)
            
            plt.xlabel('Recent Form Score')
            plt.ylabel('Position Difference (Predicted - Actual)')
            plt.title('Form Score vs Prediction Error')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{prediction_year}_recent_form_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved as: {prediction_year}_recent_form_analysis.png")
    plt.show()

def create_recent_form_trends_plot(data, teams_2024, prediction_year):
    """
    Create a detailed plot showing recent form trends for selected teams
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select 4 interesting teams (mix of improving/declining)
    interesting_teams = ['Arsenal', 'Chelsea', 'Newcastle Utd', 'Brighton']
    
    for i, team in enumerate(interesting_teams):
        if i >= 4:
            break
            
        team_data = data[data['team'] == team]
        team_data = team_data.sort_values('season_end_year')
        
        # Get last 5 years of data
        recent_data = team_data.tail(5)
        
        if len(recent_data) >= 2:
            ax = axes[i]
            
            # Plot points and position trends
            ax2 = ax.twinx()
            
            line1 = ax.plot(recent_data['season_end_year'], recent_data['points'], 
                           'o-', color='blue', linewidth=2, markersize=6, label='Points')
            line2 = ax2.plot(recent_data['season_end_year'], recent_data['position'], 
                            's-', color='red', linewidth=2, markersize=6, label='Position')
            
            # Calculate and show trend
            years_norm = recent_data['season_end_year'] - recent_data['season_end_year'].min()
            if len(recent_data) >= 2:
                points_trend = np.polyfit(years_norm, recent_data['points'], 1)[0]
                pos_trend = np.polyfit(years_norm, recent_data['position'], 1)[0]
                
                # Add trend lines
                trend_years = np.linspace(recent_data['season_end_year'].min(), 
                                        recent_data['season_end_year'].max(), 100)
                trend_years_norm = trend_years - recent_data['season_end_year'].min()
                
                points_trend_line = recent_data['points'].iloc[0] + points_trend * trend_years_norm
                pos_trend_line = recent_data['position'].iloc[0] + pos_trend * trend_years_norm
                
                ax.plot(trend_years, points_trend_line, '--', color='blue', alpha=0.5)
                ax2.plot(trend_years, pos_trend_line, '--', color='red', alpha=0.5)
                
                # Calculate form score for this team
                form_score = calculate_recent_form_score(recent_data, prediction_year)
                ax.text(0.05, 0.95, f'Form Score: {form_score:+.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Season End Year')
            ax.set_ylabel('Points', color='blue')
            ax2.set_ylabel('League Position', color='red')
            ax.set_title(f'{team} - Recent Form Trend')
            ax2.invert_yaxis()  # Lower position numbers are better
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{prediction_year}_form_trends_recent_form.png', dpi=300, bbox_inches='tight')
    print(f"Form trends plot saved as: {prediction_year}_form_trends_recent_form.png")
    plt.show()

# Function to Predict for a Specified Year using weighted averages with enhanced visualization
def compare_predictions(teams, actual_standings, prediction_year):
    # Filter the data for seasons that occurred before the given prediction year
    historical_data = data[data['season_end_year'] < prediction_year]

    if historical_data.empty:
        raise ValueError(f"No historical data available before {prediction_year}.")

    predictions = []
    recent_form_details = []
    confidence_intervals = []

    print(f"Calculating confidence intervals using bootstrap sampling...")
    for i, team in enumerate(teams, 1):
        print(f"Processing {team} ({i}/{len(teams)})...", end=' ')
        team_data = historical_data[historical_data['team'] == team]
        if team_data.empty:
            print(f"Warning: No historical data found for team '{team}'. Skipping prediction.")
            continue
        # Calculate confidence intervals using bootstrap
        ci_results = bootstrap_confidence_interval_recent_form(
            historical_data, team_data, features, prediction_year, n_bootstrap=300, confidence_level=0.95
        )
        trend_features = weighted_average(team_data, features, prediction_year)
        X_team = trend_features[features].values.reshape(1, -1)
        predicted_position = rf_model.predict(X_team)[0]
        form_score = trend_features['recent_form_score']
        form_adjustment = -form_score * 0.05
        adjusted_prediction = predicted_position + form_adjustment
        adjusted_prediction = max(1, min(20, adjusted_prediction))
        confidence_intervals.append({
            'team': team,
            'prediction': ci_results['prediction'],
            'lower_bound': ci_results['lower_bound'],
            'upper_bound': ci_results['upper_bound'],
            'standard_error': ci_results['standard_error'],
            'interval_width': ci_results['interval_width']
        })
        if abs(form_score) > 0.5:
            form_direction = "Improving ↗" if form_score > 0 else "Declining ↘" if form_score < -0.5 else "Stable →"
            recent_form_details.append({
                'team': team,
                'form_score': form_score,
                'direction': form_direction
            })
        predictions.append({'Team': team, 'Predicted Position': adjusted_prediction})
        print("✓")
    predictions.sort(key=lambda x: x['Predicted Position'])
    for rank, prediction in enumerate(predictions, start=1):
        prediction['Predicted Position'] = rank
    predicted_df = pd.DataFrame(predictions)
    actual_df = pd.DataFrame(actual_standings, columns=['Team', 'Actual Position'])
    comparison_df = pd.merge(predicted_df, actual_df, on='Team')
    comparison_df['Position Difference'] = comparison_df['Predicted Position'] - comparison_df['Actual Position']
    ci_df = pd.DataFrame(confidence_intervals)
    comparison_df = pd.merge(comparison_df, ci_df, left_on='Team', right_on='team')
    output_file = f'{prediction_year}_comparison_predictions_with_confidence.csv'
    comparison_df.to_csv(output_file, index=False)
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])
    cv_score, cv_details = cross_validate_time_series(data, features, target, prediction_year)
    print("=" * 70)
    print("PREMIER LEAGUE 2024 PREDICTION RESULTS (RECENT FORM + CONFIDENCE INTERVALS)")
    print("=" * 70)
    print(f"Model Accuracy (R² Score): {r2:.3f}")
    print(f"Cross-Validation Score: {cv_score:.3f}")
    print(f"Confidence Level: 95%")
    print(f"Results saved to: {output_file}")
    if recent_form_details:
        print("\nRECENT FORM ANALYSIS:")
        print("-" * 30)
        for detail in recent_form_details:
            print(f"• {detail['team']:<17} Form Score: {detail['form_score']:+.2f} ({detail['direction']})")
    print(f"\nPREDICTIONS WITH 95% CONFIDENCE INTERVALS:")
    print("-" * 65)
    print("Team                 Pred  Actual  Diff   95% CI Range    Width")
    print("-" * 65)
    for i, row in comparison_df.head(20).iterrows():
        diff = row['Position Difference']
        if abs(diff) <= 1:
            icon = "✓"
        elif abs(diff) <= 2:
            icon = "~"
        else:
            icon = "✗"
        ci_range = f"[{row['lower_bound']:4.1f}, {row['upper_bound']:4.1f}]"
        width = row['interval_width']
        print(f"{icon} {row['Team']:<17} {row['Predicted Position']:4.0f}  {row['Actual Position']:6.0f}  {diff:+4.0f}   {ci_range:12}  {width:5.1f}")
    print(f"\nCONFIDENCE INTERVAL ANALYSIS:")
    print("-" * 40)
    narrowest_ci = comparison_df.nsmallest(3, 'interval_width')
    print("Most confident predictions (narrow intervals):")
    for i, row in narrowest_ci.iterrows():
        print(f"• {row['Team']:<17} Width: {row['interval_width']:4.1f} positions")
    widest_ci = comparison_df.nlargest(3, 'interval_width')
    print("\nLeast confident predictions (wide intervals):")
    for i, row in widest_ci.iterrows():
        print(f"• {row['Team']:<17} Width: {row['interval_width']:4.1f} positions")
    biggest_errors = comparison_df.reindex(comparison_df['Position Difference'].abs().sort_values(ascending=False).index).head(3)
    print(f"\nBIGGEST PREDICTION ERRORS:")
    print("-" * 30)
    for i, row in biggest_errors.iterrows():
        ci_contains_actual = row['lower_bound'] <= row['Actual Position'] <= row['upper_bound']
        contains_text = "✓ within CI" if ci_contains_actual else "✗ outside CI"
        print(f"• {row['Team']:<17} Off by {abs(row['Position Difference']):.0f} positions ({contains_text})")
    within_ci = sum(1 for _, row in comparison_df.iterrows() 
                   if row['lower_bound'] <= row['Actual Position'] <= row['upper_bound'])
    ci_coverage = within_ci / len(comparison_df) * 100
    print(f"\nCONFIDENCE INTERVAL COVERAGE:")
    print(f"• {within_ci}/{len(comparison_df)} actual positions within 95% CI ({ci_coverage:.1f}%)")
    print(f"• Average CI width: {comparison_df['interval_width'].mean():.1f} positions")
    print("=" * 70)
    print("\nGenerating visualizations...")
    create_visualizations(comparison_df, cv_details, recent_form_details, prediction_year)
    create_recent_form_trends_plot(data, teams, prediction_year)
    print("All visualizations completed!")
    return comparison_df

# Bootstrap confidence interval for recent form model

def bootstrap_confidence_interval_recent_form(historical_data, team_data, features, prediction_year, n_bootstrap=500, confidence_level=0.95):
    """
    Calculate confidence intervals for the recent form model using bootstrap sampling.
    """
    bootstrap_predictions = []
    for i in range(n_bootstrap):
        # Bootstrap sample from historical data
        sample = historical_data.sample(n=len(historical_data), replace=True, random_state=i)
        # Train model on bootstrap sample
        X_boot = sample[features]
        y_boot = sample[target]
        rf_boot = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_boot.fit(X_boot, y_boot)
        # Get team sample for trend calculation
        team_bootstrap = sample[sample['team'] == team_data['team'].iloc[0]]
        if len(team_bootstrap) > 0:
            trend_features = weighted_average(team_bootstrap, features, prediction_year)
            X_team = trend_features[features].values.reshape(1, -1)
            pred = rf_boot.predict(X_team)[0]
            form_score = trend_features['recent_form_score']
            form_adjustment = -form_score * 0.05
            adjusted_prediction = pred + form_adjustment
            adjusted_prediction = max(1, min(20, adjusted_prediction))
            bootstrap_predictions.append(adjusted_prediction)
    if not bootstrap_predictions:
        return {'prediction': np.nan, 'lower_bound': np.nan, 'upper_bound': np.nan, 'standard_error': np.nan, 'confidence_level': confidence_level}
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    prediction = np.mean(bootstrap_predictions)
    lower_bound = np.percentile(bootstrap_predictions, lower_percentile)
    upper_bound = np.percentile(bootstrap_predictions, upper_percentile)
    standard_error = np.std(bootstrap_predictions)
    return {
        'prediction': prediction,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'standard_error': standard_error,
        'confidence_level': confidence_level,
        'interval_width': upper_bound - lower_bound
    }

# Hardcoded actual 2024 standings for comparison
actual_standings_2024 = [
    {'Team': 'Manchester City', 'Actual Position': 1},
    {'Team': 'Arsenal', 'Actual Position': 2},
    {'Team': 'Liverpool', 'Actual Position': 3},
    {'Team': 'Aston Villa', 'Actual Position': 4},
    {'Team': 'Tottenham', 'Actual Position': 5},
    {'Team': 'Chelsea', 'Actual Position': 6},
    {'Team': 'Newcastle Utd', 'Actual Position': 7},
    {'Team': 'Manchester Utd', 'Actual Position': 8},
    {'Team': 'West Ham', 'Actual Position': 9},
    {'Team': 'Crystal Palace', 'Actual Position': 10},
    {'Team': 'Brighton', 'Actual Position': 11},
    {'Team': 'Bournemouth', 'Actual Position': 12},
    {'Team': 'Fulham', 'Actual Position': 13},
    {'Team': 'Wolves', 'Actual Position': 14},
    {'Team': 'Everton', 'Actual Position': 15},
    {'Team': 'Brentford', 'Actual Position': 16},
    {'Team': 'Nottingham Forest', 'Actual Position': 17},
    {'Team': 'Luton Town', 'Actual Position': 18},
    {'Team': 'Burnley', 'Actual Position': 19},
    {'Team': 'Sheffield Utd', 'Actual Position': 20},
]

teams_2024 = ['Manchester City', 'Arsenal', 'Liverpool', 'Aston Villa', 'Tottenham', 'Chelsea', 'Newcastle Utd', 'Manchester Utd', 'West Ham', 'Crystal Palace', 'Brighton', 'Bournemouth', 'Fulham', 'Wolves', 'Everton', 'Brentford', 'Nottingham Forest', 'Luton Town', 'Burnley', 'Sheffield Utd']

# Call function to compare predictions for 2024 using weighted averages with visualizations
compare_predictions(teams_2024, actual_standings_2024, 2024)
