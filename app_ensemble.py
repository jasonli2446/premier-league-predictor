import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load the dataset
data = pd.read_csv('pl-tables-1993-2023.csv')

# Define features and target
features = ['points', 'gf', 'ga', 'gd', 'won', 'drawn', 'lost']
target = 'position'

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Initialize Multiple Models for Ensemble
# Model 1: Random Forest (our current champion)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model 2: Gradient Boosting (learns from mistakes iteratively)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Model 3: Linear Regression (simple but effective baseline)
lr_model = LinearRegression()

# Train all models
print("Training Random Forest...")
rf_model.fit(X_train, y_train)
print("Training Gradient Boosting...")
gb_model.fit(X_train, y_train)
print("Training Linear Regression...")
lr_model.fit(X_train, y_train)
print("All models trained successfully!")

# Test ensemble performance on validation set
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Calculate individual model scores for weight determination
rf_score = r2_score(y_test, rf_pred)
gb_score = r2_score(y_test, gb_pred)
lr_score = r2_score(y_test, lr_pred)

print(f"Individual Model Performance on Validation Set:")
print(f"Random Forest: {rf_score:.3f}")
print(f"Gradient Boosting: {gb_score:.3f}")
print(f"Linear Regression: {lr_score:.3f}")

# Calculate ensemble weights based on performance
total_score = rf_score + gb_score + lr_score
rf_weight = rf_score / total_score
gb_weight = gb_score / total_score
lr_weight = lr_score / total_score

print(f"\nEnsemble Weights:")
print(f"Random Forest: {rf_weight:.3f}")
print(f"Gradient Boosting: {gb_weight:.3f}")
print(f"Linear Regression: {lr_weight:.3f}")

# Function to calculate weighted average with recent form
def weighted_average(team_data, features, current_year):
    # Create a copy to avoid SettingWithCopyWarning
    team_data = team_data.copy()
    
    # Assign weights inversely proportional to the difference from the current year
    team_data['year_diff'] = current_year - team_data['season_end_year']
    
    # Calculate weights, giving more weight to recent seasons
    team_data['weight'] = np.exp(-team_data['year_diff'] / 1.05)
    
    # Calculate weighted average for each feature
    weighted_avg = {}
    for feature in features:
        weighted_avg[feature] = np.average(team_data[feature], weights=team_data['weight'])
    
    # Add recent form score as a new feature
    recent_form = calculate_recent_form_score(team_data, current_year)
    weighted_avg['recent_form_score'] = recent_form
    
    return pd.Series(weighted_avg)

# Function to calculate recent form analysis (tuned version)
def calculate_recent_form_score(team_data, current_year, last_n_seasons=3):
    """
    Calculate performance trend in most recent seasons to capture momentum.
    Returns a form score indicating if team is improving (+) or declining (-).
    """
    # Get data from recent seasons only
    recent_data = team_data[team_data['season_end_year'] >= (current_year - last_n_seasons)]
    
    if len(recent_data) < 2:
        return 0  # Not enough data to calculate trend
    
    # Sort by year to ensure proper trend calculation
    recent_data = recent_data.sort_values('season_end_year')
    
    # Calculate trend in points over recent seasons
    if len(recent_data) >= 2:
        # Normalize years to start from 0 for better polynomial fitting
        years_normalized = recent_data['season_end_year'] - recent_data['season_end_year'].min()
        
        # Calculate both trends with better scaling
        points_trend = np.polyfit(years_normalized, recent_data['points'], 1)[0]
        position_trend = -np.polyfit(years_normalized, recent_data['position'], 1)[0]
        
        # Scale the trends to reasonable ranges
        points_trend_scaled = points_trend / 5
        position_trend_scaled = position_trend / 2
        
        # Combine both trends with balanced weighting
        form_score = (points_trend_scaled * 0.6) + (position_trend_scaled * 0.4)
        
        # Cap the form score to prevent extreme adjustments
        form_score = max(-3, min(3, form_score))
        
        return form_score
    
    return 0

# Function to make ensemble predictions
def ensemble_predict(X_team, rf_model, gb_model, lr_model, rf_weight, gb_weight, lr_weight):
    """
    Make prediction using weighted ensemble of all three models
    """
    # Get predictions from all models
    rf_pred = rf_model.predict(X_team)[0]
    gb_pred = gb_model.predict(X_team)[0]
    lr_pred = lr_model.predict(X_team)[0]
    
    # Calculate weighted ensemble prediction
    ensemble_pred = (rf_pred * rf_weight) + (gb_pred * gb_weight) + (lr_pred * lr_weight)
    
    return ensemble_pred, rf_pred, gb_pred, lr_pred

# Function to Predict for a Specified Year using ensemble approach
def compare_predictions(teams, actual_standings, prediction_year):
    # Filter the data for seasons that occurred before the given prediction year
    historical_data = data[data['season_end_year'] < prediction_year]

    if historical_data.empty:
        raise ValueError(f"No historical data available before {prediction_year}.")

    # Retrain models on historical data only
    X_historical = historical_data[features]
    y_historical = historical_data[target]
    
    rf_model.fit(X_historical, y_historical)
    gb_model.fit(X_historical, y_historical)
    lr_model.fit(X_historical, y_historical)

    predictions = []
    recent_form_details = []
    ensemble_details = []

    for team in teams:
        # Get historical data for the specific team
        team_data = historical_data[historical_data['team'] == team]

        if team_data.empty:
            print(f"Warning: No historical data found for team '{team}'. Skipping prediction.")
            continue

        # Calculate the weighted average of past performance metrics for the team
        trend_features = weighted_average(team_data, features, prediction_year)

        # Use only the original features for prediction
        X_team = trend_features[features].values.reshape(1, -1)

        # Get ensemble prediction and individual model predictions
        ensemble_pred, rf_pred, gb_pred, lr_pred = ensemble_predict(
            X_team, rf_model, gb_model, lr_model, rf_weight, gb_weight, lr_weight
        )
        
        # Apply recent form adjustment to the ensemble prediction
        form_score = trend_features['recent_form_score']
        
        # Conservative adjustment factor (same as tuned version)
        form_adjustment = -form_score * 0.05
        adjusted_prediction = ensemble_pred + form_adjustment
        
        # Ensure prediction stays within reasonable bounds (1-20)
        adjusted_prediction = max(1, min(20, adjusted_prediction))
        
        # Store details for analysis
        if abs(form_score) > 0.5:  # Only show significant form changes
            form_direction = "Improving ↗" if form_score > 0 else "Declining ↘" if form_score < -0.5 else "Stable →"
            recent_form_details.append({
                'team': team,
                'form_score': form_score,
                'direction': form_direction
            })
        
        # Store ensemble details for interesting cases
        model_variance = np.std([rf_pred, gb_pred, lr_pred])
        if model_variance > 2:  # Show cases where models disagree significantly
            ensemble_details.append({
                'team': team,
                'rf_pred': rf_pred,
                'gb_pred': gb_pred,
                'lr_pred': lr_pred,
                'ensemble': ensemble_pred,
                'variance': model_variance
            })
        
        # Store the adjusted result
        predictions.append({'Team': team, 'Predicted Position': adjusted_prediction})

    # Sort predictions by predicted positions
    predictions.sort(key=lambda x: x['Predicted Position'])

    # Assign unique integer positions based on the sorted predictions
    for rank, prediction in enumerate(predictions, start=1):
        prediction['Predicted Position'] = rank

    # Convert predictions to DataFrame
    predicted_df = pd.DataFrame(predictions)
    
    # Merge predicted positions with actual positions
    actual_df = pd.DataFrame(actual_standings, columns=['Team', 'Actual Position'])
    comparison_df = pd.merge(predicted_df, actual_df, on='Team')

    # Calculate the difference between predicted and actual positions
    comparison_df['Position Difference'] = comparison_df['Predicted Position'] - comparison_df['Actual Position']

    # Save the comparison results to a CSV file
    output_file = f'{prediction_year}_comparison_predictions.csv'
    comparison_df.to_csv(output_file, index=False)
    
    # Calculate the R² score
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])
    
    print("\n" + "=" * 60)
    print("PREMIER LEAGUE 2024 PREDICTION RESULTS (ENSEMBLE + RECENT FORM)")
    print("=" * 60)
    print(f"Model Accuracy (R² Score): {r2:.3f}")
    print("Results saved to: 2024_comparison_predictions.csv")
    
    # Show ensemble analysis for disagreeing models
    if ensemble_details:
        print(f"\nENSEMBLE MODEL ANALYSIS:")
        print("-" * 30)
        print("Teams where models disagreed significantly:")
        for detail in ensemble_details[:5]:  # Show top 5
            print(f"• {detail['team']:<17} RF:{detail['rf_pred']:4.1f} GB:{detail['gb_pred']:4.1f} LR:{detail['lr_pred']:4.1f} → Ens:{detail['ensemble']:4.1f}")
    
    # Show recent form analysis
    if recent_form_details:
        print(f"\nRECENT FORM ANALYSIS:")
        print("-" * 30)
        for detail in recent_form_details:
            print(f"• {detail['team']:<17} Form Score: {detail['form_score']:+.2f} ({detail['direction']})")
    
    # Show top predictions vs actual
    print(f"\nTOP PREDICTIONS vs ACTUAL:")
    print("-" * 45)
    for i, row in comparison_df.head(10).iterrows():
        diff = row['Position Difference']
        if abs(diff) <= 1:
            icon = "✓"
        elif abs(diff) <= 2:
            icon = "~"
        else:
            icon = "✗"
        
        print(f"{icon} {row['Team']:<17} Pred: {row['Predicted Position']:2.0f}  Actual: {row['Actual Position']:2.0f}  (Diff: {diff:+.0f})")
    
    # Show biggest errors
    biggest_errors = comparison_df.reindex(comparison_df['Position Difference'].abs().sort_values(ascending=False).index).head(3)
    print(f"\nBIGGEST PREDICTION ERRORS:")
    print("-" * 30)
    for i, row in biggest_errors.iterrows():
        print(f"• {row['Team']:<17} Off by {abs(row['Position Difference']):.0f} positions (Predicted {row['Predicted Position']:.0f}, Actual {row['Actual Position']:.0f})")
    
    print("=" * 60)
    
    return comparison_df

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

# Call function to compare predictions for 2024 using ensemble approach
compare_predictions(teams_2024, actual_standings_2024, 2024)
