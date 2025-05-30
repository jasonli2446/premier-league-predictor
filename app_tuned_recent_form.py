import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Extended features will include recent_form_score when making predictions
extended_features = features + ['recent_form_score']

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

# Cross-validation function for time-series validation
def cross_validate_time_series(data, features, target, prediction_year):
    """
    Perform time-series cross-validation using chronological splits.
    Returns average CV score across multiple time periods.
    """
    # Define validation periods (train on earlier data, test on later)
    cv_periods = [
        {'train_end': 2013, 'test_start': 2014, 'test_end': 2016},
        {'train_end': 2016, 'test_start': 2017, 'test_end': 2019},
        {'train_end': 2019, 'test_start': 2020, 'test_end': 2023}
    ]
    
    cv_scores = []
    
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
    
    return np.mean(cv_scores) if cv_scores else 0.0

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
    """
    # Get data from recent seasons only
    recent_data = team_data[team_data['season_end_year'] >= (current_year - last_n_seasons)]
    
    if len(recent_data) < 2:
        return 0  # Not enough data to calculate trend
    
    # Sort by year to ensure proper trend calculation
    recent_data = recent_data.sort_values('season_end_year')
    
    # Calculate trend in points over recent seasons (slope of improvement)
    if len(recent_data) >= 2:
        # Normalize years to start from 0 for better polynomial fitting
        years_normalized = recent_data['season_end_year'] - recent_data['season_end_year'].min()
        
        # Calculate both trends but with better scaling
        points_trend = np.polyfit(years_normalized, recent_data['points'], 1)[0]
        position_trend = -np.polyfit(years_normalized, recent_data['position'], 1)[0]  # Negative because lower position = better
        
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

# Function to Predict for a Specified Year using weighted averages
def compare_predictions(teams, actual_standings, prediction_year):
    # Filter the data for seasons that occurred before the given prediction year
    historical_data = data[data['season_end_year'] < prediction_year]

    if historical_data.empty:
        raise ValueError(f"No historical data available before {prediction_year}.")

    predictions = []
    recent_form_details = []

    for team in teams:
        # Get historical data for the specific team
        team_data = historical_data[historical_data['team'] == team]

        if team_data.empty:
            print(f"Warning: No historical data found for team '{team}'. Skipping prediction.")
            continue

        # Calculate the weighted average of past performance metrics for the team
        trend_features = weighted_average(team_data, features, prediction_year)

        # Use only the original features for prediction (exclude recent_form_score for model)
        X_team = trend_features[features].values.reshape(1, -1)

        # Predict future standing for the team using the trained Random Forest model
        predicted_position = rf_model.predict(X_team)[0]  # Get single prediction
        
        # Apply recent form adjustment to the prediction (TUNED VERSION)
        form_score = trend_features['recent_form_score']
        
        # Much more conservative adjustment factor
        form_adjustment = -form_score * 0.05  # Reduced from 0.15 to 0.05
        adjusted_prediction = predicted_position + form_adjustment
        
        # Ensure prediction stays within reasonable bounds (1-20)
        adjusted_prediction = max(1, min(20, adjusted_prediction))
        
        # Store details for recent form analysis
        if abs(form_score) > 0.5:  # Only show significant form changes
            form_direction = "Improving ↗" if form_score > 0 else "Declining ↘" if form_score < -0.5 else "Stable →"
            recent_form_details.append({
                'team': team,
                'form_score': form_score,
                'direction': form_direction
            })
        
        # Store the adjusted result
        predictions.append({'Team': team, 'Predicted Position': adjusted_prediction})

    # Sort predictions by predicted positions (before rounding)
    predictions.sort(key=lambda x: x['Predicted Position'])

    # Assign unique integer positions based on the sorted predictions
    for rank, prediction in enumerate(predictions, start=1):
        prediction['Predicted Position'] = rank    # Convert predictions to DataFrame
    predicted_df = pd.DataFrame(predictions)
    
    # Merge predicted positions with actual positions
    actual_df = pd.DataFrame(actual_standings, columns=['Team', 'Actual Position'])
    comparison_df = pd.merge(predicted_df, actual_df, on='Team')

    # Calculate the difference between predicted and actual positions
    comparison_df['Position Difference'] = comparison_df['Predicted Position'] - comparison_df['Actual Position']

    # Save the comparison results to a CSV file
    output_file = f'{prediction_year}_comparison_predictions.csv'
    comparison_df.to_csv(output_file, index=False)
      # Calculate the R² score (how well the predictions fit the actual standings)
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])
    
    # Perform cross-validation
    cv_score = cross_validate_time_series(data, features, target, prediction_year)
    
    print("=" * 60)
    print("PREMIER LEAGUE 2024 PREDICTION RESULTS (TUNED RECENT FORM)")
    print("=" * 60)
    print(f"Model Accuracy (R² Score): {r2:.3f}")
    print(f"Cross-Validation Score: {cv_score:.3f}")
    print("Results saved to: 2024_comparison_predictions.csv")
    
    # Show recent form analysis
    if recent_form_details:
        print("\nRECENT FORM ANALYSIS:")
        print("-" * 30)
        for detail in recent_form_details:
            print(f"• {detail['team']:<17} Form Score: {detail['form_score']:+.2f} ({detail['direction']})")
    
    # Show top predictions vs actual
    print("\nTOP PREDICTIONS vs ACTUAL:")
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
    print("\nBIGGEST PREDICTION ERRORS:")
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

# Call function to compare predictions for 2024 using weighted averages
compare_predictions(teams_2024, actual_standings_2024, 2024)
