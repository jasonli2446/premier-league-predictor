# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Suppress pandas and sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load the dataset (2024 data removed)
data = pd.read_csv(r'pl-tables-1993-2023.csv')

# Data Preprocessing
# Define the features (now includes recent form analysis)
features = ['played', 'won', 'lost', 'drawn', 'gf', 'ga', 'gd', 'points']
target = 'position'

# Extended features will include recent_form_score when making predictions
extended_features = features + ['recent_form_score']

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

# Train-Test Split (80% training, 20% testing)
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

# Function to calculate recent form analysis
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
    
    # Calculate trend in points over recent seasons (slope of improvement)
    if len(recent_data) >= 2:
        points_trend = np.polyfit(recent_data['season_end_year'], recent_data['points'], 1)[0]
        
        # Also consider position trend (negative because lower position = better)
        position_trend = -np.polyfit(recent_data['season_end_year'], recent_data['position'], 1)[0]
        
        # Combine both trends (points trend weighted more heavily)
        form_score = (points_trend * 0.7) + (position_trend * 0.3)
        return form_score
    
    return 0

# Function to Predict for a Specified Year using weighted averages
def compare_predictions(teams, actual_standings, prediction_year):
    # Filter the data for seasons that occurred before the given prediction year
    historical_data = data[data['season_end_year'] < prediction_year]

    if historical_data.empty:
        raise ValueError(f"No historical data available before {prediction_year}.")

    # Train the model again using only the data from previous years
    X_historical = historical_data[features]
    y_historical = historical_data[target]

    # Re-train the model using all previous data
    rf_model.fit(X_historical, y_historical)

    # Create an empty list to store predicted positions
    predictions = []

    # Loop through each team for prediction
    for team in teams:           

        # Get past data for the specified team before the prediction year
        team_data = historical_data[historical_data['team'] == team]

        if team_data.empty:
            predicted_position = 20  # Automatically predict 20th place
            predictions.append({'Team': team, 'Predicted Position': predicted_position})
            continue        # Calculate the weighted average of past performance metrics for the team
        trend_features = weighted_average(team_data, features, prediction_year)

        # Use only the original features for prediction (exclude recent_form_score for model)
        X_team = trend_features[features].values.reshape(1, -1)

        # Predict future standing for the team using the trained Random Forest model
        predicted_position = rf_model.predict(X_team)[0]  # Get single prediction
          # Apply recent form adjustment to the prediction
        form_score = trend_features['recent_form_score']
        
        # Adjust prediction based on recent form (positive form = better position = lower number)
        # Use a smaller adjustment factor to avoid over-correction
        form_adjustment = -form_score * 0.15  # Reduced from 0.5 to 0.15
        adjusted_prediction = predicted_position + form_adjustment
        
        # Ensure prediction stays within reasonable bounds (1-20)
        adjusted_prediction = max(1, min(20, adjusted_prediction))
        
        # Store the adjusted result
        predictions.append({'Team': team, 'Predicted Position': adjusted_prediction})

    # Sort predictions by predicted positions (before rounding)
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
    comparison_df.to_csv(output_file, index=False)    # Calculate the R² score (how well the predictions fit the actual standings)
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])
      # Display results
    print("=" * 60)
    print(f"PREMIER LEAGUE {prediction_year} PREDICTION RESULTS (WITH RECENT FORM)")
    print("=" * 60)
    print(f"\nModel Accuracy (R² Score): {r2:.3f}")
    print(f"Results saved to: {output_file}")
    
    # Show recent form analysis for some teams
    print("\nRECENT FORM ANALYSIS:")
    print("-" * 30)
    sample_teams = ['Aston Villa', 'Manchester Utd', 'Brentford', 'Arsenal']
    for team in sample_teams:
        if team in teams:
            team_data = historical_data[historical_data['team'] == team]
            if not team_data.empty:
                form_score = calculate_recent_form_score(team_data, prediction_year)
                trend = "Improving ↗" if form_score > 1 else "Declining ↘" if form_score < -1 else "Stable →"
                print(f"• {team:<18} Form Score: {form_score:+.2f} ({trend})")
    
    print("\nTOP PREDICTIONS vs ACTUAL:")
    print("-" * 45)
    
    # Show top 10 teams comparison
    top_10 = comparison_df.head(10)
    for _, row in top_10.iterrows():
        diff = row['Position Difference']
        status = "✓" if abs(diff) <= 1 else "✗" if abs(diff) >= 3 else "~"
        print(f"{status} {row['Team']:<18} Pred: {row['Predicted Position']:2d}  Actual: {row['Actual Position']:2d}  (Diff: {diff:+d})")
    
    # Show biggest prediction errors
    print("\nBIGGEST PREDICTION ERRORS:")
    print("-" * 30)
    biggest_errors = comparison_df.reindex(comparison_df['Position Difference'].abs().sort_values(ascending=False).index).head(3)
    for _, row in biggest_errors.iterrows():
        diff = row['Position Difference']
        print(f"• {row['Team']:<18} Off by {abs(diff)} positions (Predicted {row['Predicted Position']}, Actual {row['Actual Position']})")
    
    print(f"\n{'='*60}")
    return comparison_df


# Actual 2024 standings (Example)
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
