# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset (2024 data removed)
data = pd.read_csv(r'pl-tables-1993-2023.csv')

# Data Preprocessing
# Define the features
features = ['played', 'won', 'lost', 'drawn', 'gf', 'ga', 'gd', 'points']
target = 'position'

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Function to calculate weighted average
def weighted_average(team_data, features, current_year):
    # Assign weights inversely proportional to the difference from the current year
    team_data['year_diff'] = current_year - team_data['season_end_year']
    
    # Calculate weights, giving more weight to recent seasons (e.g., exponentially decay)
    team_data['weight'] = np.exp(-team_data['year_diff'] / 1.05)  # Adjust the denominator for stronger/weaker decay
    
    # Calculate weighted average for each feature
    weighted_avg = {}
    for feature in features:
        weighted_avg[feature] = np.average(team_data[feature], weights=team_data['weight'])
    
    return pd.Series(weighted_avg)

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
            continue

        # Calculate the weighted average of past performance metrics for the team
        trend_features = weighted_average(team_data, features, prediction_year)

        # Reshape for model prediction (as RF expects 2D input)
        X_team = trend_features.values.reshape(1, -1)

        # Predict future standing for the team using the trained Random Forest model
        predicted_position = rf_model.predict(X_team)[0]  # Get single prediction

        # Store the result before rounding
        predictions.append({'Team': team, 'Predicted Position': predicted_position})

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
    comparison_df.to_csv(output_file, index=False)

    # Calculate the R² score (how well the predictions fit the actual standings)
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])
    
    # Output R² score
    print(f"\nR² value: {r2}")


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
