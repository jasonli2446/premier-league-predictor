# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Model Selection & Training

## Baseline Model: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

## Advanced Model: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Function to Predict for a Specified Year
def compare_predictions(teams, actual_standings):
    # Filter the data for seasons that occurred before 2024
    historical_data = data[data['season_end_year'] < 2024]

    if historical_data.empty:
        raise ValueError("No historical data available before 2024.")

    # Train the model again using only the data from previous years
    X_historical = historical_data[features]
    y_historical = historical_data[target]

    # Re-train the model using all previous data
    rf_model.fit(X_historical, y_historical)

    # Create an empty list to store predicted positions
    predictions = []

    # Loop through each team for prediction
    for team in teams:
        # Get past data for the specified team before 2024
        team_data = historical_data[historical_data['team'] == team]

        if team_data.empty:
            print(f"No historical data available for {team}, skipping...")
            continue

        # Calculate average or trend of past performance metrics for the team
        trend_features = team_data[features].mean()  # Using average as a proxy for trends

        # Reshape for model prediction (as RF expects 2D input)
        X_team = trend_features.values.reshape(1, -1)

        # Predict future standing for the team using the trained Random Forest model
        predicted_position = rf_model.predict(X_team)[0]  # Get single prediction

        # Store the result
        predictions.append({'Team': team, 'Predicted Position': predicted_position})

    # Convert predictions to DataFrame
    predicted_df = pd.DataFrame(predictions)
    
    # Merge predicted positions with actual positions
    actual_df = pd.DataFrame(actual_standings, columns=['Team', 'Actual Position'])
    comparison_df = pd.merge(predicted_df, actual_df, on='Team')

    # Calculate the difference between predicted and actual positions
    comparison_df['Position Difference'] = comparison_df['Predicted Position'] - comparison_df['Actual Position']

    # Save the comparison results to a CSV file
    output_file = '2024_comparison_predictions.csv'
    comparison_df.to_csv(output_file, index=False)

    # Calculate the R² score (how well the predictions fit the actual standings)
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])
    
    # Output R² score
    print(f"\nR² value: {r2}")


# Actual 2024 standings
actual_standings_2024 = [
    {'Team': 'Manchester City', 'Actual Position': 1},
    {'Team': 'Arsenal', 'Actual Position': 2},
    {'Team': 'Liverpool', 'Actual Position': 3},
    {'Team': 'Aston Villa', 'Actual Position': 4},
    {'Team': 'Tottenham Hotspur', 'Actual Position': 5},
    {'Team': 'Chelsea', 'Actual Position': 6},
    {'Team': 'Newcastle United', 'Actual Position': 7},
    {'Team': 'Manchester United', 'Actual Position': 8},
    {'Team': 'West Ham United', 'Actual Position': 9},
    {'Team': 'Crystal Palace', 'Actual Position': 10},
    {'Team': 'Brighton & Hove Albion', 'Actual Position': 11},
    {'Team': 'AFC Bournemouth', 'Actual Position': 12},
    {'Team': 'Fulham', 'Actual Position': 13},
    {'Team': 'Wolverhampton Wanderers', 'Actual Position': 14},
    {'Team': 'Everton', 'Actual Position': 15},
    {'Team': 'Brentford', 'Actual Position': 16},
    {'Team': 'Nottingham Forest', 'Actual Position': 17},
    {'Team': 'Luton Town', 'Actual Position': 18},
    {'Team': 'Burnley', 'Actual Position': 19},
    {'Team': 'Sheffield United', 'Actual Position': 20},
]

teams_2024 = ['Manchester Utd', 'Chelsea', 'Arsenal', 'Liverpool', 'Aston Villa', 'Everton', 'Newcastle', 'Sheffield Utd', 'Luton Town', 'Crystal Palace', 'Nottingham Forest', 'West Ham', 'Burnley', 'Tottenham', 'Manchester City', 'Brentford', 'Brighton', 'Fulham', 'Bournemouth', 'Wolves']
compare_predictions(teams_2024, actual_standings_2024)
