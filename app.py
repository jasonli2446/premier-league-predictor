# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Suppress pandas and sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load the dataset (2024 data removed)
data = pd.read_csv(r'pl-tables-1993-2023.csv')

# Data Preprocessing
# Define the features
features = ['played', 'won', 'lost', 'drawn', 'gf', 'ga', 'gd', 'points']
target = 'position'

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

## Initialize Multiple Models for Ensemble
# Model 1: Random Forest (our current champion)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Model 2: Gradient Boosting (learns from mistakes iteratively)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Model 3: Linear Regression (simple but effective baseline)
lr_model = LinearRegression()

# Function to perform time-series cross-validation
def time_series_cross_validation():
    """
    Perform time-series cross-validation using chronological splits
    Returns average performance scores and optimal ensemble weights
    """
    print("Performing Cross-Season Validation...")
    print("=" * 50)    # Define time periods for cross-validation (using 3-year test windows)
    cv_splits = [
        {'train_end': 2010, 'test_start': 2011, 'test_end': 2013},
        {'train_end': 2013, 'test_start': 2014, 'test_end': 2016},
        {'train_end': 2016, 'test_start': 2017, 'test_end': 2019},
        {'train_end': 2019, 'test_start': 2020, 'test_end': 2023},
    ]
    
    all_rf_scores = []
    all_gb_scores = []
    all_lr_scores = []
    all_ensemble_scores = []
    
    for i, split in enumerate(cv_splits, 1):
        print(f"Split {i}: Train [1993-{split['train_end']}] → Test [{split['test_start']}-{split['test_end']}]")
        
        # Create train/test splits based on years
        train_data = data[data['season_end_year'] <= split['train_end']]
        test_data = data[(data['season_end_year'] >= split['test_start']) & 
                        (data['season_end_year'] <= split['test_end'])]
        
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  ⚠️  Insufficient data for split {i}")
            continue
            
        # Prepare features and targets
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]
        
        # Train all models
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_temp = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        lr_temp = LinearRegression()
        
        rf_temp.fit(X_train, y_train)
        gb_temp.fit(X_train, y_train)
        lr_temp.fit(X_train, y_train)
        
        # Make predictions
        rf_pred = rf_temp.predict(X_test)
        gb_pred = gb_temp.predict(X_test)
        lr_pred = lr_temp.predict(X_test)
        
        # Calculate individual scores
        rf_score = r2_score(y_test, rf_pred)
        gb_score = r2_score(y_test, gb_pred)
        lr_score = r2_score(y_test, lr_pred)
        
        # Calculate ensemble weights for this split
        if rf_score > 0 and gb_score > 0 and lr_score > 0:
            total_score = rf_score + gb_score + lr_score
            rf_weight_temp = rf_score / total_score
            gb_weight_temp = gb_score / total_score
            lr_weight_temp = lr_score / total_score
            
            # Create ensemble prediction
            ensemble_pred = (rf_pred * rf_weight_temp) + (gb_pred * gb_weight_temp) + (lr_pred * lr_weight_temp)
            ensemble_score = r2_score(y_test, ensemble_pred)
        else:
            ensemble_score = max(rf_score, gb_score, lr_score)  # Fallback to best individual model
        
        # Store scores
        all_rf_scores.append(rf_score)
        all_gb_scores.append(gb_score)
        all_lr_scores.append(lr_score)
        all_ensemble_scores.append(ensemble_score)
        
        print(f"  RF: {rf_score:.3f} | GB: {gb_score:.3f} | LR: {lr_score:.3f} | Ensemble: {ensemble_score:.3f}")
    
    # Calculate averages
    avg_rf = np.mean(all_rf_scores)
    avg_gb = np.mean(all_gb_scores)
    avg_lr = np.mean(all_lr_scores)
    avg_ensemble = np.mean(all_ensemble_scores)
    
    print("-" * 50)
    print(f"Average Cross-Validation Scores:")
    print(f"Random Forest: {avg_rf:.3f} (±{np.std(all_rf_scores):.3f})")
    print(f"Gradient Boosting: {avg_gb:.3f} (±{np.std(all_gb_scores):.3f})")
    print(f"Linear Regression: {avg_lr:.3f} (±{np.std(all_lr_scores):.3f})")
    print(f"Ensemble: {avg_ensemble:.3f} (±{np.std(all_ensemble_scores):.3f})")
    
    # Calculate optimal ensemble weights based on cross-validation
    if avg_rf > 0 and avg_gb > 0 and avg_lr > 0:
        total_avg = avg_rf + avg_gb + avg_lr
        optimal_rf_weight = avg_rf / total_avg
        optimal_gb_weight = avg_gb / total_avg
        optimal_lr_weight = avg_lr / total_avg
    else:
        # Fallback to equal weights
        optimal_rf_weight = optimal_gb_weight = optimal_lr_weight = 1/3
    
    print(f"\nOptimal Ensemble Weights (based on CV):")
    print(f"Random Forest: {optimal_rf_weight:.3f}")
    print(f"Gradient Boosting: {optimal_gb_weight:.3f}")
    print(f"Linear Regression: {optimal_lr_weight:.3f}")
    print("=" * 50 + "\n")
    
    return optimal_rf_weight, optimal_gb_weight, optimal_lr_weight, avg_ensemble

# Perform cross-validation and get optimal weights
rf_weight, gb_weight, lr_weight, cv_ensemble_score = time_series_cross_validation()

# Train final models on all available data for predictions
print("Training final ensemble models on all historical data...")
rf_model.fit(X, y)
gb_model.fit(X, y)
lr_model.fit(X, y)
print("All models trained successfully!\n")

# Function to calculate weighted average
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
    
    return pd.Series(weighted_avg)

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

# Function to Predict for a Specified Year using weighted averages
def compare_predictions(teams, actual_standings, prediction_year):
    # Filter the data for seasons that occurred before the given prediction year
    historical_data = data[data['season_end_year'] < prediction_year]

    if historical_data.empty:
        raise ValueError(f"No historical data available before {prediction_year}.")    # Train the model again using only the data from previous years
    X_historical = historical_data[features]
    y_historical = historical_data[target]

    # Re-train all ensemble models using historical data
    rf_model.fit(X_historical, y_historical)
    gb_model.fit(X_historical, y_historical)
    lr_model.fit(X_historical, y_historical)

    # Create an empty list to store predicted positions
    predictions = []
    ensemble_details = []

    # Loop through each team for prediction
    for team in teams:           

        # Get past data for the specified team before the prediction year
        team_data = historical_data[historical_data['team'] == team]

        if team_data.empty:
            predicted_position = 20  # Automatically predict 20th place
            predictions.append({'Team': team, 'Predicted Position': predicted_position})
            continue        # Calculate the weighted average of past performance metrics for the team
        trend_features = weighted_average(team_data, features, prediction_year)

        # Use the original features for prediction
        X_team = trend_features.values.reshape(1, -1)

        # Get ensemble prediction and individual model predictions
        ensemble_pred, rf_pred, gb_pred, lr_pred = ensemble_predict(
            X_team, rf_model, gb_model, lr_model, rf_weight, gb_weight, lr_weight
        )
        
        # Store ensemble details for analysis
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
        
        # Store the ensemble result
        predictions.append({'Team': team, 'Predicted Position': ensemble_pred})

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
    r2 = r2_score(comparison_df['Actual Position'], comparison_df['Predicted Position'])    # Display results
    print("=" * 60)
    print(f"PREMIER LEAGUE {prediction_year} PREDICTION RESULTS (ENSEMBLE + CV)")
    print("=" * 60)
    print(f"Current Model Accuracy (R² Score): {r2:.3f}")
    print(f"Cross-Validation Ensemble Score: {cv_ensemble_score:.3f}")
    print(f"Results saved to: {output_file}")
    
    # Show ensemble analysis for disagreeing models
    if ensemble_details:
        print(f"\nENSEMBLE MODEL ANALYSIS:")
        print("-" * 30)
        print("Teams where models disagreed significantly:")
        for detail in ensemble_details[:5]:  # Show top 5
            print(f"• {detail['team']:<17} RF:{detail['rf_pred']:4.1f} GB:{detail['gb_pred']:4.1f} LR:{detail['lr_pred']:4.1f} → Ens:{detail['ensemble']:4.1f}")
    
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
