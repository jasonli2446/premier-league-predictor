# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load the dataset (2024 data removed)
data = pd.read_csv('pl-tables-1993-2023.csv')

# Step 2: Data Preprocessing
# Define the features (e.g., Wins, Losses, GF, GA, GD) and target (Position)
features = ['Games played', 'Wins', 'Losses', 'GF', 'GA', 'GD']
target = 'Position'

X = data[features]  # Feature columns
y = data[target]    # Target column (Position)

# Step 3: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection & Training

## Baseline Model: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 5: Model Evaluation for Linear Regression
y_pred_linear = linear_model.predict(X_test)

# Evaluate Linear Regression
print("Linear Regression:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

## Advanced Model: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Model Evaluation for Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Regressor:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# Step 7: Function to Predict for a Specified Year
def predict_for_year(year):
    if year > 2024 or year < 1993:
        raise ValueError("Year must be between 1993 and 2024.")
    
    # Assuming data for the year is stored in a CSV file with the naming pattern 'premier_league_YEAR.csv'
    data_year = pd.read_csv(f'premier_league_{year}.csv')
    X_year = data_year[features]  # Features for the specified year
    
    # Predict standings using Random Forest
    predicted_positions = rf_model.predict(X_year)
    
    # Save the predicted positions to a CSV file
    predicted_df = pd.DataFrame({
        'Team': data_year['Team'],  # Assuming 'Team' column is in your dataset
        'Predicted Position': predicted_positions
    })
    output_file = f'predicted_positions_{year}.csv'
    predicted_df.to_csv(output_file, index=False)
    print(f"\nPredicted standings for {year} saved to '{output_file}'")

# Example: Predict for 2024
predict_for_year(2024)