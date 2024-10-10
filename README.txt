Premier League Position Prediction Model
This project predicts the final standings of Premier League teams for the 2024 season based on historical data (1993–2023). The prediction model is built using a Random Forest Regressor from the scikit-learn library. It takes into account various team performance metrics such as matches played, won, lost, drawn, goals for, goals against, goal difference, and points. Recent seasons are weighted more heavily, using an exponential decay to prioritize current trends.

Data compiled by Evan Gower: https://www.kaggle.com/datasets/evangower/english-premier-league-standings
Features
Weighted Average Trends: The model calculates a weighted average of historical team performance, with recent seasons carrying more weight.
Prediction for Unseen Teams: Teams with no historical Premier League data (e.g., newly promoted teams) are automatically predicted to finish at the bottom of the table.
Unique Predictions: Ensures no two teams receive the same predicted rank by rounding and ranking predicted values.
Comparison with Actual Results: The model outputs the difference between predicted and actual positions, as well as the R² score, to evaluate the accuracy of predictions.

Prerequisites
Install required libraries:
pip install -r requirements.txt

Dataset
The dataset pl-tables-1993-2023.csv includes performance metrics like matches played, won, lost, goals for, goal difference, points, and final position.

Usage
Predict and compare team standings:
compare_predictions(teams_2024, actual_standings_2024, 2024)

Output
Generates a 2024_comparison_predictions.csv file with predicted vs actual standings and calculates the R² score for accuracy.
