import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("merged_data.csv")

# Handle missing values (fill with 0 or interpolate)
data.fillna(0, inplace=True)

# Filter data for a specific Entity (e.g., Afghanistan)
entity = "Afghanistan"  # Change this to the desired country
entity_data = data[data["Entity"] == entity]

# Specify the meat category (e.g., "Poultry", "Beef", "Sheep and goat")
meat_category = "Poultry"  # Change this to the desired meat category

# Feature selection
features = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood"]
target_meat = meat_category

# Prepare features (X) and target (y)
X = entity_data[features]
y_meat = entity_data[target_meat]

# Split the data into training and testing sets
X_train, X_test, y_meat_train, y_meat_test = train_test_split(X, y_meat, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso Regression model
lasso_meat = Lasso(alpha=0.1)
lasso_meat.fit(X_train_scaled, y_meat_train)

# Evaluate the model
y_meat_pred = lasso_meat.predict(X_test_scaled)
print(f"{meat_category} Model MSE for {entity}:", mean_squared_error(y_meat_test, y_meat_pred))

# Predict for the next 10 years
future_years = pd.DataFrame({
    "Year": np.arange(2022, 2032),
    "Beef": [0] * 10,  # Placeholder
    "Sheep and goat": [0] * 10,
    "Pork": [0] * 10,
    "Other meats": [0] * 10,
    "Fish and seafood": [0] * 10,
})

# Scale future data
future_years_scaled = scaler.transform(future_years)

# Predict future meat consumption
future_meat_pred = lasso_meat.predict(future_years_scaled)

# Display predictions
print(f"\nPredictions for the next 10 years for {entity} ({meat_category}):")
print("Year | Meat Consumption")
for year, meat in zip(future_years["Year"], future_meat_pred):
    print(f"{year} | {meat:.2f}")