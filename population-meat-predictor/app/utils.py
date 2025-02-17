import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

# Default Year to start the prediction
year_to_start_prediction = 2022

# Default number of future years to predict
future_years = 7

features_set = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood", "Poultry"]

# Initialize variables
future = future_years
year = year_to_start_prediction
meat_option = ""
plot_models = {}
countries_selected = []
lasso_best_performance = {}
ridge_best_performance = {}
population_column_name = 'population_historical'
country_column_name = 'Entity'


# Function to retrieve the population and meat consumption data and merge into a DataFrame
def get_data():
    population_df = pd.read_csv("https://ourworldindata.org/grapher/population.csv?country=USA~BRA~AUS~ESP~ZWE~MDV~JPN&v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
    meat_df = pd.read_csv("data/Consumptionofmeatpercapita.csv")
    merged_df = pd.merge(population_df, meat_df, on=["Entity", "Year"], how="inner")
    return merged_df

def calculate_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
# Function to filter out the meat consumption category that will be used to train the model for meat consumption
def get_meat_consumption_features(meat_category):
    features = []
   
    for feature in features_set:
        if feature != meat_category:
            features.append(feature)
    
    return features

# Function to create a future DataFrame to save the population and meat consumption predictions
def get_future_years_df(year_to_start_prediction, future_years):
    future_years_df = pd.DataFrame({
        "Year": np.arange(year_to_start_prediction, year_to_start_prediction + future_years), 
        "Beef": [0] * future_years,
        "Sheep and goat": [0] * future_years,
        "Pork": [0] * future_years,
        "Other meats": [0] * future_years,
        "Fish and seafood": [0] * future_years,
        "Poultry": [0] * future_years,
    })
    
    return future_years_df

def get_lasso_best_score(X_train_scaled, y_train):
    # Define a range of alpha values to test
    alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]

    # Set up Lasso Regression and GridSearchCV
    lasso = Lasso()
    param_grid = {'alpha': alpha_values}

    # Perform GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best alpha value and corresponding score
    best_alpha = grid_search.best_params_['alpha']
    best_score = -grid_search.cv_results_['mean_test_score']
    
    return best_alpha, best_score, alpha_values

# Function to return the best Alpha score deteremined by GrandientCSV to use for the Ridge model
def get_ridge_best_score(X_train_scaled, y_train):
    
    # Define a range of alpha values to test
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Set up Ridge Regression and GridSearchCV
    ridge = Ridge()
    param_grid = {'alpha': alpha_values}

    # Perform Grid Search with 5-fold Cross-Validation
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best alpha value and the corresponding score
    best_alpha = grid_search.best_params_['alpha']
    mean_scores = -grid_search.cv_results_['mean_test_score']
    
    return best_alpha, mean_scores, alpha_values

# Helper function to convert numpy values to Python native types
def convert_to_native(value):
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return [float(x) for x in value]
    return value