import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from app.utils import *
from app.population_predictor import PopulationPredictor

# Function to train and predict the selected country and meat consumption category using the LinearRegression model
def train_and_predict_lr(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):

    print("Enter Linear")
    
    # Filter data for a specific Entity 
    country_entity = country  # Change this to the desired country
    country_entity_data = df
    #country_entity_data.fillna(0, inplace=True)
    
    #print(f"Country Entity Data: {country_entity_data}")

    # Feature selection
    features = get_meat_consumption_features(meat_category)
    
    
    features_population = features_set
    target_meat = meat_category
    target_population = population_column_name

    # Prepare features (X) and target (y)
    X = country_entity_data[features]
    
    y_meat = country_entity_data[target_meat]
    y_population = country_entity_data[target_population]
    
    # Split the data into training and testing sets
    X_train, X_test, y_meat_train, y_meat_test = train_test_split(X, y_meat, test_size=0.2, random_state=42)

    # Scale the meat consumption data
    meat_scaler = StandardScaler()
    X_train_scaled_meat = meat_scaler.fit_transform(X_train)
    X_test_scaled_meat = meat_scaler.transform(X_test)

    # Train Lasso Regression model for meat consumption
    lr_meat = LinearRegression()
    lr_meat.fit(X_train_scaled_meat, y_meat_train)
    
    # Evaluate the meat consumption model
    y_meat_pred = lr_meat.predict(X_test_scaled_meat)
    plot_models['linear_meat'] = {'prediction': y_meat_pred, 'y_test': y_meat_test}
    plot_models['linear_bagging_meat'] = bagging_model(X_train, y_meat_train, X_test, LinearRegression)
    
    # Add Gaussian noise to predictions
    mean = 0  # Mean of the Gaussian noise
    std_dev = 2  # Standard deviation of the Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, y_meat_pred.shape)
    y_pred_noisy = y_meat_pred + gaussian_noise
    plot_models['linear_noise_meat'] = {'prediction': y_pred_noisy, 'y_test': y_meat_test}
    
    metrics_meat = calculate_metrics(y_meat_test, y_meat_pred)
    
    # Split the data into training and testing sets
    X = country_entity_data[features_population]
    
    X_train, X_test, y_population_train, y_population_test = train_test_split(X, y_population, test_size=0.2, random_state=42)

    # Scale the population data
    population_scaler = StandardScaler()
    X_train_scaled_population = population_scaler.fit_transform(X_train)
    X_test_scaled_population = population_scaler.transform(X_test)

    # Train Lasso Regression model for population
    lr_population = LinearRegression()
    lr_population.fit(X_train_scaled_population, y_population_train)
    
    # Evaluate the population model
    y_population_pred = lr_population.predict(X_test_scaled_population)
    plot_models['linear_population'] = {'prediction': y_population_pred, 'y_test': y_population_test}
    plot_models['linear_bagging_population'] = bagging_model(X_train, y_population_train, X_test, LinearRegression)
    
    gaussian_noise = np.random.normal(mean, std_dev, y_population_pred.shape)
    y_pred_noisy = y_population_pred + gaussian_noise
    plot_models['linear_noise_population'] = {'prediction': y_pred_noisy, 'y_test': y_population_test}
    
    
    metrics_pop = calculate_metrics(y_population_test, y_population_pred)
    
    future_years_population = get_future_years_df(year_to_start_prediction, future_years)
    
    # Scale future data
    future_years_scaled_population = population_scaler.transform(future_years_population)
    future_years_meat = future_years_population.drop(columns=[meat_category])
    future_years_scaled_meat = meat_scaler.transform(future_years_meat)    

    # Make predictions
    pop_predictions = lr_population.predict(future_years_scaled_population)
    meat_predictions = lr_meat.predict(future_years_scaled_meat)
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)
    
    print("Returning from Linear.")
    
    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat, plot_models
    pass

def train_and_predict_lasso(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    # Filter data for a specific Entity
    country_entity = country  # Change this to the desired country
    country_entity_data = df.copy()
    #country_entity_data.fillna(0, inplace=True) 

    # Feature selection
    # Feature selection
    features = get_meat_consumption_features(meat_category)
    
    features_population = features_set
    target_meat = meat_category
    target_population = population_column_name

    # Prepare features (X) and target (y)
    X = country_entity_data[features]
    y_meat = country_entity_data[target_meat]
    y_population = country_entity_data[target_population]

    # Split the data into training and testing sets
    X_train, X_test, y_meat_train, y_meat_test = train_test_split(X, y_meat, test_size=0.2, random_state=42)

    # Scale the meat consumption data
    meat_scaler = StandardScaler()
    X_train_scaled_meat = meat_scaler.fit_transform(X_train)
    X_test_scaled_meat = meat_scaler.transform(X_test)

    best_param, mean_scores, alpha_values = get_lasso_best_score(X_train_scaled_meat, y_meat_train)
    lasso_best_performance['meat'] = {'best_param':best_param,'features':features, 'mean_scores': mean_scores, 'alpha_values':alpha_values}
    # Train Lasso Regression model for meat consumption
    lasso_meat = Lasso(alpha=best_param)
    lasso_meat.fit(X_train_scaled_meat, y_meat_train)

    # Evaluate the meat consumption model
    y_meat_pred = lasso_meat.predict(X_test_scaled_meat)
    plot_models['lasso_meat'] = {'prediction': y_meat_pred, 'y_test': y_meat_test}
    plot_models['lasso_bagging_meat'] = bagging_model(X_train, y_meat_train, X_test, Lasso)
    # Add Gaussian noise to predictions
    mean = 0  # Mean of the Gaussian noise
    std_dev = 2  # Standard deviation of the Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, y_meat_pred.shape)
    y_pred_noisy = y_meat_pred + gaussian_noise
    plot_models['lasso_noise_meat'] = {'prediction': y_pred_noisy, 'y_test': y_meat_test}
    
    metrics_meat = calculate_metrics(y_meat_test, y_meat_pred)
    
    # Split the data into training and testing sets
    X = country_entity_data[features_population]
    X_train, X_test, y_population_train, y_population_test = train_test_split(X, y_population, test_size=0.2, random_state=42)

    # Scale the population data
    population_scaler = StandardScaler()
    X_train_scaled_population = population_scaler.fit_transform(X_train)
    X_test_scaled_population = population_scaler.transform(X_test)
    
    best_param, mean_scores, alpha_values = get_lasso_best_score(X_train_scaled_population, y_population_train)
    lasso_best_performance['population'] = {'best_param':best_param,'features':features, 'mean_scores': mean_scores, 'alpha_values':alpha_values}

    # Train Lasso Regression model for population
    lasso_population = Lasso(alpha=best_param)
    lasso_population.fit(X_train_scaled_population, y_population_train)

    # Evaluate the population model
    y_population_pred = lasso_population.predict(X_test_scaled_population)
    plot_models['lasso_population'] = {'prediction': y_population_pred, 'y_test': y_population_test}
    plot_models['lasso_bagging_population'] = bagging_model(X_train, y_population_train, X_test, Lasso)
    # Add Gaussian noise to predictions
    gaussian_noise = np.random.normal(mean, std_dev, y_population_pred.shape)
    y_pred_noisy = y_population_pred + gaussian_noise
    plot_models['lasso_noise_population'] = {'prediction': y_pred_noisy, 'y_test': y_population_test}
    
    metrics_pop = calculate_metrics(y_population_test, y_population_pred)
    
    # Predict for the next 10 years
    future_years_population = future_years_population = get_future_years_df(year_to_start_prediction, future_years)

    # Scale future data
    future_years_scaled_population = population_scaler.transform(future_years_population)
    future_years_meat = future_years_population.drop(columns=[meat_category])
    future_years_scaled_meat = meat_scaler.transform(future_years_meat)
    

    # Make predictions
    pop_predictions = lasso_population.predict(future_years_scaled_population)
    meat_predictions = lasso_meat.predict(future_years_scaled_meat)
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)

    print("Returning from Lasso.")
    
    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat, lasso_best_performance, plot_models
    pass

def train_and_predict_ridge(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    print("Entering Ridge.")
    country_entity = country  # Change this to the desired country
    country_entity_data = df.copy()
    #country_entity_data.fillna(0, inplace=True) 
    
    # Feature selection
    features = get_meat_consumption_features(meat_category)
    
    features_population = features_set
    target_meat = meat_category
    target_population = population_column_name

    # Prepare features (X) and target (y)
    X = country_entity_data[features]
    y_meat = country_entity_data[target_meat]
    y_population = country_entity_data[target_population]

    # Split the data into training and testing sets
    X_train, X_test, y_meat_train, y_meat_test = train_test_split(X, y_meat, test_size=0.2, random_state=42)

    # Scale the meat consumption data
    meat_scaler = StandardScaler()
    X_train_scaled_meat = meat_scaler.fit_transform(X_train)
    X_test_scaled_meat = meat_scaler.transform(X_test)
    
    best_param, mean_scores, alpha_values = get_ridge_best_score(X_train_scaled_meat,y_meat_train)
       
    # Train Ridge Regression model for meat consumption
    r_meat = Ridge(alpha=best_param)
    r_meat.fit(X_train_scaled_meat, y_meat_train)
    
    ridge_best_performance['meat'] = {'best_param':best_param,'features':features, 'mean_scores': mean_scores, 'alpha_values':alpha_values}

    # Evaluate the meat consumption model
    y_meat_pred = r_meat.predict(X_test_scaled_meat)
    plot_models['ridge_meat'] = {'prediction': y_meat_pred, 'y_test': y_meat_test}
    plot_models['ridge_bagging_meat'] = bagging_model(X_train, y_meat_train, X_test, Ridge)
    
    # Add Gaussian noise to predictions
    mean = 0  # Mean of the Gaussian noise
    std_dev = 2  # Standard deviation of the Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, y_meat_pred.shape)
    y_pred_noisy = y_meat_pred + gaussian_noise
    plot_models['ridge_noise_meat'] = {'prediction': y_pred_noisy, 'y_test': y_meat_test}
    
    metrics_meat = calculate_metrics(y_meat_test, y_meat_pred)
    
    # Split the data into training and testing sets
    X = country_entity_data[features_population]
    X_train, X_test, y_population_train, y_population_test = train_test_split(X, y_population, test_size=0.2, random_state=42)

    # Scale the population data
    population_scaler = StandardScaler()
    X_train_scaled_population = population_scaler.fit_transform(X_train)
    X_test_scaled_population = population_scaler.transform(X_test)

    best_param, mean_scores, alpha_values = get_ridge_best_score(X_train_scaled_population,y_population_train)
    
    # Train Lasso Regression model for population
    r_population = Ridge(alpha=best_param)
    r_population.fit(X_train_scaled_population, y_population_train)

    ridge_best_performance['population'] = {'best_param':best_param,'features':features_population, 'mean_scores': mean_scores, 'alpha_values':alpha_values}

    # Evaluate the population model
    y_population_pred = r_population.predict(X_test_scaled_population)
    plot_models['ridge_population'] = {'prediction': y_population_pred, 'y_test': y_population_test}
    plot_models['ridge_bagging_population'] = bagging_model(X_train, y_population_train, X_test, Ridge)
    
    # Add Gaussian noise to predictions
    gaussian_noise = np.random.normal(mean, std_dev, y_population_pred.shape)
    y_pred_noisy = y_population_pred + gaussian_noise
    plot_models['ridge_noise_population'] = {'prediction': y_pred_noisy, 'y_test': y_population_test}

       
    metrics_pop = calculate_metrics(y_population_test, y_population_pred)
    
    # Predict for the next 10 years
    future_years_population = future_years_population = get_future_years_df(year_to_start_prediction, future_years)

    # Scale future data
    future_years_scaled_population = population_scaler.transform(future_years_population)
    future_years_meat = future_years_population.drop(columns=[meat_category])
    future_years_scaled_meat = meat_scaler.transform(future_years_meat)
    

    # Make predictions
    pop_predictions = r_population.predict(future_years_scaled_population)
    meat_predictions = r_meat.predict(future_years_scaled_meat)
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)

    print("Returning from Ridge.")
    
    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat, ridge_best_performance, plot_models
    pass