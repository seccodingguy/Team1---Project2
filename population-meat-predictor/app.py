import os
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression, Lasso, Ridge
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split, GridSearchCV
from app.models import *
from app.utils import *
from flask_session import Session
import redis
#import json

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure Redis for storing the session data on the server-side
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.from_url('redis://127.0.0.1:6379')

# Create and initialize the Flask-Session object AFTER `app` has been configured
server_session = Session(app)

@app.route('/')
def index():
    df = get_data()
    session['df'] = df.to_json(orient="split")
    meat_categories = [col for col in df.columns if col not in ["Entity", "Code", "Year", "population_historical"]]
    countries = df['Entity'].unique().tolist()
    max_year = int(df['Year'].max())
    
    return render_template('index.html', 
                         meat_categories=meat_categories,
                         countries=countries,
                         max_year=max_year,
                         min_year=2021)

@app.route('/plots')
def plots():
    return render_template('plots.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        required_fields = ['country', 'meat_category', 'start_year', 'future_years']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        selected_country = data['country']
        selected_meat = data['meat_category']
        start_year = int(data['start_year'])
        future_years = int(data['future_years'])

        # Input validation
        if future_years < 1 or future_years > 10:
            return jsonify({'error': 'Future years must be between 1 and 10'}), 400

        df = get_data()
        
        # Filter data for selected country
        country_data = df[df['Entity'] == selected_country].copy()
        
        if country_data.empty:
            return jsonify({'error': f'No data found for {selected_country}'}), 400

        # Generate future years
        future_years_list = list(range(start_year, start_year + future_years))
        future_X = np.array(future_years_list).reshape(-1, 1)

        # Perform predictions
        try:
            future_X, pop_predictions, meat_predictions, metrics_pop, metrics_meat, plot_models = train_and_predict_lr(
                country_data,
                selected_country,
                'Entity',
                'population_historical',
                'Year',
                selected_meat,
                future_years
            )
            future_X, pop_predictions_lasso, meat_predictions_lasso, metrics_pop_lasso, metrics_meat_lasso, lasso_best_performance, plot_models_lasso = train_and_predict_lasso(
                country_data,
                selected_country,
                'Entity',
                'population_historical',
                'Year',
                selected_meat,
                future_years
            )
            future_X, pop_predictions_ridge, meat_predictions_ridge, metrics_pop_ridge, metrics_meat_ridge, ridge_best_performance, plot_models_ridge = train_and_predict_ridge(
                country_data,
                selected_country,
                'Entity',
                'population_historical',
                'Year',
                selected_meat,
                future_years
            )
        except Exception as model_error:
            print(f"Prediction error: {str(model_error)}")
            return jsonify({'error': 'Error generating predictions. Please try different parameters.'}), 500

        print("Preparing the results.")
        
        # Prepare results
        results = {}
        results['linear_regression'] = {
                'years': future_years_list,
                'population_predictions': [float(x) for x in pop_predictions],
                'meat_predictions': [float(x) for x in meat_predictions],
                'metrics_population': {k: float(v) for k, v in metrics_pop.items()},
                'metrics_meat': {k: float(v) for k, v in metrics_meat.items()},
                'population_actual': [float(x) for x in plot_models['linear_population']['y_test']],
                'population_predicted': [float(x) for x in plot_models['linear_population']['prediction']],
                'meat_actual': [float(x) for x in plot_models['linear_meat']['y_test']],
                'meat_predicted': [float(x) for x in plot_models['linear_meat']['prediction']],
                'meat_bagged' : [float(x) for x in plot_models['linear_bagging_meat']],
                'population_bagged' : [float(x) for x in plot_models['linear_bagging_population']],
                'meat_noise' : [float(x) for x in plot_models['linear_noise_population']['prediction']],
                'population_noise' : [float(x) for x in plot_models['linear_noise_population']['prediction']]
            }
        
        jsonify(results)
        print("After Linear Regression Result")
        
        results['lasso_regression'] = {
                'years': future_years_list,
                'population_predictions': [float(x) for x in pop_predictions_lasso],
                'meat_predictions': [float(x) for x in meat_predictions_lasso],
                'metrics_population': {k: float(v) for k, v in metrics_pop_lasso.items()},
                'metrics_meat': {k: float(v) for k, v in metrics_meat_lasso.items()},
                'population_actual': [float(x) for x in plot_models_lasso['lasso_population']['y_test']],
                'population_predicted': [float(x) for x in plot_models_lasso['lasso_population']['prediction']],
                'meat_actual': [float(x) for x in plot_models_lasso['lasso_meat']['y_test']],
                'meat_predicted': [float(x) for x in plot_models_lasso['lasso_meat']['prediction']],
                'meat_bagged' : [float(x) for x in plot_models_lasso['lasso_bagging_meat']],
                'population_bagged' : [float(x) for x in plot_models_lasso['lasso_bagging_population']],
                'meat_noise' : [float(x) for x in plot_models_lasso['lasso_noise_population']['prediction']],
                'population_noise' : [float(x) for x in plot_models_lasso['lasso_noise_population']['prediction']]
            }
        
        jsonify(results)
        print("After Lasso Regression Result")
        
        results['ridge_regression'] = {
                'years': future_years_list,
                'population_predictions': [float(x) for x in pop_predictions_ridge],
                'meat_predictions': [float(x) for x in meat_predictions_ridge],
                'metrics_population': {k: float(v) for k, v in metrics_pop_ridge.items()},
                'metrics_meat': {k: float(v) for k, v in metrics_meat_ridge.items()},
                'population_actual': [float(x) for x in plot_models_ridge['ridge_population']['y_test']],
                'population_predicted': [float(x) for x in plot_models_ridge['ridge_population']['prediction']],
                'meat_actual': [float(x) for x in plot_models_ridge['ridge_meat']['y_test']],
                'meat_predicted': [float(x) for x in plot_models_ridge['ridge_meat']['prediction']],
                'meat_bagged' : [float(x) for x in plot_models_ridge['ridge_bagging_meat']],
                'population_bagged' : [float(x) for x in plot_models_ridge['ridge_bagging_population']],
                'meat_noise' : [float(x) for x in plot_models_ridge['ridge_noise_population']['prediction']],
                'population_noise' : [float(x) for x in plot_models_ridge['ridge_noise_population']['prediction']]
            }
        
        jsonify(results)
        print("After Ridge Regress Search Result")
        
        results['ridge_grid_search'] = {
                'alphas': [float(x) for x in ridge_best_performance['population']['alpha_values']],
                'population_scores': [float(x) for x in ridge_best_performance['population']['mean_scores']],
                'meat_scores': [float(x) for x in ridge_best_performance['meat']['mean_scores']],
                'best_params': {
                    'population': ridge_best_performance['population']['best_param'],
                    'meat': ridge_best_performance['meat']['best_param']
                }
            }
        
        jsonify(results)
        print("After Ridge Grid Search Result")
        
        results['lasso_grid_search'] = {
                'alphas': [float(x) for x in lasso_best_performance['population']['alpha_values']],
                'population_scores': convert_to_native(lasso_best_performance['population']['mean_scores']),
                'meat_scores': convert_to_native(lasso_best_performance['meat']['mean_scores']),
                'best_params': {
                    'population': convert_to_native(lasso_best_performance['population']['best_param']),
                    'meat': convert_to_native(lasso_best_performance['meat']['best_param'])
                }
            }
        
        jsonify(results)
        print("After Lasso Grid Search Result")
        
        return jsonify(results)

    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)