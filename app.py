import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import streamlit as st
import seaborn as sns

# Year to start the prediction
year_to_start_prediction = 2022

# Number of future years to predict
future_years = 7

# Initialize the visibility attribute if it doesn't exist
if 'visibility' not in st.session_state:
    st.session_state.visibility = 'visible'  # or False, depending on your needs

# Initialize the 'disabled' attribute if it doesn't exist
if 'disabled' not in st.session_state:
    st.session_state.disabled = False  # or True, depending on your logic
    
if 'future' not in st.session_state:
    st.session_state['future'] = future_years
            
if 'year' not in st.session_state:
    st.session_state['year'] = year_to_start_prediction

if 'meat_option' not in st.session_state:
    st.session_state['meat_option'] = ""

if 'plot_models' not in st.session_state:
    st.session_state['plot_models'] = {}

if 'countries_selected' not in st.session_state:
    st.session_state['countries_selected'] = []

if 'ridge_best_performance' not in st.session_state:
    st.session_state['ridge_best_performance'] = {}

def get_data():
    population_df = pd.read_csv("https://ourworldindata.org/grapher/population.csv?country=USA~BRA~AUS~ESP~ZWE~MDV~JPN&v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
    meat_df = pd.read_csv("Consumption of meat per capita.csv")
    merged_df = pd.merge(population_df, meat_df, on=["Entity", "Year"], how="inner")
    #merged_df.to_csv("merged_data.csv")
    return merged_df #pd.merge(population_df, meat_df, on=["Entity", "Year"], how="inner")

def filter_and_prepare_training_data(df, column_to_filter, filter_value, X_column, Y_column):
    filtered_df = df[df[column_to_filter] == filter_value].copy()
    X = df[[X_column]].values
    Y = df[Y_column].values
    return filtered_df, X, Y

def split_into_training_and_test(X, Y, test_perc = 0.2, random_state_val = 42):
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_val)
    return X_train, X_test, y_train, y_test

def train_and_predict_lr(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    # Filter data for a specific Entity 
    country_entity = country  # Change this to the desired country
    country_entity_data = df[df[country_column_name] == country_entity]
    country_entity_data.fillna(0, inplace=True) 

    # Feature selection
    features = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood"]
    features_population = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood", meat_category]
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
    st.session_state['plot_models']['linear_meat'] = {'prediction': y_meat_pred, 'y_test': y_meat_test}
    #st.write(y_meat_pred)
    #st.write(f"No scaling for {meat_category} Model MSE for {country}:", mean_squared_error(y_meat_test, y_meat_pred))
    #metrics_train_meat, metrics_test_meat = evaluate_model(lasso_meat,X_train,X_test,y_meat_train,y_meat_test)
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
    st.session_state['plot_models']['linear_population'] = {'prediction': y_population_pred, 'y_test': y_population_test}

    #st.write(f"No scaling for {population_column_name} Model MSE for {country}:", mean_squared_error(y_population_test, y_population_pred))
    #metrics_train_population, metrics_test_population = evaluate_model(lasso_population,X_train,X_test,y_population_train,y_population_test)
    metrics_pop = calculate_metrics(y_population_test, y_population_pred)
    
    # Predict for the next 10 years
    future_years_population = pd.DataFrame({
        "Year": np.arange(year_to_start_prediction, year_to_start_prediction + future_years), 
        "Beef": [0] * 10,
        "Sheep and goat": [0] * 10,
        "Pork": [0] * 10,
        "Other meats": [0] * 10,
        "Fish and seafood": [0] * 10,
        meat_category: [0] * 10,
    })

    # Scale future data
    future_years_scaled_population = population_scaler.transform(future_years_population)
    future_years_meat = future_years_population.drop(columns=[meat_category])
    future_years_scaled_meat = meat_scaler.transform(future_years_meat)
    

    # Make predictions
    pop_predictions = lr_population.predict(future_years_scaled_population)
    meat_predictions = lr_meat.predict(future_years_scaled_meat)
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)

    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat
    
def train_and_predict_lasso(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    # Filter data for a specific Entity 
    country_entity = country  # Change this to the desired country
    country_entity_data = df[df[country_column_name] == country_entity]
    country_entity_data.fillna(0, inplace=True) 

    # Feature selection
    features = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood"]
    features_population = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood", meat_category]
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
    lasso_meat = Lasso(alpha=0.1)
    lasso_meat.fit(X_train_scaled_meat, y_meat_train)

    # Evaluate the meat consumption model
    y_meat_pred = lasso_meat.predict(X_test_scaled_meat)
    st.session_state['plot_models']['lasso_meat'] = {'prediction': y_meat_pred, 'y_test': y_meat_test}
    
    metrics_meat = calculate_metrics(y_meat_test, y_meat_pred)
    
    # Split the data into training and testing sets
    X = country_entity_data[features_population]
    X_train, X_test, y_population_train, y_population_test = train_test_split(X, y_population, test_size=0.2, random_state=42)

    # Scale the population data
    population_scaler = StandardScaler()
    X_train_scaled_population = population_scaler.fit_transform(X_train)
    X_test_scaled_population = population_scaler.transform(X_test)

    # Train Lasso Regression model for population
    lasso_population = Lasso(alpha=0.1)
    lasso_population.fit(X_train_scaled_population, y_population_train)

    # Evaluate the population model
    y_population_pred = lasso_population.predict(X_test_scaled_population)
    st.session_state['plot_models']['lasso_population'] = {'prediction': y_population_pred, 'y_test': y_population_test}

    metrics_pop = calculate_metrics(y_population_test, y_population_pred)
    
    # Predict for the next 10 years
    future_years_population = pd.DataFrame({
        "Year": np.arange(year_to_start_prediction, year_to_start_prediction + future_years), 
        "Beef": [0] * 10,
        "Sheep and goat": [0] * 10,
        "Pork": [0] * 10,
        "Other meats": [0] * 10,
        "Fish and seafood": [0] * 10,
        meat_category: [0] * 10,
    })

    # Scale future data
    future_years_scaled_population = population_scaler.transform(future_years_population)
    future_years_meat = future_years_population.drop(columns=[meat_category])
    future_years_scaled_meat = meat_scaler.transform(future_years_meat)
    

    # Make predictions
    pop_predictions = lasso_population.predict(future_years_scaled_population)
    meat_predictions = lasso_meat.predict(future_years_scaled_meat)
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)

    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat

def get_ridge_best_score(X_train_scaled, y_train):
    
    # Define a range of alpha values to test
    alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    # Set up Ridge Regression and GridSearchCV
    ridge = Ridge()
    param_grid = {'alpha': alpha_values}

    # Perform Grid Search with 5-fold Cross-Validation
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=(len(alpha_values)+1), scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best alpha value and the corresponding score
    best_alpha = grid_search.best_params_['alpha']
    mean_scores = -grid_search.cv_results_['mean_test_score']
    
    return best_alpha, mean_scores, alpha_values
    
def train_and_predict_ridge(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
   # Filter data for a specific Entity
    country_entity = country  # Change this to the desired country
    country_entity_data = df[df[country_column_name] == country_entity]
    country_entity_data.fillna(0, inplace=True) 
    

    # Feature selection
    features = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood"]
    features_population = ["Year", "Beef", "Sheep and goat", "Pork", "Other meats", "Fish and seafood", meat_category]
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
    
    st.session_state['ridge_best_performance']['meat'] = {'best_param':best_param,'features':features, 'mean_scores': mean_scores, 'alpha_values':alpha_values}

    # Evaluate the meat consumption model
    y_meat_pred = r_meat.predict(X_test_scaled_meat)
    st.session_state['plot_models']['ridge_meat'] = {'prediction': y_meat_pred, 'y_test': y_meat_test}
    
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

    st.session_state['ridge_best_performance']['population'] = {'best_param':best_param,'features':features_population, 'mean_scores': mean_scores, 'alpha_values':alpha_values}

    # Evaluate the population model
    y_population_pred = r_population.predict(X_test_scaled_population)
    st.session_state['plot_models']['ridge_population'] = {'prediction': y_population_pred, 'y_test': y_population_test}

       
    metrics_pop = calculate_metrics(y_population_test, y_population_pred)
    
    # Predict for the next 10 years
    future_years_population = pd.DataFrame({
        "Year": np.arange(year_to_start_prediction, year_to_start_prediction + future_years), 
        "Beef": [0] * 10,
        "Sheep and goat": [0] * 10,
        "Pork": [0] * 10,
        "Other meats": [0] * 10,
        "Fish and seafood": [0] * 10,
        meat_category: [0] * 10,
    })

    # Scale future data
    future_years_scaled_population = population_scaler.transform(future_years_population)
    future_years_meat = future_years_population.drop(columns=[meat_category])
    future_years_scaled_meat = meat_scaler.transform(future_years_meat)
    

    # Make predictions
    pop_predictions = r_population.predict(future_years_scaled_population)
    meat_predictions = r_meat.predict(future_years_scaled_meat)
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)

    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat

def calculate_metrics(y_values, prediction):
    metrics = {}
    mse_val = mean_squared_error(y_values, prediction)
    metrics["mse"] = mse_val
    metrics["rmse"] = np.sqrt(mse_val)
    metrics["mae"] = mean_absolute_error(y_values,prediction)
    metrics["r2"] = r2_score(y_values, prediction)
    return metrics

def get_combined_future_df(population_prediction, meat_prediction, years_list, meat_category):
    # Future Predicition
    predicted_data = []
    for year, pop, meat in zip(years_list, population_prediction, meat_prediction):
        predicted_data.append([year, pop, meat])
    
    return pd.DataFrame(predicted_data, columns=['Year', 'future_population', f'future_{meat_category}_consumption'])

def plot_meat_prediction(df_pred, meat_category, country, year_start, year_end):
    # Plot the results
    # Plot the results
    ax = df_pred[['Year',f'future_{meat_category}_consumption']].plot(kind='bar', title =f'Predicted {meat_category} consumption per capita from {year_start} to {year_end}', legend=True, fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Future Consumption (per capita)", fontsize=12)
    st.pyplot(plt)

def plot_population_prediction(df_pred, country, year_start, year_end):
    # Plot the results
    ax = df_pred[['Year','future_population']].plot(kind='bar', title =f'Predicted {country} Population from {year_start} to {year_end}', legend=True, fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Future Population (millions)", fontsize=12)
    st.pyplot(plt)


def plot_ridge_best_match(mean_scores, alpha_values, category):
    # Extract mean scores for each alpha
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_values, mean_scores, marker='o', linestyle='--', color='b')
    plt.xscale('log')  # Use log scale for alpha values
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Mean Cross-Validated MSE')
    plt.title(f'Grid Search Results: Alpha vs Mean Cross-Validated MSE for {category}')
    plt.grid(True)
    st.pyplot(plt)
    
def plot_actual_vs_predicted(model_name, y_test, model_predictions, category):
    plt.scatter(y_test, model_predictions, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title(f"{model_name} Regression Actual vs Predicted for {category}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.tight_layout()
    st.pyplot(plt)
    

def filter_countries_by_names(data_frame, countries_list, country_column_name):
    filtered_df = data_frame[data_frame[country_column_name].isin(countries_list)]
    return filtered_df

def print_prediction(years_list,pop_predicted_df, meat_predicted_df, metrics_pop, metrics_meat, model_name):
    
    st.subheader(f"Population Predictions {model_name} (millions)", divider=True)
    for year, pop in zip(years_list, pop_predicted_df):
        st.write(f"{year}: {pop:.2f}")
        
    st.subheader(f"Meat Consumption Predictions {model_name} (kg per capita)", divider=True)
    for year, meat in zip(years_list, meat_predicted_df):
        st.write(f"{year}: {meat:.2f}")
      
    st.subheader("Model Scores", divider=True)
    st.html("<div>")
    st.subheader("Population")
    print_metrics(metrics_pop)
    st.html("</div><div>")
    st.subheader("Meat Consumption")
    print_metrics(metrics_meat)
    st.html("</div>")


def print_metrics(metrics_info):
    
    st.html("<div><table>")
    st.html(f"<tr><td><b>MSE:</b> {metrics_info['mse']:.4f}</td></tr>")
    st.html(f"<tr><td><b>RMSE:</b> {metrics_info['rmse']:.4f}</td></tr>")
    st.html(f"<tr><td><b>MAE:</b> {metrics_info['mae']:.4f}</td>")
    st.html(f"<tr><td><b>R2 Score:</b> {metrics_info['r2']:.4f}</td></tr>")
    st.html("</table></div>")

    
def main():
    future_years = int(st.session_state['future'])

    tab1 = st.tabs(["Population and Meat Consumption Predictor"])
    # Year to start the prediction
    year_to_start_prediction = int(st.session_state['year'])
    
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)

    st.subheader("This application predicts population and meat consumption using Linear Regression, Boosting, and Decision Tree models.", divider=True)
    # Load or create data
    df = get_data() #create_sample_data()
    
    countries_to_pull = ['United States', 'Brazil', 'Australia', 'Spain', 'Zimbabwe', 'Maldives', 'Japan']
    filtered_population_df = filter_countries_by_names(df, countries_to_pull, 'Entity')
    meat_category = st.session_state['meat_option']
    
    # List of countries to analyze
    countries = filtered_population_df['Entity'].unique()
    counter = 0
    tab_country = st.tabs(countries.tolist())
    for country in countries:
        
        with tab_country[counter]:
            
            st.write(f"Predictions for {country}:")

            
            years, pop_pred, meat_pred, metrics_pop, metrics_meat = train_and_predict_lr(filtered_population_df, country, 'Entity', 'population_historical', 'Year', meat_category, future_years)
            combined_future_df = get_combined_future_df(pop_pred,meat_pred,future_X,meat_category)
            st.html("<div>")
            st.subheader("Linear Regression",divider=True)
            st.subheader("Actual vs. Prediction Chart for Meat Consumption", divider=True)
            plot_actual_vs_predicted("Linear", st.session_state['plot_models']['linear_meat']['prediction'],st.session_state['plot_models']['linear_meat']['y_test'],"Linear Regression")
            st.subheader(f"Actual vs. Prediction Chart for Population", divider=True)
            plot_actual_vs_predicted("Linear",st.session_state['plot_models']['linear_population']['prediction'],st.session_state['plot_models']['linear_population']['y_test'],"Linear Regression")
            plot_population_prediction(combined_future_df,country,year_to_start_prediction,(year_to_start_prediction+future_years))
            plot_meat_prediction(combined_future_df,meat_category,country,year_to_start_prediction,(year_to_start_prediction+future_years))
                #plot_predictions(filtered_population_df, country, future_years, pop_pred, meat_pred, 'Entity', 'Year', 'population_historical', st.session_state['meat_option'],st.session_state['meat_option'],'LinearRegression')
            print_prediction(years,pop_pred, meat_pred, metrics_pop, metrics_meat,"LinearRegression")
                
                
            ls_years, ls_pop_pred, ls_meat_pred, ls_metrics_pop, ls_metrics_meat = train_and_predict_lasso(filtered_population_df, country, 'Entity', 'population_historical', 'Year', meat_category, future_years)
            combined_future_df = get_combined_future_df(ls_pop_pred,ls_meat_pred,future_X,meat_category)
            st.html("</div>")
            st.subheader("Lasso Regression",divider=True)
            st.subheader("Actual vs. Prediction Chart for Meat Consumption", divider=True)
            plot_actual_vs_predicted("Lasso", st.session_state['plot_models']['lasso_meat']['prediction'],st.session_state['plot_models']['lasso_meat']['y_test'],"Lasso Regression")
            st.subheader(f"Actual vs. Prediction Chart for Population", divider=True)
            plot_actual_vs_predicted("Lasso",st.session_state['plot_models']['lasso_population']['prediction'],st.session_state['plot_models']['lasso_population']['y_test'],"Lasso Regression")
            plot_population_prediction(combined_future_df,country,year_to_start_prediction,(year_to_start_prediction+future_years))
            plot_meat_prediction(combined_future_df,meat_category,country,year_to_start_prediction,(year_to_start_prediction+future_years))
                #plot_predictions(filtered_population_df, country, future_years, ls_pop_pred, ls_meat_pred, 'Entity', 'Year', 'population_historical', st.session_state['meat_option'],st.session_state['meat_option'],'Lasso')
            print_prediction(ls_years,ls_pop_pred, ls_meat_pred, ls_metrics_pop, ls_metrics_meat,"LassoRegression")
            st.html("</div>")
            
            
            r_years, r_pop_pred, r_meat_pred, r_metrics_pop, r_metrics_meat = train_and_predict_ridge(filtered_population_df, country, 'Entity', 'population_historical', 'Year', meat_category, future_years)
            combined_future_df = get_combined_future_df(r_pop_pred,r_meat_pred,future_X,meat_category)
            st.html("<div>")    
            st.subheader("Ridge Regression",divider=True)
            st.subheader("Actual vs. Prediction Chart for Meat Consumption", divider=True)
            plot_actual_vs_predicted("Ridge", st.session_state['plot_models']['ridge_meat']['prediction'],st.session_state['plot_models']['ridge_meat']['y_test'],"Ridge Regression")
            st.subheader(f"Actual vs. Prediction Chart for Population", divider=True)
            plot_actual_vs_predicted("Ridge",st.session_state['plot_models']['ridge_population']['prediction'],st.session_state['plot_models']['ridge_population']['y_test'],"Ridge Regression")
            plot_population_prediction(combined_future_df,country,year_to_start_prediction,(year_to_start_prediction+future_years))
            plot_meat_prediction(combined_future_df,meat_category,country,year_to_start_prediction,(year_to_start_prediction+future_years))
                #plot_predictions(filtered_population_df, country, future_years, r_pop_pred, r_meat_pred, 'Entity', 'Year', 'population_historical', st.session_state['meat_option'],st.session_state['meat_option'],'GradientBoost')
            best_matches = st.session_state['ridge_best_performance']
            st.subheader(f'GradientCV Scores for Population', divider=True)
            plot_ridge_best_match(best_matches['population']['mean_scores'],best_matches['population']['alpha_values'],"Population")
            st.subheader(f'GradientCV Scores for {meat_category} consumption', divider=True)
            plot_ridge_best_match(best_matches['meat']['mean_scores'],best_matches['meat']['alpha_values'],"Meat Consumption")
            print_prediction(r_years,r_pop_pred, r_meat_pred, r_metrics_pop, r_metrics_meat,"RidgeRegression")
                
            st.html("</div>")       
            
        counter = counter + 1
        
#if __name__ == "__main__":

df = get_data()
column_names = list(df.columns.values)

meat_categories = column_names[4:len(column_names)] #column_names.remove(["Entity","Code","Year","population_historical"])
countries_list = df['Entity'].unique()
max_year = df['Year'].unique().max()

    
st.session_state['future'] = st.sidebar.text_input(
    "Enter number of future years to predict 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled
)

st.sidebar.write(f"Year cannot be before {max_year}")        
st.session_state['year'] = st.sidebar.text_input(
    "Enter Year to start prediction 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled
)


st.session_state['meat_option'] = st.sidebar.selectbox(
    "How would you like to be contacted?",
    (meat_categories),
)

st.session_state['countries_selected'] = st.sidebar.multiselect(
    "Select Countries to predict.",
    countries_list
)

with st.form("Predictions"):
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        main()
        

