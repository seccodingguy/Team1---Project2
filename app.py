import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

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

if 'evaluate_models' not in st.session_state:
    st.session_state['evaluate_models'] = {}

tab1 = st.tabs(["Filter Data"])

def get_data():
    population_df = pd.read_csv("https://ourworldindata.org/grapher/population.csv?country=USA~BRA~AUS~ESP~ZWE~MDV~JPN&v=1&csvType=full&useColumnShortNames=true", storage_options = {'User-Agent': 'Our World In Data data fetch/1.0'})
    meat_df = pd.read_csv("Consumption of meat per capita.csv")
    
    return pd.merge(population_df, meat_df, on=["Entity", "Year"], how="inner")

def filter_and_prepare_training_data(df, column_to_filter, filter_value, X_column, Y_column):
    filtered_df = df[df[column_to_filter] == filter_value].copy()
    X = df[[X_column]].values
    Y = df[Y_column].values
    return filtered_df, X, Y

def split_into_training_and_test(X, Y, test_perc = 0.2, random_state_val = 42):
    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state_val)
    return X_train, X_test, y_train, y_test

def train_and_predict_dt(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    country_data, X, y_pop = filter_and_prepare_training_data(df,country_column_name,country,year_column_name,population_column_name)
    country_data, X, y_meat = filter_and_prepare_training_data(df,country_column_name,country,year_column_name,meat_category)
    
    X_train_pop, X_test_pop, y_pop_train, y_pop_test = split_into_training_and_test(X, y_pop)
    X_train_meat, X_test_meat, y_meat_train, y_meat_test = split_into_training_and_test(X, y_meat)
    
    model_pop = DecisionTreeRegressor(random_state=42, max_depth=5)
    model_meat = DecisionTreeRegressor(random_state=42, max_depth=5)
    
    metrics_train, metrics_test = evaluate_model(model_pop,X_train_pop,X_test_pop,y_pop_train,y_pop_test)
    st.session_state['evaluate_models']["decisiontree_pop_train"] = metrics_train
    st.session_state['evaluate_models']["decisiontree_pop_test"] = metrics_test
    #st.write("Meat:")
    metrics_train, metrics_test = evaluate_model(model_meat,X_train_meat,X_test_meat,y_meat_train,y_meat_test)
    st.session_state['evaluate_models']["decisiontree_meat_train"] = metrics_train
    st.session_state['evaluate_models']["decisiontree_meat_test"] = metrics_test

    
    model_pop.fit(X, y_pop)
    model_meat.fit(X, y_meat)

    
    # Generate future years for prediction
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)
    
    # Make predictions
    pop_predictions = model_pop.predict(future_X)
    meat_predictions = model_meat.predict(future_X)

    # Calculate metrics
    metrics_pop = calculate_metrics(y_pop, model_pop.predict(X))
    metrics_meat = calculate_metrics(y_meat, model_meat.predict(X))
    
    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat


def train_and_predict_lr(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    country_data, X, y_pop = filter_and_prepare_training_data(df,country_column_name,country,year_column_name,population_column_name)
    country_data, X, y_meat = filter_and_prepare_training_data(df,country_column_name,country,year_column_name,meat_category)
    
    X_train_pop, X_test_pop, y_pop_train, y_pop_test = split_into_training_and_test(X, y_pop)
    X_train_meat, X_test_meat, y_meat_train, y_meat_test = split_into_training_and_test(X, y_meat)
        
    # Create and train models
    model_pop = LinearRegression()
    model_meat = LinearRegression()
    
    #st.write("Training Evaluation Results for DecisionTree:")
    #st.write("Population:")
    metrics_train, metrics_test = evaluate_model(model_pop,X_train_pop,X_test_pop,y_pop_train,y_pop_test)
    st.session_state['evaluate_models']["linearregression_pop_train"] = metrics_train
    st.session_state['evaluate_models']["linearregression_pop_test"] = metrics_test
    #st.write("Meat:")
    metrics_train, metrics_test = evaluate_model(model_meat,X_train_meat,X_test_meat,y_meat_train,y_meat_test)
    st.session_state['evaluate_models']["linearregression_meat_train"] = metrics_train
    st.session_state['evaluate_models']["linearregression_meat_test"] = metrics_test

    
    model_pop.fit(X, y_pop)
    model_meat.fit(X, y_meat)

    # Generate future years for prediction
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)
    
    # Make predictions
    pop_predictions = model_pop.predict(future_X)
    meat_predictions = model_meat.predict(future_X)
    
    # Calculate metrics
    metrics_pop = calculate_metrics(y_pop, model_pop.predict(X))
    metrics_meat = calculate_metrics(y_meat, model_meat.predict(X))
    
    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat

def train_and_predict_xgboost(df, country, country_column_name, population_column_name, year_column_name, meat_category, future_years=7):
    # Filter data for specific country
    #country_data = df[df[country_column_name] == country].copy()
    
    country_data, X, y_pop = filter_and_prepare_training_data(df,country_column_name,country,year_column_name,population_column_name)
    country_data, X, y_meat = filter_and_prepare_training_data(df,country_column_name,country,year_column_name,meat_category)

    X_train_pop, X_test_pop, y_pop_train, y_pop_test = split_into_training_and_test(X, y_pop)
    X_train_meat, X_test_meat, y_meat_train, y_meat_test = split_into_training_and_test(X, y_meat)
        
    # Initialize and train Gradient Boosting model
    gb_model_pop = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    gb_model_meat = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    #st.write("Training Evaluation Results for GradientBoostingRegressor:")
    #st.write("Population:")
    metrics_train, metrics_test = evaluate_model(gb_model_pop,X_train_pop,X_test_pop,y_pop_train,y_pop_test)
    st.session_state['evaluate_models']["gradientboost_pop_train"] = metrics_train
    st.session_state['evaluate_models']["gradientboost_pop_test"] = metrics_test
    #st.write("Meat:")
    metrics_train, metrics_test  = evaluate_model(gb_model_meat,X_train_meat,X_test_meat,y_meat_train,y_meat_test)
    st.session_state['evaluate_models']["gradientboost_meat_train"] = metrics_train
    st.session_state['evaluate_models']["gradientboost_meat_test"] = metrics_test

    # Train model
    gb_model_pop.fit(X, y_pop)
    gb_model_meat.fit(X, y_meat)

    # Generate future years for prediction
    future_X = np.array(range(year_to_start_prediction, year_to_start_prediction + future_years)).reshape(-1, 1)
    
    # Make predictions
    pop_predictions = gb_model_pop.predict(future_X)
    meat_predictions = gb_model_meat.predict(future_X)
    
    # Calculate metrics
    metrics_pop = calculate_metrics(y_pop, gb_model_pop.predict(X))
    metrics_meat = calculate_metrics(y_meat, gb_model_meat.predict(X))
    
    return future_X.flatten(), pop_predictions, meat_predictions, metrics_pop, metrics_meat

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    
    #st.write("Train vs. Test")
    metrics_train = calculate_metrics(y_train, y_train_pred)
    
    #print_metrics(metrics)
    #st.write("Test vs. Train")
    metrics_test = calculate_metrics(y_test, y_test_pred)
    #print_metrics(metrics)
    
    return metrics_train, metrics_test

def calculate_metrics(y_values, prediction):
    metrics = {}
    mse_val = mean_squared_error(y_values, prediction)
    metrics["mse"] = mse_val
    metrics["rmse"] = np.sqrt(mse_val)
    metrics["mae"] = mean_absolute_error(y_values,prediction)
    metrics["r2"] = r2_score(y_values, prediction)
    return metrics
    

def plot_predictions(df, country, future_years, pop_pred, meat_pred, country_column_name, year_column_name, population_column_name, meat_column_name, meat_category, model_name):
    country_data = df[df[country_column_name] == country]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Population plot
    ax1.plot(country_data[year_column_name], country_data[population_column_name], 'b-', label='Historical')
    ax1.plot(range(year_to_start_prediction, year_to_start_prediction + future_years), pop_pred, 'r--', label='Predicted ' + model_name)
    ax1.set_title(f"{country} - Population Projection")
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population (millions)')
    ax1.legend()
    ax1.grid(True)
    
    # Meat consumption plot
    ax2.plot(country_data[year_column_name], country_data[meat_column_name], 'g-', label='Historical')
    ax2.plot(range(year_to_start_prediction, year_to_start_prediction + future_years), meat_pred, 'r--', label='Predicted ' + model_name)
    ax2.set_title(f'{country} -{meat_category} Consumption Projection')
    ax2.set_xlabel('Year')
    ax2.set_ylabel(f"{meat_category} Consumption (kg per capita)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(plt)
    

def filter_countries_by_names(data_frame, countries_list, country_column_name):
    filtered_df = data_frame[data_frame[country_column_name].isin(countries_list)]
    return filtered_df

def print_prediction(years_list,pop_predicted_df, meat_predicted_df, metrics_pop, metrics_meat, model_name):
    
    st.write(f"Population Predictions {model_name} (millions):")
    for year, pop in zip(years_list, pop_predicted_df):
        st.write(f"{year}: {pop:.2f}")
        
    st.write(f"Meat Consumption Predictions {model_name} (kg per capita):")
    for year, meat in zip(years_list, meat_predicted_df):
        st.write(f"{year}: {meat:.2f}")
        
    st.write("Model scores:")
        
    st.write("Population:")
    st.write(print_metrics(metrics_pop))
    st.write("Meat Consumption:")
    st.write(print_metrics(metrics_meat))


def print_metrics(metrics_info):
    st.write("Model Performance Metrics:")
    st.write(f"MSE: {metrics_info['mse']:.4f}")
    st.write(f"RMSE: {metrics_info['rmse']:.4f}")
    st.write(f"MAE: {metrics_info['mae']:.4f}")
    st.write(f"R2 Score: {metrics_info['r2']:.4f}")
    
def main():
    future_years = int(st.session_state['future'])

    # Year to start the prediction
    year_to_start_prediction = int(st.session_state['year'])

    st.title("Population and Meat Consumption Predictor")
    st.write("This application predicts population and meat consumption using Linear Regression, Boosting, and Decision Tree models.")
    # Load or create data
    df = get_data() #create_sample_data()
    
    countries_to_pull = ['United States', 'Brazil', 'Australia', 'Spain', 'Zimbabwe', 'Maldives', 'Japan']
    filtered_population_df = filter_countries_by_names(df, countries_to_pull, 'Entity')
    
    # List of countries to analyze
    countries = filtered_population_df['Entity'].unique()
    counter = 0
    tab_country = st.tabs(countries.tolist())
    
    for country in countries:
        
        with tab_country[counter]:
            st.write(f"Predictions for {country}:")

            years, pop_pred, meat_pred, metrics_pop, metrics_meat = train_and_predict_lr(filtered_population_df, country, 'Entity', 'population_historical', 'Year', 'Poultry', future_years)
            gb_years, gb_pop_pred, gb_meat_pred, gb_metrics_pop, gb_metrics_meat = train_and_predict_xgboost(filtered_population_df, country, 'Entity', 'population_historical', 'Year', 'Poultry', future_years)
            dt_years, dt_pop_pred, dt_meat_pred, dt_metrics_pop, dt_metrics_meat = train_and_predict_dt(filtered_population_df, country, 'Entity', 'population_historical', 'Year', 'Poultry', future_years)


            col1, col2, col3 = st.columns(3)

           
            with col1:
                st.header("Linear Regression")
                print_prediction(years,pop_pred, meat_pred, metrics_pop, metrics_meat,"LinearRegression")
                st.write("Model Performance")
                lr_perf_col1, lr_perf_col2  = st.columns(2)
                with lr_perf_col1:
                    st.write("Population Training Performance")
                    metrics = st.session_state['evaluate_models']["linearregression_pop_train"]
                    print_metrics(metrics)
                    
                with lr_perf_col2:
                    st.write("Meat Training Performance")
                    metrics = st.session_state['evaluate_models']["linearregression_meat_train"]
                    print_metrics(metrics)

                plot_predictions(filtered_population_df, country, future_years, pop_pred, meat_pred, 'Entity', 'Year', 'population_historical', st.session_state['meat_option'],st.session_state['meat_option'],'LinearRegression')

            with col2:
                st.header("Gradient Boosting")
                print_prediction(gb_years,gb_pop_pred, gb_meat_pred, gb_metrics_pop, gb_metrics_meat,"GradientBoosting")
                st.write("Model Performance")
                gb_perf_col1, gb_perf_col2  = st.columns(2)
                with gb_perf_col1:
                    st.write("Population Training Performance")
                    metrics = st.session_state['evaluate_models']["gradientboost_pop_train"]
                    print_metrics(metrics)
                    
                with gb_perf_col2:
                    st.write("Meat Consumption Training Performance")
                    metrics = st.session_state['evaluate_models']["gradientboost_meat_train"]
                    print_metrics(metrics)

                plot_predictions(filtered_population_df, country, future_years, gb_pop_pred, gb_meat_pred, 'Entity', 'Year', 'population_historical', st.session_state['meat_option'],st.session_state['meat_option'],'GradientBoost')
            
            with col3:
                st.header("Decision Tree")
                print_prediction(dt_years,gb_pop_pred, dt_meat_pred, dt_metrics_pop, dt_metrics_meat,"DecisionTree")
                st.write("Model Performance")
                dt_perf_col1, dt_perf_col2  = st.columns(2)
                with dt_perf_col1:
                    st.write("Population Training Performance")
                    metrics = st.session_state['evaluate_models']["decisiontree_pop_train"]
                    print_metrics(metrics)
                    
                with dt_perf_col2:
                    st.write("Meat Consumption Training Performance")
                    metrics = st.session_state['evaluate_models']["decisiontree_meat_train"]
                    print_metrics(metrics)
                plot_predictions(filtered_population_df, country, future_years, dt_pop_pred, dt_meat_pred, 'Entity', 'Year', 'population_historical', st.session_state['meat_option'],st.session_state['meat_option'],'DecisionTree')

        counter = counter + 1
        
#if __name__ == "__main__":

df = get_data()
column_names = list(df.columns.values)

meat_categories = column_names[4:len(column_names)] #column_names.remove(["Entity","Code","Year","population_historical"])
countries_list = df['Entity'].unique()
max_year = df['Year'].unique().max()

    
st.session_state['future'] = st.text_input(
    "Enter number of future years to predict 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled
)

st.write(f"Year cannot be before {max_year}")        
st.session_state['year'] = st.text_input(
    "Enter Year to start prediction 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled
)


st.session_state['meat_option'] = st.selectbox(
    "How would you like to be contacted?",
    (meat_categories),
)

#st.session_state['countries'] = st.selectbox(
#    "How would you like to be contacted?",
#    (meat_categories),
#)


with st.form("Predictions"):
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        main()

