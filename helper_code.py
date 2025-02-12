# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_country_names(country_column,df):
    return df[country_column].unique()

def get_numeric_mapping(list_of_countries):
    i = 0
    mapped_values = {}
    for country in list_of_countries:
        mapped_values[country] = i
        i = i + 1
    
    return mapped_values

def slice_by_country(country_names, country_column_name, data_frame):
    sliced_df = data_frame[data_frame[country_column_name].isin(country_names)]
    return sliced_df

def map_countries_to_numeric(countries_mapping,country_column_name,numeric_column_name,df):
    numeric_mapping_vals = []
    # loop through the rows using iterrows()
    for index, row in df.iterrows():
        if row[country_column_name] in countries_mapping:
            numeric_mapping_vals += [countries_mapping[row[country_column_name]]]
            print(numeric_mapping_vals) 
    df[numeric_column_name] = numeric_mapping_vals
    return df

iso_country_code_dict = {}

with open("iso_country_codes.txt") as file:
    for item in file:
        item.replace("\n","")
        code_ref = item.split("  ")
        iso_country_code_dict[code_ref[0]] = code_ref[1].replace("\n","")
        

print(iso_country_code_dict)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample historical data - you can replace this with actual data
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024] * 3,
    'Country': ['USA'] * 10 + ['China'] * 10 + ['Brazil'] * 10,
    'Population': [
        # USA population (millions)
        321, 323, 325, 327, 329, 331, 332, 334, 336, 338,
        # China population (millions)
        1397, 1403, 1409, 1415, 1420, 1424, 1426, 1428, 1430, 1432,
        # Brazil population (millions)
        204, 206, 208, 210, 211, 213, 214, 215, 216, 217
    ],
    'Poultry_Consumption': [
        # USA consumption (kg per capita)
        47.6, 48.2, 48.8, 49.4, 50.0, 50.6, 51.2, 51.8, 52.4, 53.0,
        # China consumption (kg per capita)
        12.8, 13.2, 13.6, 14.0, 14.4, 14.8, 15.2, 15.6, 16.0, 16.4,
        # Brazil consumption (kg per capita)
        39.5, 40.1, 40.7, 41.3, 41.9, 42.5, 43.1, 43.7, 44.3, 44.9
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to make predictions for a country
def predict_country(country_data, future_years):
    years = country_data['Year'].values.reshape(-1, 1)
    
    # Population prediction
    pop_model = LinearRegression()
    pop_model.fit(years, country_data['Population'])
    future_population = pop_model.predict(future_years)
    
    # Consumption prediction
    cons_model = LinearRegression()
    cons_model.fit(years, country_data['Poultry_Consumption'])
    future_consumption = cons_model.predict(future_years)
    
    return future_population, future_consumption

# Generate future years
future_years = np.array(range(2025, 2032)).reshape(-1, 1)

# Create plots for each country
countries = df['Country'].unique()
fig, axes = plt.subplots(len(countries), 2, figsize=(15, 5*len(countries)))
plt.subplots_adjust(hspace=0.4)

# Store predictions for printing
predictions = []

for idx, country in enumerate(countries):
    country_data = df[df['Country'] == country]
    future_pop, future_cons = predict_country(country_data, future_years)
    
    # Store predictions
    for year, pop, cons in zip(future_years.flatten(), future_pop, future_cons):
        predictions.append({
            'Country': country,
            'Year': int(year),
            'Population': round(pop, 1),
            'Poultry_Consumption': round(cons, 1)
        })
    
    # Population plot
    axes[idx, 0].scatter(country_data['Year'], country_data['Population'], 
                        color='blue', label='Historical')
    axes[idx, 0].scatter(future_years, future_pop, color='red', label='Predicted')
    axes[idx, 0].set_title(f'{country} - Population Projection')
    axes[idx, 0].set_xlabel('Year')
    axes[idx, 0].set_ylabel('Population (millions)')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True)
    
    # Consumption plot
    axes[idx, 1].scatter(country_data['Year'], country_data['Poultry_Consumption'], 
                        color='blue', label='Historical')
    axes[idx, 1].scatter(future_years, future_cons, color='red', label='Predicted')
    axes[idx, 1].set_title(f'{country} - Poultry Consumption Projection')
    axes[idx, 1].set_xlabel('Year')
    axes[idx, 1].set_ylabel('Poultry Consumption (kg per capita)')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True)

plt.show()

# Print predictions in a formatted table
predictions_df = pd.DataFrame(predictions)
print("\nPredictions for 2025-2031:")
print("\nCountry | Year | Population (millions) | Poultry Consumption (kg per capita)")
print("-" * 75)
for _, row in predictions_df.sort_values(['Country', 'Year']).iterrows():
    print(f"{row['Country']:<7} | {row['Year']} | {row['Population']:>18.1f} | {row['Poultry_Consumption']:>31.1f}")

# Plot the results
plt.figure(figsize=(12, 6))

# Population subplot
plt.subplot(1, 2, 1)
plt.scatter(years, population, color='blue', label='Historical')
plt.scatter(future_years, future_population, color='red', label='Predicted')
plt.plot(years, pop_model.predict(years), color='blue', linestyle='--')
plt.plot(future_years, future_population, color='red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Population (millions)')
plt.title('Population Projection')
plt.legend()
plt.grid(True)

# Consumption subplot
plt.subplot(1, 2, 2)
plt.scatter(years, poultry_consumption, color='blue', label='Historical')
plt.scatter(future_years, future_consumption, color='red', label='Predicted')
plt.plot(years, consumption_model.predict(years), color='blue', linestyle='--')
plt.plot(future_years, future_consumption, color='red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Poultry Consumption (kg per capita)')
plt.title('Poultry Consumption Projection')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Print predictions
print("\nPredictions for 2025-2031:")
print("\nYear | Population (millions) | Poultry Consumption (kg per capita)")
print("-" * 60)
for year, pop, cons in zip(future_years, future_population, future_consumption):
    print(f"{int(year)} | {pop:.1f} | {cons:.1f}")

plt.show()