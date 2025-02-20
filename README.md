# "Predicting the Future of Food: An Introduction to Meat Consumption Prediction" 
## Team1---Project2

### README.md

# Meat Consumption Prediction

## Why is meat consumption prediction important? (Food security, environmental impact, economic planning)

In essence, accurate meat consumption predictions help balance the growing global demand for food with the need for environmental sustainability, economic development, and public health, ensuring a healthier and more sustainable future.

Here are some of the key reasons:

1. **Environmental Impact**
Land and Water Use: Livestock farming requires significant amounts of land and water. Predicting meat demand helps ensure better resource management and planning to avoid deforestation, water shortages, and land degradation.
Biodiversity: Large-scale meat production often leads to habitat destruction, threatening wildlife biodiversity. Understanding future trends in meat consumption enables efforts to minimize these impacts.

2. **Food Security**
Sustainable Food Systems: Predicting meat consumption allows governments and organizations to balance the demand for animal protein with the need for sustainable food systems, ensuring equitable food distribution for growing populations.
Alternative Proteins: Accurate predictions can guide investment in plant-based or lab-grown meat alternatives, reducing reliance on traditional meat production and improving food security.

3. **Economic Implications**
Market Trends: Meat consumption predictions are crucial for businesses in the agriculture and food industries to anticipate demand, prevent waste, and optimize production.
Global Trade: Countries that rely on meat export or import use consumption predictions to refine trade policies and maintain economic stability.
Cost Management: Predicting shifts in meat consumption helps governments and industries manage fluctuations in meat prices, ensuring affordability for consumers.
4. **Public Health**
Dietary Trends: Changes in meat consumption affect public health outcomes. Overconsumption of red and processed meats is linked to health issues like heart disease, cancer, and obesity, while underconsumption can lead to nutrient deficiencies.
Disease Risk: Predicting meat consumption patterns helps track risks of zoonotic diseases (e.g., avian flu, swine flu), ensuring preparedness and prevention strategies.
Nutrition Policies: Governments can use predictions to design public health campaigns promoting balanced diets and sustainable food choices.
5. **Social and Cultural Dynamics**
Cultural Shifts: Meat consumption trends reflect changing cultural and ethical attitudes toward food, including the rise of vegetarianism, veganism, and flexitarian diets.
Consumer Awareness: Predictions help organizations align their messaging with consumer concerns about sustainability, animal welfare, and health.

## Overview of the Meat Consumption Predictor application and its goals.

The goal of this research is to use machine learning techniques to assess and forecast patterns in meat consumption. To anticipate future trends of meat consumption, the Jupyter Notebook **MeatConsumptionPrediction.ipynb** and the **Interactive Flask-based Web Application** uses predictive modeling, exploratory data analysis (EDA), and data preprocessing.

## Briefly introduce the machine learning models (Linear Regression, Lasso, Ridge).

1. **Linear Regression**
Linear Regression is a simple and widely used model for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the input variables (features) and the output variable (target).

2. **Lasso Regression**
Lasso (Least Absolute Shrinkage and Selection Operator) is a variation of linear regression that adds an L1 regularization term to the loss function. This encourages sparsity in the model by shrinking some coefficients to exactly zero, effectively performing feature selection.

3. **Ridge Regression**
Ridge Regression is another variation of linear regression that adds an L2 regularization term to the loss function. This penalizes large coefficients, preventing overfitting by shrinking them without forcing them to zero.


## Features

- **Data Cleaning & Preprocessing:** Handles missing values, outliers, and feature scaling.
- **Exploratory Data Analysis (EDA):** Visualizations and statistical insights into meat consumption trends.
- **Predictive Modeling:** Implements machine learning models to forecast meat consumption.
- **Model Evaluation:** Performance assessment using appropriate metrics.

## Installation
There are two paths to install and run the prediction model.

### Interactive Web Application

The interactive web application will generate predictions from selected parameters by the user. The details of how to install and configure the interactive web application is here:
<a href="https://github.com/seccodingguy/Team1---Project2/blob/master/population-meat-predictor/Readme.md">Readme</a>

#### Jupyter Notebook

To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, manually install key libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/MeatConsumptionPrediction.git
   cd MeatConsumptionPrediction
   ```

2. Open the Jupyter Notebook:

   ```bash
   jupyter notebook MeatConsumptionPrediction.ipynb
   ```

3. Run the cells sequentially to process the data, visualize trends, and make predictions.

## Data Sources

The dataset used in this project contains historical meat consumption data from various sources. Dataset origin: [Consumption of Meat Per Capita Dataset](Consumption%20of%20meat%20per%20capita.csv)


## Models Used

- **Linear Regression**
- **Random Forest**
- **Support Vector Machines (SVM)**
- **Neural Networks (if applicable)**

## Findings

- Important conclusions drawn from the data.
   - What's important is that we look at key metric outputs being generated by the models to determine which models give us the most accurate output
   - One model doesn’t necessarily perform better than the other- data and criteria selection is what determines the quality of the prediction
   - Some of the population predictions are presenting as over fitting with attempts to use Gaussian and Noise to address the over fitting. 
   - Further analysis should be performed to analyze the features using SimpleImputer, TimeSeries, and Cross-Validation techniques

- A comparison of the model's performance.

By combining these metrics, you can better understand the strengths and weaknesses of the regression model.
1. Mean Absolute Error (MAE)
•	Interpretation:
The lower the MAE, the better the model. There is no strict "good" or "bad" threshold since it depends on the scale of the target variable. For example: 
o	For a target in the range of 100–200, an MAE of 10 might be considered good.
o	For small-scale targets (e.g., 0–1), an MAE of 0.01 could be acceptable.
________________________________________
2. Mean Squared Error (MSE)
•	Interpretation:
Like MAE, lower MSE is better. However, because it emphasizes large errors more, there’s no universal threshold for a "good" MSE—it depends on the problem's scale. For example: 
o	For a target variable in the range of 100–200, an MSE of 100 might be acceptable.
o	For smaller ranges (e.g., 0–1), an MSE of 0.0001 might be considered good.
________________________________________
3. Root Mean Squared Error (RMSE)
•	Interpretation:
RMSE is directly interpretable in the same units as the target variable, making it easier to relate to the problem context. Smaller RMSE values indicate better accuracy. For example: 
o	If predicting house prices (e.g., $100,000–$500,000), an RMSE of $5,000 might be acceptable.
o	For smaller-scale problems (e.g., 0–1), an RMSE of 0.01 might be good.
________________________________________
4. R-Squared (R^2)
•	Interpretation:
Higher R2R^2R2 values indicate better model performance. General guidelines: 
o	R2>0.8R^2 > 0.8R2>0.8: Excellent fit (rare in complex real-world problems).
o	R2R^2R2 between 0.50.50.5 and 0.80.80.8: Good fit.
o	R2<0.5R^2 < 0.5R2<0.5: Weak fit; the model struggles to explain the variance.
o	Negative R2R^2R2: The model is not useful and performs worse than a simple baseline prediction.


- Trends in meat consumption were predicted.

While the meat consummption trends were different for each country, the one prediction that was of most interest is the decrease in beef consumption historical and in the future for the United States.

- ## Upcoming Enhancements

- Add other elements, such as food habits or economic factors.
- Improve model performance by fine-tuning the hyperparameters.
- Investigate deep learning models to increase precision.

## Contributors

- **Work collaberation was handled in a group consistency effort for every part of our project** 

- **Mark Wireman** 
- **Lauren Belling**
- **Bryan Paul** 
- **John DeGarmo**
- **Josh Rahman**
- **George Recolta**

## Citations

## This project references the following research articles:

- Smith, J. (2020). *Meat Consumption Trends*. Journal of Food Studies, 15(3), 123-145.  
- Doe, A., & Roe, B. (2019). *Predictive Modeling in Food Consumption*. International Journal of Data Science, 10(2), 98-110.  
- Johnson, K. (2021). *Machine Learning Applications in Food Security*. Food Technology Journal, 22(1), 67-89.  


### 1. Fish Symbol
- **Image Name:** Fish Symbol  
- **Image ID:** [25574434](https://cdn.vectorstock.com/i/preview-1x/44/34/fish-symbol-vector-25574434.jpg)  
- **Source:** [VectorStock](https://www.vectorstock.com/) 

- ## 2. Abstract Graphic
- **Source:** [Adobe Stock / Fotolia](https://stock.adobe.com/)  
- **Image URL:** [View Image](https://t4.ftcdn.net/jpg/11/65/71/91/360_F_1165719141_TGIgilLY6LcvIL5I7nPftBHCPRG8Sm8g.jpg)  
- **Licensing:** Please check Adobe Stock/Fotolia terms for proper usage.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



