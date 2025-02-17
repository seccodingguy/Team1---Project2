# Meat Consumption Predictor - README

This document provides instructions on how to set up, configure, and run the Meat Consumption Predictor application.

## Overview

The Meat Consumption Predictor is a Flask-based web application that predicts future meat consumption and population based on historical data. It uses machine learning models (Linear Regression, Lasso Regression, and Ridge Regression) to generate predictions.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.x:**  Make sure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/).
*   **pip:**  Python package installer (usually included with Python installations).
*   **Redis:**  A data structure store, used for session management.  Download and install from [redis.io](https://redis.io/).
*   **Git (Optional):** If you want to clone the repository from a remote source.

## Setup and Installation

Follow these steps to set up the application:

1.  **Clone the Repository (Optional):**

    If you have the code in a Git repository, clone it to your local machine:

    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a Virtual Environment (Recommended):**

    It's best practice to create a virtual environment to isolate the project dependencies:

    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Virtual Environment:**

    *   **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

    *   **On macOS and Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**

    Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    If a `requirements.txt` file is not provided, you'll need to install the dependencies manually. Based on the code, these likely include:

    ```bash
    pip install Flask pandas scikit-learn flask_session redis
    ```

5.  **Download the Data:**

    The application uses a CSV file for population data.  The application attempts to download it from `https://ourworldindata.org/grapher/population.csv?country=USA~BRA~AUS~ESP~ZWE~MDV~JPN&v=1&csvType=full&useColumnShortNames=true`.  If this fails, download the data manually and save it as `population.csv` in the project directory (or adjust the path in `utils.py`).

    The application also requires `Consumptionofmeatpercapita.csv`.  Place this file in a `data` subdirectory.

## Configuration

1.  **Set the Secret Key (Important):**

    The Flask application uses a secret key for session management.  It's crucial to set a strong, random secret key in your production environment.  You can do this by setting the `SECRET_KEY` environment variable.  For development, the `config.py` file provides a default.

    ```bash
    export SECRET_KEY="your-strong-random-key"
    ```

    Replace `"your-strong-random-key"` with a real, randomly generated key.

2.  **Redis Configuration:**

    The application uses Redis for session management. Ensure Redis is running on your local machine (or the specified host and port in `app.py`). The default configuration in `app.py` assumes Redis is running on `127.0.0.1:6379`.  If your Redis instance is configured differently, modify the `app.config['SESSION_REDIS']` setting in `app.py` accordingly.

## Running the Application

1.  **Start the Flask Application:**

    Run the `app.py` file to start the application:

    ```bash
    python app.py
    ```

    This will start the Flask development server.  You should see output indicating the server is running, usually on `http://127.0.0.1:5000/`.

2.  **Access the Application:**

    Open your web browser and go to the address provided by the Flask development server (e.g., `http://127.0.0.1:5000/`).

## Using the Application

1.  **Prediction Form:**

    On the main page, you'll find a form where you can select:

    *   **Country:** The country for which you want to generate predictions.
    *   **Meat Category:** The type of meat to predict consumption for.
    *   **Start Year:** The year to start the prediction from.
    *   **Number of Years to Predict:** The number of future years to predict (between 1 and 10).

2.  **Generate Predictions:**

    Fill out the form and click the "Generate Predictions" button.

3.  **View Results:**

    The application will display the prediction results, including:

    *   **Model Metrics:** Performance metrics for each regression model (Linear Regression, Lasso Regression, Ridge Regression).
    *   **Detailed Predictions:** A table showing the predicted population and meat consumption for each year and model.
    *   **Prediction Charts:**  Line charts visualizing the population and meat consumption predictions for each model.
    *   **Grid Search Results:**  Charts showing the cross-validation scores for different alpha values in the Lasso and Ridge regression models, along with the best parameters found.
    *   **Actual vs Predicted Values:** Scatter plots comparing the actual and predicted values for the test data.

## Troubleshooting

*   **"ModuleNotFoundError: No module named 'flask'":**  Make sure you've activated your virtual environment and installed the dependencies using `pip install -r requirements.txt`.
*   **"Redis::Cannot connect to Redis at 127.0.0.1:6379":**  Ensure Redis is running and accessible on the specified host and port.
*   **"Error: No data received from the server":**  Check the Flask server logs for any errors that occurred during the prediction process.  Verify that the data files are correctly located and accessible.
*   **"Error generating predictions. Please try different parameters.":** This error indicates a problem during the model training or prediction phase. Check the server logs for specific error messages from the scikit-learn library.  Try different combinations of countries, meat categories, and start years.
*   **Application doesn't load CSS/JS:** Ensure the `STATIC_FOLDER` in `config.py` is correctly configured and that the static files are located in the correct directory.

## Deployment (Beyond Local Development)

To deploy this application to a production environment, you'll need to:

1.  **Choose a Web Server:** Select a web server like Gunicorn or uWSGI.
2.  **Configure the Web Server:** Configure the web server to run the Flask application.
3.  **Set up a Reverse Proxy:** Use a reverse proxy like Nginx or Apache to handle incoming requests and forward them to the web server.
4.  **Configure Redis:**  Use a production-ready Redis instance (e.g., a managed Redis service).
5.  **Set Environment Variables:**  Set the `SECRET_KEY` environment variable and any other required configuration variables.

This README provides a basic guide to setting up and running the Meat Consumption Predictor application.  For more advanced configuration and deployment options, consult the Flask documentation and the documentation for the other libraries used in the project.