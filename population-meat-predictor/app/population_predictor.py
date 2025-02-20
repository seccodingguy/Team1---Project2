# app/models/population_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.linear_model import ElasticNetCV
from scipy import stats
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PopulationPredictor:
    def __init__(self, model, imputation_strategy='mean'):
        """
        Initialize the predictor with any sklearn model and imputation strategy.
        
        Parameters:
        -----------
        model : sklearn estimator
            Any sklearn model (e.g., LinearRegression, RandomForestRegressor, etc.)
        imputation_strategy : str, default='mean'
            Strategy for imputation: 'mean', 'median', 'most_frequent', or 'constant'
        """
        # Create the imputer pipeline
        self.imputation_pipeline = FeatureUnion(
            transformer_list=[
                ('features', Pipeline([
                    ('imputer', SimpleImputer(strategy=imputation_strategy)),
                    ('scaler', StandardScaler())
                ])),
                ('missing_indicators', MissingIndicator(features='all'))
            ]
        )
        
        # Create the full pipeline
        self.pipeline = Pipeline([
            ('imputer_transformer', self.imputation_pipeline),
            ('model', model)
        ])
        
        self.feature_columns = None
        self.fitted = False
        
    def create_features(self, data):
        """
        Create feature matrix from input data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with datetime index and 'population' column
        """
        try:
            features = pd.DataFrame(index=data.index)
            # Basic time features
            features['year'] = features.index.Year
            features['year_squared'] = features['year'] ** 2
            features['log_year'] = np.log(features['year'] - features['year'].min() + 1)
            
            # Population-based features (if enough data)
            if len(data) > 5:
                features['population'] = data['population_historical']
                features['population_lag1'] = data['population_historical'].shift(1)
                features['population_lag2'] = data['population_historical'].shift(2)
                features['growth_rate'] = (data['population_historical'] / 
                                         data['population_historical'].shift(1)) - 1
            
            self.feature_columns = features.columns
            return features
            
        except Exception as e:
            raise ValueError(f"Error creating features: {str(e)}")

    def create_future_features(self, historical_features, future_dates):
        """
        Create feature matrix for future predictions.
        
        Parameters:
        -----------
        historical_features : pd.DataFrame
            Features from historical data
        future_dates : pd.DatetimeIndex
            Dates for which to create features
        """
        try:
            future_features = pd.DataFrame(index=future_dates)
            
            # Basic time features
            future_features['year'] = future_dates.Year
            future_features['year_squared'] = future_features['year'] ** 2
            future_features['log_year'] = np.log(
                future_features['year'] - historical_features['year'].min() + 1
            )
            
            # Population-based features
            if 'population' in self.feature_columns:
                # Use last known values for lags
                future_features['population'] = np.nan
                future_features['population_lag1'] = historical_features['population'].iloc[-1]
                future_features['population_lag2'] = historical_features['population'].iloc[-2]
                future_features['growth_rate'] = historical_features['growth_rate'].iloc[-1]
            
            return future_features
            
        except Exception as e:
            raise ValueError(f"Error creating future features: {str(e)}")

    def fit(self, historical_data):
        """
        Fit the model to historical data.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical data with datetime index and 'population' column
        """
        try:
            # Create features
            X = self.create_features(historical_data)
            y = historical_data['population_historical']
            
            # Fit pipeline
            self.pipeline.fit(X, y)
            self.fitted = True
            
            return self
            
        except Exception as e:
            raise ValueError(f"Error fitting model: {str(e)}")

    def predict(self, future_periods):
        """
        Make predictions for future periods.
        
        Parameters:
        -----------
        future_periods : int
            Number of future periods to predict
        
        Returns:
        --------
        pd.Series
            Predictions indexed by date
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            # Create future dates
            last_date = pd.to_datetime(self.feature_columns.index[-1])
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=future_periods,
                freq='Y'
            )
            
            # Create future features
            future_features = self.create_future_features(
                self.feature_columns, 
                future_dates
            )
            
            # Make predictions
            predictions = self.pipeline.predict(future_features)
            
            return pd.Series(predictions, index=future_dates)
            
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")

    def evaluate(self, X, y, cv=5):
        """
        Evaluate the model using cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        cv : int, default=5
            Number of cross-validation folds
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            
            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv)
            
            # Calculate scores
            scores = cross_val_score(
                self.pipeline,
                X,
                y,
                cv=tscv,
                scoring=['neg_mean_squared_error', 'r2']
            )
            
            return {
                'rmse': np.sqrt(-scores['neg_mean_squared_error'].mean()),
                'rmse_std': np.sqrt(-scores['neg_mean_squared_error'].std()),
                'r2': scores['r2'].mean(),
                'r2_std': scores['r2'].std()
            }
            
        except Exception as e:
            raise ValueError(f"Error evaluating model: {str(e)}")