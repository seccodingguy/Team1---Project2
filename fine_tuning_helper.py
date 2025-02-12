# 1. Feature Engineering
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create year-based features
df['Decade'] = (df['Year'] // 10) * 10
df['Year_Squared'] = df['Year'] ** 2

# Create interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = ['Year', 'Poultry', 'Beef', 'Sheep_and_goat', 'Other_meats', 'Fish_and_seafood'] + [f'Interaction_{i}' for i in range(X_poly.shape[1] - X.shape[1])]

# 2. Feature Selection using different methods
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE

# Using SelectKBest
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_train_scaled, y_train)

# Using RFE
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X_train_scaled, y_train)

# 3. Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# 4. Try Different Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': XGBRegressor(random_state=42),
    'Lasso': LassoCV(random_state=42)
}

# Compare models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2

# Print results
print("\nModel Comparison (R2 Scores):")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# 5. Ensemble Methods
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Create ensemble
estimators = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42))
]

# Voting Regressor
voting_reg = VotingRegressor(estimators=estimators)
voting_reg.fit(X_train_scaled, y_train)
voting_pred = voting_reg.predict(X_test_scaled)
voting_r2 = r2_score(y_test, voting_pred)

# Stacking Regressor
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LassoCV()
)
stacking_reg.fit(X_train_scaled, y_train)
stacking_pred = stacking_reg.predict(X_test_scaled)
stacking_r2 = r2_score(y_test, stacking_pred)

print("\nEnsemble Methods R2 Scores:")
print(f"Voting Regressor: {voting_r2:.4f}")
print(f"Stacking Regressor: {stacking_r2:.4f}")

# 6. Cross-validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    best_model, 
    X_train_scaled, 
    y_train, 
    cv=5, 
    scoring='r2'
)

print("\nCross-validation scores:")
print(f"Mean R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. Visualize Results
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.ylabel('R2 Score')
plt.tight_layout()
plt.show()

# Plot learning curves
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

plot_learning_curve(best_model, "Learning Curve", X_train_scaled, y_train)
plt.show()