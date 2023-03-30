# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
#%%
# Load dataset
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
#%%
# Data exploration and preprocessing
# Drop irrelevant columns
data = data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)
#%%
# Convert categorical features to numerical using LabelEncoder
cat_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_columns:
    data[col] = le.fit_transform(data[col])
#%%
# Correlation heatmap visualization
plt.figure(figsize=(16, 12))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
#%%
# Split data into features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']
#%%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Model training with hyperparameter tuning using GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
#%%
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 4, 6]
}
#%%
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
#%%
# Model evaluation
y_pred = best_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#%%
# Feature importance
importances = best_estimator.named_steps['classifier'].feature_importances_
feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': importances})
print("Feature Importances:\n", feature_importances.sort_values(by='importance', ascending=False))
#%%
# Feature importance visualization
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.sort_values(by='importance', ascending=False))
plt.title("Feature Importance")
plt.show()
