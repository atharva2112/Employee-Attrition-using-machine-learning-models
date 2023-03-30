#%%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load the dataset and preprocess it
data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
# Preprocess the dataset and create relevant visualizations
# Data exploration and preprocessing
# Drop irrelevant columns
data = data.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1)

# Convert categorical features to numerical using LabelEncoder
cat_columns = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_columns:
    data[col] = le.fit_transform(data[col])

# Correlation heatmap visualization
plt.figure(figsize=(16, 12))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Split data into features and target
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with hyperparameter tuning using GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 4, 6]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Model evaluation
y_pred = best_estimator.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = best_estimator.named_steps['classifier'].feature_importances_
feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': importances})
print("Feature Importances:\n", feature_importances.sort_values(by='importance', ascending=False))

# Feature importance visualization
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.sort_values(by='importance', ascending=False))
plt.title("Feature Importance")
plt.show()

external_stylesheets = [
    {
        "href": "https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css",
        "rel": "stylesheet",
        "integrity": "sha384-pzjw8f+ua7Kw1TIq0v8FqFjcJ6pajs/rfdfs3SO+kD4Ck5BdPtF+to8xMkQcJszb",
        "crossorigin": "anonymous",
    }
]

app = dash.Dash("__name__", external_stylesheets=external_stylesheets)


df = data

# Pie chart for Attrition status
fig_pie = px.pie(df, names='Attrition', title='Attrition Status Distribution')

# Bar chart for Department-wise Attrition
fig_bar = px.histogram(df, x='Department', color='Attrition', barmode='group', title='Department-wise Attrition Distribution')

# Heatmap for Correlation between features
le = LabelEncoder()
temp_df = df.apply(le.fit_transform)
corr = temp_df.corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr, x=corr.columns, y=corr.columns))

app.layout = html.Div([
    dcc.Dropdown(
        id='visualization-selector',
        options=[
            {'label': 'Attrition Status', 'value': 'attrition_status'},
            {'label': 'Department-wise Attrition', 'value': 'department_attrition'},
            {'label': 'Feature Correlation', 'value': 'feature_correlation'}
        ],
        value='attrition_status'
    ),
    html.Div(id='visualization-container')
])


@app.callback(
    Output(component_id='visualization-container', component_property='children'),
    [Input(component_id='visualization-selector', component_property='value')]
)
def update_layout(vis_type):
    if vis_type == 'attrition_status':
        return html.Div([
            html.H1('Attrition Status Distribution'),
            dcc.Graph(figure=fig_pie)
        ])
    elif vis_type == 'department_attrition':
        return html.Div([
            html.H1('Department-wise Attrition Distribution'),
            dcc.Graph(figure=fig_bar)
        ])
    elif vis_type == "feature_correlation":
        return html.Div([
            html.H1('Correlation between Features'),
            dcc.Graph(figure=fig_heatmap)
        ])

app.run_server(
    port=8024,
    host='0.0.0.0'
)

