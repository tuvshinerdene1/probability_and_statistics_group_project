# model_builder.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_all_models(df):
    X = df.drop('Target_Exam_Score', axis=1)
    y = df['Target_Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'y_pred': y_pred,
                'y_test': y_test
            }
        except Exception as e:
            results[name] = {'error': str(e)}

    return results, X_test, y_test

def train_model(df, model_name='Linear Regression'):
    X = df.drop('Target_Exam_Score', axis=1)
    y = df['Target_Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    model = model_dict.get(model_name, LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, r2, mae, X_test, y_test, y_pred