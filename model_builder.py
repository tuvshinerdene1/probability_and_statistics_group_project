from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np

def train_all_models(df):
    """
    Бүх Регрессийн загваруудыг сургаж, үр дүнг харьцуулах функц.
    """
    # X (Features) болон y (Target) хувьсагчдыг салгах
    X = df.drop('Target_GPA', axis=1)  # Зорилтот баганаас бусад нь оролт
    y = df['Target_GPA']               # Таамаглах гэж буй утга
    
    # Өгөгдлийг сургалтын (80%) болон тестийн (20%) хэсэгт хуваах
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Туршиж үзэх загваруудын жагсаалт
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
    
    # Загвар бүрээр давталт хийж сургах
    for name, model in models.items():
        try:
            # Загварыг сургах
            model.fit(X_train, y_train)
            
            # Тестийн өгөгдөл дээр таамаглал хийх
            y_pred = model.predict(X_test)
            
            # Үнэлгээний үзүүлэлтүүдийг тооцох (R2 болон MAE)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Үр дүнг толь бичигт хадгалах
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'y_pred': y_pred,
                'y_test': y_test
            }
        except Exception as e:
            # Хэрэв аль нэг загвар дээр алдаа гарвал тэмдэглэж авах
            results[name] = {'error': str(e)}
    
    return results, X_test, y_test

def train_model(df, model_name='Linear Regression'):
    """
    Сонгосон нэг Регрессийн загварыг сургах функц.
    """
    # X болон y хувьсагчдыг салгах
    X = df.drop('Target_GPA', axis=1)
    y = df['Target_GPA']
    
    # Өгөгдлийг хуваах
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Боломжит загваруудын тодорхойлолт
    model_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Хэрэв сонгосон загвар жагсаалтад байхгүй бол анхдагчаар Linear Regression авах
    if model_name not in model_dict:
        model_name = 'Linear Regression'
    
    # Сонгосон загварыг авах
    model = model_dict[model_name]
    
    # Загварыг сургах
    model.fit(X_train, y_train)
    
    # Таамаглал хийх
    y_pred = model.predict(X_test)
    
    # Үнэлгээ тооцох
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Сургасан загвар, үнэлгээ, тестийн өгөгдөл, таамаглалыг буцаах
    return model, r2, mae, X_test, y_test, y_pred