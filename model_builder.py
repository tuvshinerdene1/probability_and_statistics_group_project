# model_builder.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_all_models(df):
    """
    Бүх тодорхойлсон моделиудыг сургаж, үр дүнг харьцуулах функц
    """
    # Зорилтот хувьсагчаас (Target_Exam_Score) бусад бүх баганыг оролтын өгөгдөл (X) гэж авна
    X = df.drop('Target_Exam_Score', axis=1)
    # Таамаглах гэж буй гол утга (y)
    y = df['Target_Exam_Score']

    # Өгөгдлийг сургалтын (train) болон тестийн (test) хэсэгт хуваах (20% нь тест, 80% нь сургалт)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Турших гэж буй моделиудын жагсаалт (Dictionary хэлбэрээр)
    models = {
        'Linear Regression': LinearRegression(), # Шугаман регресс
        'Ridge Regression': Ridge(),             # Ridge регресс
        'Decision Tree': DecisionTreeRegressor(random_state=42), # Шийдвэрийн мод
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), # Санамсаргүй ой
        'SVR': SVR(),                            # Тулгуур векторын машин
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5), # K-хамгийн ойрын хөрш
        'Gradient Boosting': GradientBoostingRegressor(random_state=42) # Gradient Boosting
    }

    results = {}
    # Модель тус бүрээр давталт хийж сургах
    for name, model in models.items():
        try:
            # Моделийг сургалтын өгөгдөл дээр сургах
            model.fit(X_train, y_train)
            
            # Тестийн өгөгдөл дээр таамаглал хийх
            y_pred = model.predict(X_test)
            
            # Үнэлгээний үзүүлэлтүүдийг тооцох
            r2 = r2_score(y_test, y_pred) # R2 оноо (Нарийвчлал)
            mae = mean_absolute_error(y_test, y_pred) # Дундаж абсолют алдаа

            # Үр дүнг хадгалах
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'y_pred': y_pred,
                'y_test': y_test
            }
        except Exception as e:
            # Алдаа гарвал бүртгэж авах
            results[name] = {'error': str(e)}

    # Бүх моделийн үр дүн болон тестийн өгөгдлийг буцаах
    return results, X_test, y_test

def train_model(df, model_name='Linear Regression'):
    """
    Сонгосон зөвхөн нэг моделийг сургах функц
    """
    # Оролт (X) болон гаралт (y)-ыг ялгах
    X = df.drop('Target_Exam_Score', axis=1)
    y = df['Target_Exam_Score']

    # Сургалт болон тестийн өгөгдөлд хуваах
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Моделиудын тодорхойлолт
    model_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Хэрэглэгчийн сонгосон нэрээр моделийг авах (Байхгүй бол Linear Regression-г авна)
    model = model_dict.get(model_name, LinearRegression())
    
    # Сургах
    model.fit(X_train, y_train)
    
    # Таамаглах
    y_pred = model.predict(X_test)
    
    # Үнэлэх
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Сургагдсан модель болон үр дүнгүүдийг буцаах
    return model, r2, mae, X_test, y_test, y_pred