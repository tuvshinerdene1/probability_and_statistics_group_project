from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def train_all_models(df):
    """
    Бүх боломжит моделиудыг нэгэн зэрэг сургаж, үр дүнг нь харьцуулах функц.
    """
    # Зорилтот хувьсагч (Target_Exam_Score) болон оролтын шинж чанаруудыг (X) салгах
    X = df.drop('Target_Exam_Score', axis=1)
    y = df['Target_Exam_Score']

    # Өгөгдлийг сургалтын (train) болон тестийн (test) багц болгон хуваах
    # test_size=0.2 -> Өгөгдлийн 20%-ийг шалгалтад, 80%-ийг сургалтад ашиглана
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Турших гэж буй бүх алгоритмуудын жагсаалт
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
    # Модель тус бүрээр давталт хийж сургах
    for name, model in models.items():
        try:
            # Моделийг сургах
            model.fit(X_train, y_train)
            # Тестийн өгөгдөл дээр таамаглал дэвшүүлэх
            y_pred = model.predict(X_test)
            
            # Үнэлгээний хэмжүүрүүдийг тооцоолох
            # R2: Модель өгөгдлийн хэдэн хувийг тайлбарлаж чадаж байгааг харуулна (1-рүү дөхөх тусам сайн)
            r2 = r2_score(y_test, y_pred)
            # MAE: Таамаглал бодит утгаас дунджаар хэр зөрж байгааг харуулна (Бага байх тусам сайн)
            mae = mean_absolute_error(y_test, y_pred)

            # Үр дүнг хадгалах
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mae': mae,
                'y_pred': y_pred,
                'y_test': y_test
            }
        except Exception as e:
            # Алдаа гарвал програмыг зогсоохгүйгээр алдааг тэмдэглэж авах
            results[name] = {'error': str(e)}

    return results, X_test, y_test

def train_model(df, model_name='Linear Regression'):
    """
    Хэрэглэгчийн сонгосон нэг моделийг сургах функц.
    """
    X = df.drop('Target_Exam_Score', axis=1)
    y = df['Target_Exam_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Моделиудын толь бичиг (Dictionary)
    model_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Сонгосон нэрээр тохирох моделийг авах, олдохгүй бол Linear Regression-ийг default-аар авах
    model = model_dict.get(model_name, LinearRegression())
    
    # Сургах
    model.fit(X_train, y_train)
    # Таамаглах
    y_pred = model.predict(X_test)
    # Үнэлэх
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, r2, mae, X_test, y_test, y_pred