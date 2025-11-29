from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(df):
    # X = All columns except the Target
    X = df.drop('Target_Class', axis=1)
    # y = The Target (0 or 1)
    y = df['Target_Class']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Test it
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cm, X_test, y_test