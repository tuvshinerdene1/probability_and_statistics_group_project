import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# MEMBER 2: DATA ENGINEER (Preprocessing)
# ==========================================
def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # 1. Handle Missing Values (Drop them for simplicity)
        df = df.dropna()
        
        # 2. ENCODING: Convert Text to Numbers
        # Example: "Motivation" (High/Low) -> 1/0
        # We use a LabelEncoder for every text column
        encoders = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le # Save encoder to reverse it later if needed

        # 3. CRITICAL FOR NAIVE BAYES: BINNING
        # We need to turn the Target (e.g., 'Grade') into Categories
        # Let's assume the column is called 'GPA' or 'Score'. 
        # CHANGE 'GPA' TO YOUR ACTUAL COLUMN NAME
        target_col_name = 'GPA' 
        
        if target_col_name in df.columns:
            # Logic: If GPA >= 3.0 then 1 (Good), else 0 (Bad)
            df['Target_Class'] = df[target_col_name].apply(lambda x: 1 if x >= 3.0 else 0)
            # Drop the original number column, keep the class
            df = df.drop(columns=[target_col_name])
            
        return df, encoders
    return None, None

# ==========================================
# MEMBER 3: MODEL MAKER (Naive Bayes)
# ==========================================
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

# ==========================================
# MEMBER 4: VISUALIZER (Charts)
# ==========================================
def plot_charts(cm, accuracy):
    st.subheader("📊 Model Performance")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix (True vs Predicted)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# ==========================================
# MEMBER 5: PREDICTOR (User Input Logic)
# ==========================================
def predict_user_input(model, feature_names):
    st.sidebar.header("📝 Student Profile")
    user_data = {}
    
    # Dynamically create sliders for every feature
    for col in feature_names:
        # Simple slider for everything (0 to 10 scale for simplicity)
        # You can customize this if you know specific column ranges
        user_data[col] = st.sidebar.slider(f"{col}", 0, 10, 5)
        
    if st.sidebar.button("Predict Performance"):
        # Convert dict to DataFrame
        input_df = pd.DataFrame([user_data])
        prediction = model.predict(input_df)[0]
        
        st.divider()
        st.subheader("🤖 AI Prediction")
        
        if prediction == 1:
            st.success("🌟 Result: HIGH PERFORMER (GPA > 3.0)")
            st.balloons()
        else:
            st.error("⚠️ Result: NEEDS IMPROVEMENT (GPA < 3.0)")
            st.info("Tip: Try increasing study time or motivation level!")

# ==========================================
# MEMBER 1: ARCHITECT (Main App)
# ==========================================
def main():
    st.set_page_config(page_title="Naive Bayes Student Predictor", layout="wide")
    st.title("🎓 Student Performance Predictor (Naive Bayes)")
    st.markdown("Predicting if a student will satisfy **GPA > 3.0** based on habits.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Student Data CSV", type=['csv'])
    
    if uploaded_file:
        # 1. Member 2 works here
        df, encoders = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.write("Data Loaded Successfully!", df.head())
            
            # 2. Member 3 works here
            model, accuracy, cm, X_test, y_test = train_model(df)
            
            # 3. Member 4 works here
            plot_charts(cm, accuracy)
            
            # 4. Member 5 works here
            # We pass the column names so Member 5 knows what sliders to make
            feature_names = df.drop('Target_Class', axis=1).columns
            predict_user_input(model, feature_names)

if __name__ == "__main__":
    main()