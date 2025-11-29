import streamlit as st
# Import functions from other files
from data_loader import load_and_clean_data
from model_builder import train_model
from visualizer import plot_charts
from predictor import predict_user_input

def main():
    st.set_page_config(page_title="Naive Bayes Student Predictor", layout="wide")
    st.title("🎓 Student Performance Predictor (Naive Bayes)")
    st.markdown("Predicting if a student will satisfy **GPA > 3.0** based on habits.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Student Data CSV", type=['csv'])
    
    if uploaded_file:
        # 1. Member 2's Code
        df, encoders = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.write("Data Loaded Successfully!", df.head())
            
            # 2. Member 3's Code
            model, accuracy, cm, X_test, y_test = train_model(df)
            
            # 3. Member 4's Code
            plot_charts(cm, accuracy)
            
            # 4. Member 5's Code
            feature_names = df.drop('Target_Class', axis=1).columns
            predict_user_input(model, feature_names)

if __name__ == "__main__":
    main()