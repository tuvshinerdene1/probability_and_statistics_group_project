import streamlit as st
# Import functions from other files
from data_loader import load_and_clean_data
from model_builder import train_model, train_all_models
from visualizer import plot_charts, compare_all_models
from predictor import predict_user_input

def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    st.title("🎓 Student Performance Predictor")
    st.markdown("Predicting if a student will satisfy **GPA > 3.0** based on habits.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Student Data CSV", type=['csv'])
    
    if uploaded_file:
        # 1. Member 2's Code
        df, encoders = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.write("Data Loaded Successfully!", df.head())
            
            # Model selection and comparison
            st.divider()
            st.subheader("🤖 Model Selection")
            
            mode = st.radio(
                "Choose mode:",
                ["Compare All Models", "Use Single Model"],
                horizontal=True
            )
            
            if mode == "Compare All Models":
                # 2. Member 3's Code - Train all models
                with st.spinner("Training all models... This may take a moment."):
                    results, X_test, y_test = train_all_models(df)
                
                # 3. Member 4's Code - Compare all models
                compare_all_models(results)
                
                # Show individual confusion matrices
                st.divider()
                st.subheader("📊 Individual Model Confusion Matrices")
                
                selected_model = st.selectbox(
                    "Select model to view details:",
                    [name for name, result in results.items() if 'error' not in result]
                )
                
                if selected_model:
                    result = results[selected_model]
                    plot_charts(result['confusion_matrix'], result['accuracy'], selected_model)
                    
                    # 4. Member 5's Code - Use selected model for prediction
                    feature_names = df.drop('Target_Class', axis=1).columns
                    predict_user_input(result['model'], feature_names)
            
            else:
                # Single model mode
                model_options = [
                    'Naive Bayes',
                    'Logistic Regression',
                    'Decision Tree',
                    'Random Forest',
                    'SVM',
                    'K-Nearest Neighbors',
                    'Gradient Boosting'
                ]
                
                selected_model = st.selectbox("Select a model:", model_options)
                
                # 2. Member 3's Code
                with st.spinner(f"Training {selected_model}..."):
                    model, accuracy, cm, X_test, y_test = train_model(df, selected_model)
                
                # 3. Member 4's Code
                plot_charts(cm, accuracy, selected_model)
                
                # 4. Member 5's Code
                feature_names = df.drop('Target_Class', axis=1).columns
                predict_user_input(model, feature_names)

if __name__ == "__main__":
    main()