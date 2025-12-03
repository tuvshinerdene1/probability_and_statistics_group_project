# app.py
import streamlit as st
from data_loader import load_and_clean_data
from model_builder import train_model, train_all_models
from visualizer import plot_charts, compare_all_models
from predictor import predict_user_input

def main():
    st.set_page_config(page_title="Student Exam Score Predictor", layout="wide")
    st.title("🎓 Student Exam Score Predictor")
    st.markdown("""
    This app predicts a student's **final exam score (0–100)** based on study habits, attendance, sleep, etc.
    """)

    uploaded_file = st.file_uploader("Upload Student Data CSV", type=['csv'])

    if uploaded_file:
        df, encoders = load_and_clean_data(uploaded_file)
        if df is not None:
            st.success("Data Loaded & Cleaned Successfully!")
            st.write(df.head())

            st.divider()
            st.subheader("Model Selection")
            mode = st.radio("Choose mode:", ["Compare All Models", "Use Single Model"], horizontal=True)

            feature_names = df.drop('Target_Exam_Score', axis=1).columns

            if mode == "Compare All Models":
                with st.spinner("Training all models..."):
                    results, X_test, y_test = train_all_models(df)
                compare_all_models(results, X_test, y_test)  # Pass X_test, y_test properly

                st.divider()
                st.subheader("Detailed Model Analysis")
                valid_models = [name for name, res in results.items() if 'error' not in res]
                selected_model = st.selectbox("Select a model to explore:", valid_models)

                if selected_model:
                    result = results[selected_model]
                    plot_charts(result, selected_model, X_test, y_test)
                    predict_user_input(result['model'], feature_names, encoders)

            else:
                model_options = ['Linear Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'SVR']
                selected_model = st.selectbox("Select a model:", model_options)

                with st.spinner(f"Training {selected_model}..."):
                    model, r2, mae, X_test, y_test, y_pred = train_model(df, selected_model)
                    single_result = {
                        'model': model,
                        'r2_score': r2,
                        'mae': mae,
                        'y_pred': y_pred,
                        'y_test': y_test
                    }
                    plot_charts(single_result, selected_model, X_test, y_test)
                    predict_user_input(model, feature_names, encoders)

if __name__ == "__main__":
    main()