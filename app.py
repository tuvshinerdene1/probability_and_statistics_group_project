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
            
# ... inside main function ...
            
            if mode == "Compare All Models":
                with st.spinner("Training all models..."):
                    results, X_test, y_test = train_all_models(df)
                
                compare_all_models(results)
                
                st.divider()
                st.subheader("📊 Individual Model Details")
                
                selected_model = st.selectbox(
                    "Select model:",
                    [name for name, result in results.items() if 'error' not in result]
                )
                
                if selected_model:
                    result = results[selected_model]
                    plot_charts(result, selected_model, X_test, y_test)
                    
                    feature_names = df.drop('Target_GPA', axis=1).columns
                    predict_user_input(result['model'], feature_names, encoders)
            
            else:
                # Single model mode
                model_options = [
                    'Linear Regression',
                    'Random Forest',
                    'Decision Tree',
                    'Gradient Boosting',
                    'SVR'
                ]
                
                selected_model = st.selectbox("Select a model:", model_options)
                
                with st.spinner(f"Training {selected_model}..."):
                    # NOTE: Updated unpacking to match new return values
                    model, r2, mae, X_test, y_test, y_pred = train_model(df, selected_model)

                single_result = {
                    'model': model,
                    'r2_score': r2,
                    'mae': mae,
                    'y_pred': y_pred
                }
                plot_charts(single_result, selected_model, X_test, y_test)

                feature_names = df.drop('Target_GPA', axis=1).columns
                predict_user_input(model, feature_names, encoders)
if __name__ == "__main__":
    main()