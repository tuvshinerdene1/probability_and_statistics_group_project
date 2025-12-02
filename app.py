import streamlit as st
# Import functions from other files
from data_loader import load_and_clean_data
from model_builder import train_model, train_all_models
from visualizer import plot_charts, compare_all_models
from predictor import predict_user_input

def main():
    # Веб хуудасны гарчиг болон тохиргоо
    st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    st.title("🎓 Student Performance Predictor")
    st.markdown("Predicting if a student will satisfy **GPA > 3.0** based on habits.")
    
    # Өгөгдлийг байршуулах
    uploaded_file = st.file_uploader("Upload Student Data CSV", type=['csv'])
    
    if uploaded_file:
        # Өгөгдлийг цэвэрлэнэ
        df, encoders = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.write("Data Loaded Successfully!", df.head())
            
            # Загвар сонгох хэсэг
            st.divider()
            st.subheader("Model Selection")
            
            # Бүх загвар эсвэл ганц загвар сургах сонголт
            mode = st.radio(
                "Choose mode:",
                ["Compare All Models", "Use Single Model"],
                horizontal=True
            )
            
            # Бүх загварыг сонгох
            if mode == "Compare All Models":
                with st.spinner("Training all models..."):
                    # бүх загварыг сурган үр дүнг авах
                    results, X_test, y_test = train_all_models(df)
                
                # загваруудын үр дүнг харьцуулна
                compare_all_models(results)
                
                st.divider()
                st.subheader("📊 Individual Model Details")
                
                selected_model = st.selectbox(
                    "Select model:",
                    [name for name, result in results.items() if 'error' not in result]
                )
                
                # Нарийвчилж харах загварыг сонгох (Алдаа гараагүй загваруудаас)
                if selected_model:
                    result = results[selected_model]
                    # Сонгосон загварын нарийвчилсан график болон үр дүнг харуулах
                    plot_charts(result, selected_model, X_test, y_test)

                    # Таамаглал хийх хэсэг (Prediction)
                    feature_names = df.drop('Target_GPA', axis=1).columns
                    predict_user_input(result['model'], feature_names, encoders)
            
            # Ганц загвар сонгох
            else:
                # нэг загвар сургах
                model_options = [
                    'Linear Regression',
                    'Random Forest',
                    'Decision Tree',
                    'Gradient Boosting',
                    'SVR'
                ]
                
                # загвараа сонгох
                selected_model = st.selectbox("Select a model:", model_options)
                
                with st.spinner(f"Training {selected_model}..."):
                    # сонгосон загварыг сургах
                    model, r2, mae, X_test, y_test, y_pred = train_model(df, selected_model)

                # үр дүнг нэгтгэх
                single_result = {
                    'model': model,
                    'r2_score': r2,
                    'mae': mae,
                    'y_pred': y_pred
                }
                # график болон үнэлгээг харуулах
                plot_charts(single_result, selected_model, X_test, y_test)

                # таамаглал хийх
                feature_names = df.drop('Target_GPA', axis=1).columns
                predict_user_input(model, feature_names, encoders)
if __name__ == "__main__":
    main()