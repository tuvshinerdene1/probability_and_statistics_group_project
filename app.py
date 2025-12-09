import streamlit as st
from data_loader import load_and_clean_data
from model_builder import train_model, train_all_models
from visualizer import plot_charts, compare_all_models
from predictor import predict_user_input

def main():
    # Хуудасны үндсэн тохиргоог хийх (Гарчиг, layout гэх мэт)
    st.set_page_config(page_title="Student Exam Score Predictor", layout="wide")
    
    # Аппликейшний гарчиг болон тайлбар
    st.title("🎓 Student Exam Score Predictor")
    st.markdown("""
    This app predicts a student's **final exam score (0–100)** based on study habits, attendance, sleep, etc.
    """)

    # Хэрэглэгчээс CSV файл хүлээн авах хэсэг
    uploaded_file = st.file_uploader("Upload Student Data CSV", type=['csv'])

    if uploaded_file:
        # Өгөгдлийг ачааллах болон цэвэрлэх функцийг дуудах
        df, encoders = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.success("Data Loaded & Cleaned Successfully!")
            
            # Өгөгдлийн хэмжээ хэт том эсэхийг шалгах (15,000 мөрөөс дээш бол)
            # Том өгөгдөл SVR гэх мэт загваруудыг удаашруулж болзошгүй
            if len(df) > 15000:
                st.warning(f"Dataset contains {len(df):,} rows. SVR and complex models may crash or take very long.")
                
                # Санамсаргүй түүвэрлэлт (sampling) ашиглах эсэхийг асуух
                use_subset = st.checkbox("Use 10,000 random samples for faster performance (Recommended)", value=True)
                
                if use_subset:
                    # 10,000 мөрийг санамсаргүйгээр сонгож авах
                    df = df.sample(n=10000, random_state=42)
                    st.info("Dataset downsampled to 10,000 random rows (Fair Random Sample).")
                else:
                    st.error("Using full dataset. Please be patient!")
            
            # Өгөгдлийн эхний хэдэн мөрийг харуулах
            st.write(df.head())

            st.divider()
            st.subheader("Model Selection")
            
            # Ажиллах горимыг сонгох: Бүх моделийг харьцуулах эсвэл нэгийг сонгож ашиглах
            mode = st.radio("Choose mode:", ["Compare All Models", "Use Single Model"], horizontal=True)

            # Зорилтот хувьсагчаас (Target_Exam_Score) бусад багануудын нэрийг авах
            feature_names = df.drop('Target_Exam_Score', axis=1).columns

            # --- ГОРИМ 1: БҮХ МОДЕЛИЙГ ХАРЬЦУУЛАХ ---
            if mode == "Compare All Models":
                with st.spinner("Training all models..."):
                    # Бүх моделиудыг сургаж үр дүнг авах
                    results, X_test, y_test = train_all_models(df)
                
                # Үр дүнгүүдийг хүснэгт болон графикаар харьцуулах
                compare_all_models(results, X_test, y_test)

                st.divider()
                st.subheader("Detailed Model Analysis")
                
                # Алдаа гараагүй, амжилттай сургагдсан моделиудыг шүүж авах
                valid_models = [name for name, res in results.items() if 'error' not in res]
                
                # Дэлгэрэнгүй шинжлэх моделийг сонгох
                selected_model = st.selectbox("Select a model to explore:", valid_models)

                if selected_model:
                    result = results[selected_model]
                    # Сонгосон моделийн нарийвчилсан графикуудыг зурах
                    plot_charts(result, selected_model, X_test, y_test)
                    # Хэрэглэгчийн гараас өгөгдөл авч таамаглал хийх
                    predict_user_input(result['model'], feature_names, encoders)

            # --- ГОРИМ 2: НЭГ МОДЕЛЬ АШИГЛАХ ---
            else:
                model_options = ['Linear Regression','Ridge Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'SVR']
                selected_model = st.selectbox("Select a model:", model_options)

                with st.spinner(f"Training {selected_model}..."):
                    # Сонгосон моделийг сургах
                    model, r2, mae, X_test, y_test, y_pred = train_model(df, selected_model)
                    
                    # Үр дүнг dictionary хэлбэрээр хадгалах
                    single_result = {
                        'model': model,
                        'r2_score': r2,
                        'mae': mae,
                        'y_pred': y_pred,
                        'y_test': y_test
                    }
                    # Графикуудыг зурах
                    plot_charts(single_result, selected_model, X_test, y_test)
                    # Хэрэглэгчийн гараас өгөгдөл авч таамаглал хийх
                    predict_user_input(model, feature_names, encoders)

if __name__ == "__main__":
    main()