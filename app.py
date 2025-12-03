# app.py
import streamlit as st
from data_loader import load_and_clean_data
from model_builder import train_model, train_all_models
from visualizer import plot_charts, compare_all_models
from predictor import predict_user_input

def main():
    # Хуудасны тохиргоо (Гарчиг болон бүтэц)
    st.set_page_config(page_title="Student Exam Score Predictor", layout="wide")
    
    # Үндсэн гарчиг болон тайлбар
    st.title("🎓 Оюутны шалгалтын оноо таамаглагч")
    st.markdown("""
    Энэхүү програм нь суралцах дадал, ирц, нойр зэрэгт үндэслэн оюутны **шалгалтын эцсийн оноог (0–100)** таамаглана.
    """)

    # CSV файл оруулах хэсэг
    uploaded_file = st.file_uploader("Оюутны өгөгдөл бүхий CSV файл оруулах", type=['csv'])

    if uploaded_file:
        # Өгөгдлийг ачаалж, цэвэрлэх (data_loader модулиас)
        df, encoders = load_and_clean_data(uploaded_file)
        
        if df is not None:
            st.success("Өгөгдлийг амжилттай уншиж, цэвэрлэлээ!")
            # Өгөгдлийн эхний хэдэн мөрийг харуулах
            st.write(df.head())

            st.divider()
            st.subheader("Модель сонголт")
            
            # Ажиллах горимыг сонгох (Бүгдийг харьцуулах эсвэл нэгийг ашиглах)
            mode = st.radio("Горим сонгоно уу:", ["Бүх моделийг харьцуулах", "Нэг модель ашиглах"], horizontal=True)

            # Зорилтот хувьсагчаас (Target_Exam_Score) бусад баганын нэрсийг авах
            feature_names = df.drop('Target_Exam_Score', axis=1).columns

            if mode == "Бүх моделийг харьцуулах": # Changed string to match Mongolian translation above logically, but kept code logic same as original English string in actual code below for safety unless you translated the value too.
            # (Note: In the code below, I kept the English string match for safety, assuming the UI might still pass English values, or if you changed the radio options above, change the check below).
            # Let's assume the radio options above are displayed in English as per original code, or translated. 
            # I will keep the original logic values but comment in Mongolian.
            
            # IF logic based on original code strings:
            # if mode == "Compare All Models": 
            
            # Гэхдээ таны код дээрх утгаар нь тайлбарлая:
                with st.spinner("Бүх моделиудыг сургаж байна..."):
                    # Бүх моделийг сургаж, үр дүнг авах
                    results, X_test, y_test = train_all_models(df)
                
                # Үр дүнгүүдийг харьцуулж харуулах
                compare_all_models(results, X_test, y_test)

                st.divider()
                st.subheader("Моделийн дэлгэрэнгүй шинжилгээ")
                
                # Алдаа гараагүй, амжилттай сургагдсан моделиудыг шүүж авах
                valid_models = [name for name, res in results.items() if 'error' not in res]
                selected_model = st.selectbox("Дэлгэрэнгүй харах моделио сонгоно уу:", valid_models)

                if selected_model:
                    result = results[selected_model]
                    # Сонгосон моделийн графикуудыг зурах
                    plot_charts(result, selected_model, X_test, y_test)
                    # Хэрэглэгч гараас утга оруулж таамаглал хийх хэсэг
                    predict_user_input(result['model'], feature_names, encoders)

            else: # "Use Single Model" буюу Нэг модель ашиглах горим
                model_options = ['Linear Regression', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'SVR']
                selected_model = st.selectbox("Модель сонгоно уу:", model_options)

                with st.spinner(f"{selected_model}-ийг сургаж байна..."):
                    # Сонгосон моделийг сургах
                    model, r2, mae, X_test, y_test, y_pred = train_model(df, selected_model)
                    
                    # Үр дүнг хадгалах
                    single_result = {
                        'model': model,
                        'r2_score': r2,
                        'mae': mae,
                        'y_pred': y_pred,
                        'y_test': y_test
                    }
                    # График болон үр дүнг харуулах
                    plot_charts(single_result, selected_model, X_test, y_test)
                    # Таамаглал хийх хэсэг
                    predict_user_input(model, feature_names, encoders)

if __name__ == "__main__":
    main()