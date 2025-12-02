import streamlit as st
import pandas as pd
import numpy as np

def predict_user_input(model, feature_names, encoders):
    """
    Хэрэглэгчээс хажуугийн самбараар (sidebar) өгөгдөл авч, 
    сонгосон загвараар GPA-ийг таамаглах функц.
    """
    st.sidebar.markdown("## 📝 Student Profile Input")
    
    # Хэрэглэгчийн оруулсан утгуудыг хадгалах толь бичиг
    user_input = {}
    
    # Слайдеруудын утгын хязгаарыг тохируулах (min, max, default, step)
    custom_ranges = {
        'age': (16, 50, 20, 1),
        'study_hours_per_day': (0.0, 24.0, 4.0, 0.5),
        'sleep_hours': (0.0, 24.0, 7.0, 0.5),
        'social_media_hours': (0.0, 12.0, 2.0, 0.1),
        'attendance_percentage': (0.0, 100.0, 85.0, 1.0),
        'stress_level': (0, 10, 5, 1),
        'motivation_level': (0, 10, 5, 1),
        'exam_score': (0, 100, 75, 1),
        'netflix_hours': (0.0, 10.0, 1.0, 0.5),
    }

    # Хажуугийн самбар дээр форм үүсгэх
    with st.sidebar.form("prediction_form"):
        # Сургалтын өгөгдлийн бүх баганаар давталт хийж, тохирох input үүсгэх
        for col in feature_names:
            col_lower = col.lower()

            # Зорилтот багана (Target) болон ID багануудыг оруулах хэсгээс хасах
            if 'target' in col_lower:
                continue
            if 'student_id' in col_lower or col_lower == 'id' or col_lower.endswith('_id'):
                continue
            
            # Хэрэв багана нь категори (текст) өгөгдөл байвал Selectbox ашиглах
            if col in encoders:
                le = encoders[col]
                options = list(le.classes_) # Сонголтуудыг авах
                selected_val = st.selectbox(f"{col.replace('_', ' ').title()}", options)
                # Сонгосон утгыг тоон хэлбэрт (encoder) шилжүүлж хадгалах
                user_input[col] = le.transform([selected_val])[0]
            
            # Хэрэв тоон өгөгдөл байвал Slider ашиглах
            else:
                formatted_name = col.replace('_', ' ').title()
                # Тусгайлан тохируулсан хязгаар (custom_ranges) байгаа эсэхийг шалгах
                if col in custom_ranges:
                    min_v, max_v, def_v, step_v = custom_ranges[col]
                    user_input[col] = st.slider(formatted_name, min_v, max_v, def_v, step_v)
                else:
                    # Анхдагч тохиргоо
                    user_input[col] = st.slider(f"{formatted_name}", 0.0, 100.0, 10.0)

        # Таамаглах товчлуур
        submit_button = st.form_submit_button("🔮 Predict GPA")

    # Товч дарагдсан үед ажиллах хэсэг
    if submit_button:
        # Оруулсан өгөгдлийг DataFrame болгох
        input_df = pd.DataFrame([user_input])
        # Баганын дарааллыг сургалтын өгөгдөлтэй яг ижил болгож, дутууг 0-ээр дүүргэх
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        try:
            # Загварыг ашиглан таамаглал хийх
            raw_prediction = model.predict(input_df)[0]
        
            # GPA нь 0.0 - 4.0 хооронд байх ёстой тул хязгаарлах
            prediction = min(max(raw_prediction, 0.0), 4.0)

            # Үр дүнг харуулах хэсэг
            st.divider()
            st.markdown("### 🤖 Prediction Result")
            
            col1, col2 = st.columns([1, 3])
            
            # Зураг харуулах
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100) 

            # Үр дүнгийн тоо болон зөвлөгөөг харуулах
            with col2:
                st.metric("Predicted GPA", f"{prediction:.2f}")
                
                # GPA-ийн утгаас хамаарч өөр өөр мессеж харуулах
                if prediction == 4.0:
                    st.success("🏆 Perfect Score! You are maximizing your potential.")
                elif prediction >= 3.5:
                    st.success("🌟 Excellent! You are on track for top performance.")
                elif prediction >= 3.0:
                    st.info("✅ Good job. You are maintaining a solid GPA.")
                elif prediction >= 2.0:
                    st.warning("⚠️ Average. Consider increasing study hours.")
                else:
                    st.error("🚨 At Risk. Major changes in habits recommended.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")