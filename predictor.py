import streamlit as st
import pandas as pd
import numpy as np

def predict_user_input(model, feature_names, encoders):
    st.sidebar.markdown("## 📝 Student Profile Input")
    
    user_input = {}
    
    # Custom ranges for sliders
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

    with st.sidebar.form("prediction_form"):
        for col in feature_names:
            col_lower = col.lower()

            # 1. SKIP Logic
            if 'target' in col_lower:
                continue
            if 'student_id' in col_lower or col_lower == 'id' or col_lower.endswith('_id'):
                continue
            
            # 2. Categorical
            if col in encoders:
                le = encoders[col]
                options = list(le.classes_)
                selected_val = st.selectbox(f"{col.replace('_', ' ').title()}", options)
                user_input[col] = le.transform([selected_val])[0]
            
            # 3. Numerical
            else:
                formatted_name = col.replace('_', ' ').title()
                if col in custom_ranges:
                    min_v, max_v, def_v, step_v = custom_ranges[col]
                    user_input[col] = st.slider(formatted_name, min_v, max_v, def_v, step_v)
                else:
                    user_input[col] = st.slider(f"{formatted_name}", 0.0, 100.0, 10.0)

        submit_button = st.form_submit_button("🔮 Predict GPA")

    if submit_button:
        input_df = pd.DataFrame([user_input])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        try:
            # --- THE FIX IS HERE ---
            raw_prediction = model.predict(input_df)[0]
            
            # Force the result to be between 0.0 and 4.0
            prediction = min(max(raw_prediction, 0.0), 4.0)
            # -----------------------

            st.divider()
            st.markdown("### 🤖 Prediction Result")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image("https://cdn-icons-png.flaticon.com/512/2232/2232688.png", width=100) 

            with col2:
                # Show the clipped prediction
                st.metric("Predicted GPA", f"{prediction:.2f}")
                
                # Dynamic Feedback
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