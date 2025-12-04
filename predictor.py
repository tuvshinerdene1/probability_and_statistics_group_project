# predictor.py
import streamlit as st
import pandas as pd
import numpy as np

def predict_user_input(model, feature_names, encoders):
    st.sidebar.markdown("## Student Profile – Predict Exam Score")

    user_input = {}

    # --- NUMERIC FEATURES (with sliders) ---
    numeric_ranges = {
        'age': (16, 35, 22, 1),
        'study_hours_per_day': (0.0, 20.0, 5.0, 0.5),
        'social_media_hours': (0.0, 12.0, 2.0, 0.5),
        'netflix_hours': (0.0, 10.0, 1.5, 0.5),
        'attendance_percentage': (40.0, 100.0, 85.0, 1.0),
        'sleep_hours': (3.0, 12.0, 7.0, 0.5),
        'stress_level': (0, 10, 5, 1),
        'motivation_level': (0, 10, 7, 1),
        'mental_health_rating': (0.0, 10.0, 6.5, 0.5),
        'exam_anxiety_score': (0, 10, 5, 1),
        'time_management_score': (0.0, 10.0, 6.0, 0.5),
        'screen_time': (0.0, 20.0, 8.0, 0.5),
        'exercise_frequency': (0, 7, 3, 1),
        'dropout_risk': (0, 10, 3, 1),
        'semester': (1, 8, 4, 1),
        # These two are now treated as numeric sliders (0–10)
        'parental_support_level': (0, 10, 7, 1),
        'social_activity': (0, 10, 6, 1),
    }

    # --- CATEGORICAL FEATURES (selectbox with nice labels) ---
    categorical_mapping = {
        'gender': ['Male', 'Female', 'Other'],
        'major': ['Computer Science', 'Arts', 'Psychology', 'Business', 'Engineering', 'Medicine', 'Law', 'Other'],
        'part_time_job': ['Yes', 'No'],
        'diet_quality': ['Poor', 'Fair', 'Good', 'Excellent'],
        'parental_education_level': ['High School', 'Some College', 'Bachelor', 'Master', 'PhD'],
        'internet_quality': ['Low', 'Medium', 'High'],
        'extracurricular_participation': ['Yes', 'No'],
        'study_environment': ['Library', 'Quiet Room', 'Co-Learning Group', 'Cafe', 'Home'],
        'access_to_tutoring': ['Yes', 'No'],
        'family_income_range': ['Low', 'Medium', 'High'],
        'learning_style': ['Reading', 'Visual', 'Auditory', 'Kinesthetic'],
    }

    with st.sidebar.form("student_prediction_form"):
        st.markdown("### Student Details")

        for col in feature_names:
            if col == 'Target_Exam_Score':
                continue

            pretty_name = col.replace('_', ' ').title()

            # Treat these two as numeric sliders (even if originally categorical)
            if col in ['parental_support_level', 'social_activity']:
                mn, mx, default, step = numeric_ranges[col]
                user_input[col] = st.slider(pretty_name, mn, mx, default, step, key=col)

            # Other numeric columns
            elif col in numeric_ranges:
                mn, mx, default, step = numeric_ranges[col]
                user_input[col] = st.slider(pretty_name, float(mn), float(mx), float(default), float(step), key=col)

            # Categorical columns
            elif col in encoders:
                le = encoders[col]
                options = le.classes_.tolist()

                # Use friendly display names
                display_options = categorical_mapping.get(col, [str(x) for x in options])
                selected_display = st.selectbox(pretty_name, options=display_options, key=col)

                # Map back to original value for encoding
                if col in categorical_mapping:
                    original_value = categorical_mapping[col][display_options.index(selected_display)]
                else:
                    original_value = selected_display

                user_input[col] = le.transform([original_value])[0]

            else:
                # Fallback (should not happen)
                user_input[col] = st.number_input(pretty_name, value=5.0, key=col)

        submitted = st.form_submit_button("Predict Exam Score", use_container_width=True, type="primary")

        if submitted:
            input_df = pd.DataFrame([user_input])
            input_df = input_df.reindex(columns=feature_names, fill_value=0)

            try:
                prediction = float(model.predict(input_df)[0])
                prediction = np.clip(prediction, 0, 100)

                st.markdown("## Prediction Result")
                st.metric(label="Predicted Exam Score", value=f"{prediction:.1f}/100", delta=None)

                if prediction >= 95:
                    st.balloons()
                    st.success("Top-tier performance expected! Likely A+")
                elif prediction >= 85:
                    st.success("Excellent!")
                elif prediction >= 75:
                    st.success("Very good result")
                elif prediction >= 70:
                    st.info("Good passing score")
                elif prediction >= 60:
                    st.warning("Passing, but needs improvement (C/D range)")
                else:
                    st.error("At risk of failing – urgent action needed!")

                # Quick personalized insights
                st.markdown("### Quick Insights")
                if user_input.get('study_hours_per_day', 0) > 7:
                    st.success("High study time – excellent habit!")
                if user_input.get('attendance_percentage', 0) > 90:
                    st.success("Outstanding attendance!")
                if user_input.get('stress_level', 0) > 7:
                    st.warning("High stress detected – consider relaxation techniques")
                if user_input.get('parental_support_level', 0) >= 8:
                    st.success("Strong parental support – big advantage!")
                if user_input.get('sleep_hours', 0) >= 7:
                    st.success("Good sleep – keeps the brain sharp!")

            except Exception as e:
                st.error(f"Prediction error: {e}")