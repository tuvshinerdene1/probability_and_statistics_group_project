import streamlit as st
import pandas as pd
import numpy as np

def predict_user_input(model, feature_names, encoders):
    """
    Generates a sidebar form dynamically based on columns.
    - Skips 'student_id'
    - Uses Dropdowns (Selectbox) for Categorical data (found in encoders)
    - Uses Sliders for Numerical data with intelligent ranges
    """
    st.sidebar.markdown("## 📝 Student Profile Input")
    st.sidebar.caption("Adjust values to predict GPA performance.")
    
    user_input = {}
    
    # Define custom ranges for specific known columns to make UI better
    # Format: 'column_name': (min, max, default, step)
    custom_ranges = {
        'age': (16, 50, 20, 1),
        'study_hours_per_day': (0.0, 24.0, 4.0, 0.5),
        'sleep_hours': (0.0, 24.0, 7.0, 0.5),
        'social_media_hours': (0.0, 12.0, 2.0, 0.1),
        'attendance_percentage': (0.0, 100.0, 85.0, 1.0),
        'stress_level': (0, 10, 5, 1),
        'motivation_level': (0, 10, 5, 1),
        'exam_score': (0, 100, 75, 1),
        'previous_gpa': (0.0, 4.0, 3.0, 0.1), # Just in case it slips in
        'netflix_hours': (0.0, 10.0, 1.0, 0.5),
    }

    # Group inputs for better UX (Optional logical grouping)
    with st.sidebar.form("prediction_form"):
        
        for col in feature_names:
            # 1. Skip ID columns or Target columns if they exist
            if 'id' in col.lower() or 'target' in col.lower():
                continue
            
            # 2. Handle Categorical Data (if encoder exists for this column)
            if col in encoders:
                le = encoders[col]
                # specific clean up for label names if needed
                options = list(le.classes_)
                selected_val = st.selectbox(f"{col.replace('_', ' ').title()}", options)
                # Transform back to number for the model
                user_input[col] = le.transform([selected_val])[0]
            
            # 3. Handle Numerical Data
            else:
                formatted_name = col.replace('_', ' ').title()
                
                if col in custom_ranges:
                    min_v, max_v, def_v, step_v = custom_ranges[col]
                    user_input[col] = st.slider(formatted_name, min_v, max_v, def_v, step_v)
                else:
                    # Generic fallback for unknown numeric columns
                    user_input[col] = st.slider(f"{formatted_name}", 0.0, 100.0, 10.0)

        submit_button = st.form_submit_button("🔮 Predict Performance")

    if submit_button:
        # Create DataFrame from input
        input_df = pd.DataFrame([user_input])
        
        # Ensure column order matches the training data
        # We reindex just to be safe, filling missing with 0 (though shouldn't happen)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
            
            st.divider()
            st.markdown("### 🤖 AI Prediction Result")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if prediction == 1:
                    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100) # Graduation cap
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100) # Warning sign

            with col2:
                if prediction == 1:
                    st.success("## 🌟 HIGH PERFORMER")
                    st.markdown("**Predicted Outcome:** GPA > 3.0")
                    if probability:
                        st.write(f"**Confidence:** {probability:.1%}")
                    st.balloons()
                else:
                    st.error("## ⚠️ AT RISK / NEEDS IMPROVEMENT")
                    st.markdown("**Predicted Outcome:** GPA < 3.0")
                    if probability:
                        st.write(f"**Confidence:** {(1-probability):.1%}")
                    st.info("💡 **Tip:** Check 'Feature Importance' in the Visualizer tab to see what habits to change!")

        except Exception as e:
            st.error(f"Prediction Error: {e}")