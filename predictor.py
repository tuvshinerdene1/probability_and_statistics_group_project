import streamlit as st
import pandas as pd

# NOTE: Ideally, you should pass 'encoders' here too for dropdowns,
# but keeping it simple as per your skeleton.
def predict_user_input(model, feature_names):
    st.sidebar.header("📝 Student Profile")
    user_data = {}
    
    # Dynamically create sliders for every feature
    for col in feature_names:
        user_data[col] = st.sidebar.slider(f"{col}", 0, 10, 5)
        
    if st.sidebar.button("Predict Performance"):
        # Convert dict to DataFrame
        input_df = pd.DataFrame([user_data])
        
        try:
            prediction = model.predict(input_df)[0]
            
            st.divider()
            st.subheader("🤖 AI Prediction")
            
            if prediction == 1:
                st.success("🌟 Result: HIGH PERFORMER (GPA > 3.0)")
                st.balloons()
            else:
                st.error("⚠️ Result: NEEDS IMPROVEMENT (GPA < 3.0)")
                st.info("Tip: Try increasing study time or motivation level!")
        except Exception as e:
            st.error(f"Prediction Error: {e}")