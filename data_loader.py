import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            columns_to_drop = ['student_id', 'previous_gpa']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

            df = df.dropna()

            encoders = {}
            categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

            if 'exam_score' not in df.columns:
                st.error("Column 'exam_score' not found in the dataset!")
                return None, None

            df = df.rename(columns={'exam_score': 'Target_Exam_Score'})
            return df, encoders

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    return None, None