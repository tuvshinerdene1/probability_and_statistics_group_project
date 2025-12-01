import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Student_ID' in df.columns:
                df = df.drop(columns=['Student_ID'])
            
            # 1. Handle Missing Values
            df = df.dropna()
            
            # 2. ENCODING
            encoders = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le 

            # 3. DEFINE TARGET (REGRESSION)
            # We do NOT convert to 0/1 anymore. We keep the specific number.
            target_col_name = 'previous_gpa' # Ensure this matches your CSV column name
            
            if target_col_name in df.columns:
                # Rename to standard target name for other files to use
                df = df.rename(columns={target_col_name: 'Target_GPA'})
                return df, encoders
            else:
                st.error(f"Column '{target_col_name}' not found. Please check your CSV.")
                return None, None
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return None, None
    return None, None