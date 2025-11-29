import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # 1. Handle Missing Values
            df = df.dropna()
            
            # 2. ENCODING
            encoders = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le 

            # 3. BINNING (Target Creation)
            # CHECK THIS NAME MATCHES YOUR CSV!
            target_col_name = 'GPA' 
            
            if target_col_name in df.columns:
                df['Target_Class'] = df[target_col_name].apply(lambda x: 1 if x >= 3.0 else 0)
                df = df.drop(columns=[target_col_name])
                return df, encoders
            else:
                st.error(f"Column '{target_col_name}' not found. Please check your CSV.")
                return None, None
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return None, None
    return None, None