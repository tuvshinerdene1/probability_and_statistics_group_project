import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(uploaded_file):
    """
    Өгөгдлийг CSV файлаас уншиж, цэвэрлэж, сургалтад бэлтгэх функц.
    """
    if uploaded_file is not None:
        try:
            # CSV файлыг Pandas DataFrame болгож унших
            df = pd.read_csv(uploaded_file)
            
            # 'Student_ID' багана байвал хасах (сургалтад шаардлагагүй өгөгдөл)
            if 'Student_ID' in df.columns:
                df = df.drop(columns=['Student_ID'])
            
            # Дутуу утгатай мөрүүдийг устгах (Missing values handling)
            df = df.dropna()
            
            # Категори (текст) өгөгдлийг тоон хэлбэрт шилжүүлэх (Label Encoding)
            encoders = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le  # Encoder-ийг дараа нь утга тайлахад ашиглахаар хадгалах

            # Бидний таамаглах гэж буй гол баганын нэр (CSV доторх нэр)
            target_col_name = 'previous_gpa' 
            
            # Зорилтот багана байгаа эсэхийг шалгаж, нэрийг 'Target_GPA' болгож өөрчлөх
            if target_col_name in df.columns:
                df = df.rename(columns={target_col_name: 'Target_GPA'})
                return df, encoders
            else:
                # Хэрэв зорилтот багана олдохгүй бол алдаа заах
                st.error(f"Column '{target_col_name}' not found. Please check your CSV.")
                return None, None
                
        except Exception as e:
            # Өгөгдөл боловсруулах явцад ямар нэгэн алдаа гарвал хэрэглэгчид харуулах
            st.error(f"Error processing data: {e}")
            return None, None
            
    # Файл оруулаагүй тохиолдолд хоосон утга буцаах
    return None, None