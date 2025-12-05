
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(uploaded_file):
    """
    Өгөгдлийг CSV файлаас уншиж, цэвэрлэж, сургалтад бэлтгэх функц.
    """
    if uploaded_file is not None:
        try:
            # CSV файлыг Pandas DataFrame болгон унших
            df = pd.read_csv(uploaded_file)

            # Сургалтад шаардлагагүй эсвэл үр дүнд шууд нөлөөлөх (leakage) багануудыг хасах
            # Жишээ нь: Оюутны ID дугаар болон өмнөх голч дүн
            columns_to_drop = ['student_id', 'previous_gpa']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

            # Хоосон утгатай (NaN) мөрүүдийг устгах
            df = df.dropna()

            encoders = {}
            # Текст (object) болон Boolean төрлийн багануудыг ялгаж авах
            categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
            
            for col in categorical_cols:
                le = LabelEncoder()
                # Категори өгөгдлийг тоон хэлбэрт (0, 1, 2...) шилжүүлэх
                df[col] = le.fit_transform(df[col].astype(str))
                # Дараа нь хэрэглэгчийн оролтыг хувиргахад ашиглахын тулд encoder-ийг хадгалах
                encoders[col] = le

            # Зорилтот хувьсагч буюу 'exam_score' багана байгаа эсэхийг шалгах
            if 'exam_score' not in df.columns:
                st.error("Column 'exam_score' not found in the dataset!")
                return None, None

            # Тодорхой байх үүднээс зорилтот баганын нэрийг өөрчлөх
            df = df.rename(columns={'exam_score': 'Target_Exam_Score'})
            
            return df, encoders

        except Exception as e:
            # Ямар нэгэн алдаа гарвал Streamlit дээр харуулах
            st.error(f"Error loading data: {e}")
            return None, None
            
    return None, None