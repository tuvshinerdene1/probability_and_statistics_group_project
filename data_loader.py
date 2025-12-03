# data_loader.py
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(uploaded_file):
    """
    Өгөгдлийг ачаалж, цэвэрлэх үндсэн функц
    """
    if uploaded_file is not None:
        try:
            # CSV файлыг Pandas DataFrame руу унших
            df = pd.read_csv(uploaded_file)

            # Хасах шаардлагатай багануудыг тодорхойлох (ID болон өмнөх дүн гэх мэт)
            # Эдгээр нь сургалтанд шаардлагагүй гэж үзсэн баганууд
            columns_to_drop = ['student_id', 'previous_gpa']
            
            # Баганууд дата дотор байгаа эсэхийг шалгаад хасах
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

            # Дутуу утгатай мөрүүдийг устгах (Missing values handling)
            df = df.dropna()

            # Категори (текст) өгөгдлийг тоон утга руу хөрвүүлэх бэлтгэл
            encoders = {}
            # Объект (текст) болон boolean төрлийн багануудыг ялгаж авах
            categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
            
            for col in categorical_cols:
                le = LabelEncoder()
                # Текст утгуудыг тоонд шилжүүлж (Encoding), өгөгдлөө шинэчлэх
                df[col] = le.fit_transform(df[col].astype(str))
                # Буцаагаад хөрвүүлэхэд хэрэгтэй тул encoder-ийг хадгалах
                encoders[col] = le

            # 'exam_score' буюу таамаглах гэж буй гол багана байгаа эсэхийг шалгах
            if 'exam_score' not in df.columns:
                st.error("Өгөгдөл дотор 'exam_score' багана олдсонгүй!")
                return None, None

            # Гол таамаглах баганын нэрийг 'Target_Exam_Score' болгож өөрчлөх (код дотор ашиглахад хялбар болгох үүднээс)
            df = df.rename(columns={'exam_score': 'Target_Exam_Score'})
            
            # Цэвэрлэсэн өгөгдөл болон encoder-уудыг буцаах
            return df, encoders

        except Exception as e:
            # Алдаа гарсан тохиолдолд дэлгэцэнд харуулах
            st.error(f"Өгөгдлийг уншихад алдаа гарлаа: {e}")
            return None, None
    
    # Файл оруулаагүй бол None буцаах
    return None, None