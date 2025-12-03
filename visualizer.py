# visualizer.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def plot_charts(result, model_name="Model", X_test=None, y_test=None):
    """
    Нэг моделийн үр дүнг нарийвчлан шинжилж, графикууд зурах функц
    """
    model = result['model']
    r2 = result['r2_score']
    mae = result['mae']
    
    # Хэрэв X_test өгөгдсөн бол шинээр таамаглал хийх, үгүй бол хадгалсан y_pred-ийг ашиглах
    y_pred = result.get('y_pred', model.predict(X_test)) if X_test is not None else result['y_pred']

    st.subheader(f"📊 {model_name} Performance") # Моделийн нэр

    # --- РЕГРЕССИЙН ҮНЭЛГЭЭ ---
    st.markdown("#### Regression Metrics")
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}") # Таамаглалын нарийвчлал (1-д ойр бол сайн)
    c2.metric("MAE (Mean Absolute Error)", f"{mae:.2f}") # Дундаж алдаа (Бага бол сайн)

    # --- АНГИЛЛЫН ҮНЭЛГЭЭ (Classification) ---
    # Регрессийн тоон үр дүнг "Тэнцсэн/Унасан" гэсэн ангилал руу хөрвүүлж шалгах
    threshold = 70.0 # Тэнцэх босго оноо
    y_test_bin = (y_test >= threshold).astype(int) # Бодит байдал дээр тэнцсэн эсэх
    y_pred_bin = (y_pred >= threshold).astype(int) # Таамаглалаар тэнцсэн эсэх

    # Ангиллын хэмжүүрүүдийг тооцох
    acc = accuracy_score(y_test_bin, y_pred_bin)
    prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

    st.markdown(f"#### Classification Metrics (Pass Threshold: ≥ {threshold})")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{acc:.2%}")   # Нийт зөв таасан хувь
    k2.metric("Precision", f"{prec:.2f}") # Тэнцэнэ гэж тааснаас хэд нь үнэхээр тэнцсэн бэ
    k3.metric("Recall", f"{rec:.2f}")     # Бодит тэнцсэн хүмүүсээс хэдийг нь олж чадсан бэ
    k4.metric("F1 Score", f"{f1:.2f}")    # Precision, Recall-ийн тэнцвэржүүлсэн оноо

    # Графикуудыг 3 өөр цонхонд (Tab) хувааж харуулах
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Feature Importance", "Detailed Results"])

    with tab1:
        # График 1: Бодит утга vs Таамагласан утгын харьцуулалт (Scatter plot)
        fig, ax = plt.subplots()
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect") # Төгс шугам
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Actual Exam Score")
        ax.set_ylabel("Predicted Exam Score")
        ax.set_title("Prediction Accuracy")
        ax.legend()
        st.pyplot(fig)

        # График 2: Төөрөгдлийн матриц (Confusion Matrix) - Тэнцсэн/Унасан таамаглал
        cm = confusion_matrix