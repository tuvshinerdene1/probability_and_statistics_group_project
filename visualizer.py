# visualizer.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def plot_charts(result, model_name="Model", X_test=None, y_test=None):
    """
    Сонгосон нэг моделийн үр дүнг нарийвчлан шинжилж, графикуудыг зурах функц.
    """
    model = result['model']
    r2 = result['r2_score']
    mae = result['mae']
    
    # Хэрэв y_pred (таамагласан утга) үр дүн дотор байхгүй бол шинээр тооцоолох
    y_pred = result.get('y_pred', model.predict(X_test)) if X_test is not None else result['y_pred']

    st.subheader(f"📊 {model_name} Performance")

    # --- 1. РЕГРЕССИЙН ҮНЭЛГЭЭ ---
    st.markdown("#### Regression Metrics")
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}") # 1-рүү дөхөх тусам сайн
    c2.metric("MAE (Mean Absolute Error)", f"{mae:.2f}") # Бага байх тусам сайн

    # --- 2. АНГИЛЛЫН ҮНЭЛГЭЭ (Classification) ---
    # Шалгалтын оноог босго оноо (60)-оор таслан "Тэнцсэн/Унасан" гэж ангилах
    threshold = 60.0
    y_test_bin = (y_test >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_test_bin, y_pred_bin)
    prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

    st.markdown(f"#### Classification Metrics (Pass Threshold: ≥ {threshold})")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{acc:.2%}")
    k2.metric("Precision", f"{prec:.2f}")
    k3.metric("Recall", f"{rec:.2f}")
    k4.metric("F1 Score", f"{f1:.2f}")

    # --- 3. ГРАФИК БОЛОН ДЭЛГЭРЭНГҮЙ ҮР ДҮН ---
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Feature Importance", "Detailed Results"])

    with tab1:
        # График 1: Бодит vs Таамагласан утгын хамаарал (Scatter Plot)
        fig, ax = plt.subplots()
        # Төгс таамаглалын шугам (Улаан тасархай шугам)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect")
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Actual Exam Score")
        ax.set_ylabel("Predicted Exam Score")
        ax.set_title("Prediction Accuracy")
        ax.legend()
        st.pyplot(fig)

        # График 2: Төөрөгдлийн матриц (Confusion Matrix) - Тэнцсэн/Унасан байдлаар
        cm = confusion_matrix(y_test_bin, y_pred_bin)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fail <60', 'Pass ≥60'],
                    yticklabels=['Fail <60', 'Pass ≥60'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with tab2:
        # Аль шинж чанарууд (feature) үр дүнд хамгийн их нөлөөлсөн бэ?
        if hasattr(model, 'feature_importances_'):
            # Decision Tree, Random Forest гэх мэт мод суурьтай загварууд
            imp = model.feature_importances_
            feats = X_test.columns
            df_imp = pd.DataFrame({'Feature': feats, 'Importance': imp}).sort_values('Importance', ascending=False).head(10)
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=df_imp, ax=ax)
            st.pyplot(fig)
        elif hasattr(model, 'coef_'):
            # Linear Regression гэх мэт шугаман загварууд
            imp = np.abs(model.coef_)
            df_imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': imp}).sort_values('Importance', ascending=False).head(10)
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=df_imp, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Feature importance not available.")

    with tab3:
        # Нарийвчилсан тоон үр дүнг хүснэгтээр харуулах
        df_res = pd.DataFrame({
            'Actual': np.round(y_test, 1),
            'Predicted': np.round(y_pred, 1),
            'Error': np.round(np.abs(y_test - y_pred), 1),
            'Status': ['Pass' if x >= threshold else 'Fail' for x in y_test],
            'Pred Status': ['Pass' if x >= threshold else 'Fail' for x in y_pred]
        })
        # Алдаа ихтэй хэсгүүдийг улаан өнгөөр тодруулах
        st.dataframe(df_res.style.background_gradient(cmap='Reds', subset=['Error']), use_container_width=True)

def compare_all_models(results, X_test, y_test):
    """
    Сургасан бүх моделиудын үр дүнг хооронд нь харьцуулах функц.
    """
    st.subheader("Model Comparison")
    threshold = 60.0
    y_test_bin = (y_test >= threshold).astype(int)

    data = []
    # Үр дүн бүрээс хэрэгтэй мэдээллийг цуглуулах
    for name, res in results.items():
        if 'error' not in res and 'y_pred' in res:
            y_pred = res['y_pred']
            y_pred_bin = (y_pred >= threshold).astype(int)
            data.append({
                'Model': name,
                'R² Score': res['r2_score'],
                'MAE': res['mae'],
                'Accuracy': accuracy_score(y_test_bin, y_pred_bin),
                'F1 Score': f1_score(y_test_bin, y_pred_bin, zero_division=0)
            })

    if data:
        # Үр дүнг эрэмбэлж харуулах
        df = pd.DataFrame(data).sort_values('R² Score', ascending=False)
        st.dataframe(df.style.format({'R² Score': '{:.3f}', 'MAE': '{:.2f}', 'Accuracy': '{:.2%}', 'F1 Score': '{:.3f}'})
                     .background_gradient(cmap='Greens', subset=['R² Score', 'Accuracy']), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # R2 онооны харьцуулалт (График)
            fig, ax = plt.subplots()
            sns.barplot(x='R² Score', y='Model', data=df, ax=ax)
            ax.set_title("R² Score Comparison")
            st.pyplot(fig)
        with col2:
            # Нарийвчлалын харьцуулалт (График)
            fig, ax = plt.subplots()
            sns.barplot(x='Accuracy', y='Model', data=df, ax=ax)
            ax.set_title("Accuracy (≥60 = Pass)")
            st.pyplot(fig)