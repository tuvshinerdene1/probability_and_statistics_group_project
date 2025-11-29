import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def plot_charts(cm, accuracy):
    st.subheader("📊 Model Performance")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix (True vs Predicted)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)