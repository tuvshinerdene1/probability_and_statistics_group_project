import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_charts(cm, accuracy, model_name="Model"):
    """Нэг моделийн гүйцэтгэлийг харуулах"""
    st.subheader(f"📊 {model_name} Performance")
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

def compare_all_models(results):
    """Бүх моделуудыг харьцуулах хүснэгт болон график"""
    st.subheader("📈 All Models Comparison")
    
    # Хүснэгт үүсгэх
    comparison_data = []
    for name, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.2%}",
                'Accuracy (Decimal)': result['accuracy']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Accuracy (Decimal)', ascending=False)
        
        # Хүснэгт харуулах
        st.dataframe(df_comparison[['Model', 'Accuracy']].style.highlight_max(axis=0, subset=['Accuracy']), use_container_width=True)
        
        # График харуулах
        fig, ax = plt.subplots(figsize=(10, 6))
        df_sorted = df_comparison.sort_values('Accuracy (Decimal)', ascending=True)
        ax.barh(df_sorted['Model'], df_sorted['Accuracy (Decimal)'], color='steelblue')
        ax.set_xlabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlim(0, 1)
        for i, v in enumerate(df_sorted['Accuracy (Decimal)']):
            ax.text(v + 0.01, i, f'{v:.2%}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Хамгийн сайн моделийг харуулах
        best_model = df_comparison.iloc[0]
        st.success(f"🏆 Best Model: **{best_model['Model']}** with {best_model['Accuracy']} accuracy")