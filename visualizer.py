import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_charts(result, model_name="Model", X_test=None, y_test=None):
    model = result['model']
    r2 = result['r2_score']
    mae = result['mae']
    y_pred = result.get('y_pred', model.predict(X_test))
    
    st.subheader(f"📊 {model_name} Performance")
    
    # Metrics
    c1, c2 = st.columns(2)
    c1.metric("R² Score (Accuracy)", f"{r2:.2f}")
    c2.metric("Mean Error (GPA Points)", f"{mae:.2f}")
    
    tab1, tab2 = st.tabs(["Actual vs Predicted", "Feature Importance"])
    
    # --- TAB 1: SCATTER PLOT ---
    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        # Plot the ideal line
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
        # Plot real data
        ax.scatter(y_test, y_pred, alpha=0.7, color='blue')
        
        ax.set_xlabel("Actual GPA")
        ax.set_ylabel("Predicted GPA")
        ax.set_title("How close are the predictions?")
        ax.legend()
        st.pyplot(fig)

    # --- TAB 2: FEATURE IMPORTANCE ---
    with tab2:
        if hasattr(model, 'feature_importances_') and X_test is not None:
            importances = model.feature_importances_
            feature_names = X_test.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax_feat)
            ax_feat.set_title("What affects GPA the most?")
            st.pyplot(fig_feat)
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = model.coef_
            feature_names = X_test.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importances)})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma', ax=ax_feat)
            st.pyplot(fig_feat)
        else:
            st.info("Feature importance not available for this model type.")

def compare_all_models(results):
    st.subheader("📈 Model Comparison")
    
    comparison_data = []
    for name, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                'Model': name,
                'R2 Score': result['r2_score'],
                'Error (MAE)': result['mae']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data).sort_values('R2 Score', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df_comparison.style.background_gradient(cmap='Greens', subset=['R2 Score']), use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='R2 Score', y='Model', data=df_comparison, palette='viridis', ax=ax)
            ax.set_xlim(0, 1.0)
            ax.set_title("Accuracy (R² Score) - Higher is Better")
            st.pyplot(fig)