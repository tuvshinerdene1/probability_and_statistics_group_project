import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def plot_charts(result, model_name="Model", X_test=None, y_test=None):
    """
    Shows detailed performance metrics: Confusion Matrix, ROC Curve, and Feature Importance.
    """
    model = result['model']
    accuracy = result['accuracy']
    cm = result['confusion_matrix']
    
    st.subheader(f"📊 {model_name} Deep Dive")
    
    # 1. Top Level Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.2%}")
    
    # Extract specific metrics from classification report if needed, 
    # but for now, we'll show the heatmap.
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])
    
    # --- TAB 1: CONFUSION MATRIX ---
    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        st.pyplot(fig)
        
        st.caption("0: Low GPA (< 3.0) | 1: High GPA (>= 3.0)")

    # --- TAB 2: ROC CURVE ---
    with tab2:
        if hasattr(model, "predict_proba") and X_test is not None:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC)')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        else:
            st.info("This model does not support probability predictions for ROC curves (or X_test was missing).")

    # --- TAB 3: FEATURE IMPORTANCE ---
    with tab3:
        # Check if model has feature importances (Tree-based) or coefficients (Linear)
        if hasattr(model, 'feature_importances_') and X_test is not None:
            importances = model.feature_importances_
            feature_names = X_test.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax_feat)
            ax_feat.set_title("Top 10 Influential Factors")
            st.pyplot(fig_feat)
            
        elif hasattr(model, 'coef_') and X_test is not None:
            # For Logistic Regression / SVM linear
            importances = model.coef_[0]
            feature_names = X_test.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importances)})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma', ax=ax_feat)
            ax_feat.set_title("Top 10 Influential Factors (by Coefficient Magnitude)")
            st.pyplot(fig_feat)
        else:
            st.info("Feature importance not available for this model type.")

def compare_all_models(results):
    """
    Comparing all models using a bar chart and a highlighted dataframe.
    """
    st.subheader("📈 All Models Comparison")
    
    comparison_data = []
    for name, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy']
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
        
        # Display Metrics
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(
                df_comparison.style.format({'Accuracy': '{:.2%}'}).background_gradient(cmap='Greens'),
                use_container_width=True
            )
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Accuracy', y='Model', data=df_comparison, palette='viridis', ax=ax)
            ax.set_xlim(0, 1.0)
            ax.set_title("Model Accuracy Leaderboard")
            for i, v in enumerate(df_comparison['Accuracy']):
                ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=10)
            st.pyplot(fig)
            
        best_model_name = df_comparison.iloc[0]['Model']
        best_acc = df_comparison.iloc[0]['Accuracy']
        st.success(f"🏆 Recommendation: Use **{best_model_name}** ({best_acc:.1%} accuracy)")