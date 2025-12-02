import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def plot_charts(result, model_name="Model", X_test=None, y_test=None):
    model = result['model']
    r2 = result['r2_score']
    mae = result['mae']
    # Ensure y_pred is available
    y_pred = result.get('y_pred', model.predict(X_test))
    
    st.subheader(f"📊 {model_name} Performance")
    
    # --- 1. REGRESSION METRICS ---
    st.markdown("#### 🔢 Regression Metrics")
    c1, c2 = st.columns(2)
    c1.metric("R² Score (Fit)", f"{r2:.2f}")
    c2.metric("Mean Error (MAE)", f"{mae:.2f}")
    
    # --- 2. CLASSIFICATION METRICS (Converted) ---
    st.markdown("#### 🚦 Classification Metrics (Threshold: GPA ≥ 3.0)")
    st.caption("Converting continuous GPA predictions to Binary (Good vs. Low) to calculate Accuracy/F1.")
    
    # Define Threshold (e.g., 3.0 out of 4.0)
    threshold = 3.0
    
    # Binarize data: 1 if >= threshold, else 0
    y_test_bin = (y_test >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test_bin, y_pred_bin)
    prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{acc:.2%}")
    k2.metric("Precision", f"{prec:.2f}")
    k3.metric("Recall", f"{rec:.2f}")
    k4.metric("F1 Score", f"{f1:.2f}")

    # --- TABS FOR VISUALIZATION ---
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Feature Importance", "Posterior Table"])
    
    # --- TAB 1: SCATTER PLOT ---
    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")
        ax.scatter(y_test, y_pred, alpha=0.7, color='blue')
        ax.set_xlabel("Actual GPA")
        ax.set_ylabel("Predicted GPA")
        ax.set_title("Regression Fit")
        ax.legend()
        st.pyplot(fig)
        
        # Add Confusion Matrix for the classification part
        st.markdown("---")
        st.markdown("**Confusion Matrix (based on Threshold)**")
        cm = confusion_matrix(y_test_bin, y_pred_bin)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['< 3.0', '>= 3.0'], yticklabels=['< 3.0', '>= 3.0'])
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    # --- TAB 2: FEATURE IMPORTANCE ---
    with tab2:
        if hasattr(model, 'feature_importances_') and X_test is not None:
            importances = model.feature_importances_
            feature_names = X_test.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis', ax=ax_feat)
            ax_feat.set_title("Feature Importance")
            st.pyplot(fig_feat)
        elif hasattr(model, 'coef_'):
            importances = model.coef_
            feature_names = X_test.columns
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importances)})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma', ax=ax_feat)
            st.pyplot(fig_feat)
        else:
            st.info("Feature importance not available for this model type.")

    # --- TAB 3: POSTERIOR TABLE (Detailed Data) ---
    with tab3:
        st.markdown("### 📋 Detailed Predictions")
        
        # Create a clean dataframe for display
        results_df = pd.DataFrame({
            'Actual GPA': y_test.values if hasattr(y_test, 'values') else y_test,
            'Predicted GPA': np.round(y_pred, 2),
            'Error (Diff)': np.round(np.abs(y_test - y_pred), 2),
            'Status (Act)': ['High' if x >= threshold else 'Low' for x in y_test],
            'Status (Pred)': ['High' if x >= threshold else 'Low' for x in y_pred]
        })
        
        # Highlight large errors
        st.dataframe(
            results_df.style.background_gradient(cmap='Reds', subset=['Error (Diff)']),
            use_container_width=True
        )

def compare_all_models(results, y_test_data):
    """
    y_test_data is needed globally here to calculate the extra metrics for comparison
    """
    st.subheader("📈 Model Comparison")
    
    comparison_data = []
    threshold = 3.0 # Consistent threshold
    
    for name, result in results.items():
        if 'error' not in result:
            # Re-calculate metrics for the table
            model = result['model']
            # If y_pred is stored in result, use it, else predict
            if 'y_pred' in result:
                y_p = result['y_pred']
            else:
                # Assuming X_test was saved or we skip (This part depends on your main loop structure)
                # Ideally pass y_pred into result dict in your main loop
                continue 
            
            # Binary conversion
            y_t_bin = (y_test_data >= threshold).astype(int)
            y_p_bin = (y_p >= threshold).astype(int)
            
            comparison_data.append({
                'Model': name,
                'R2 Score': result['r2_score'],
                'MAE (Error)': result['mae'],
                'Accuracy': accuracy_score(y_t_bin, y_p_bin),
                'F1 Score': f1_score(y_t_bin, y_p_bin, zero_division=0)
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data).sort_values('R2 Score', ascending=False)
        
        # Display Metrics Table
        st.dataframe(
            df_comparison.style.format("{:.2f}").background_gradient(cmap='Greens', subset=['R2 Score', 'Accuracy', 'F1 Score']),
            use_container_width=True
        )
        
        # Visualization Comparison
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='R2 Score', y='Model', data=df_comparison, palette='viridis', ax=ax)
            ax.set_xlim(0, 1.0)
            ax.set_title("R² Comparison")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Accuracy', y='Model', data=df_comparison, palette='magma', ax=ax)
            ax.set_xlim(0, 1.0)
            ax.set_title(f"Accuracy Comparison (GPA >= {threshold})")
            st.pyplot(fig)