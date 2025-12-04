import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# === Load model & resources ===
model = joblib.load("best_model.pkl")

feature_engineering = joblib.load("feature_engineering.pkl")
overtime_median = feature_engineering["overtime_median"]

# === Feature engineering ===
def engineer_features(df):
    df = df.copy()

    for col in sensitive_cols:
        if col not in df.columns:
            raise ValueError(f"Missing sensitive column: {col}")
        
    # Validasi 'age'
    if "age" not in df.columns:
        raise ValueError("Kolom 'age' diperlukan untuk analisis fairness.")

    df["is_married"] = (df["marital_status"].str.strip().str.title() == "Married").astype(int)
    df["total_workload"] = df["working_hours_per_week"] + df["overtime_hours_per_week"]
    df["performance_efficiency"] = df["target_achievement"] / (df["total_workload"] + 1e-5)
    df["long_distance_overwork"] = df["distance_to_office_km"] / (df["working_hours_per_week"] + 1)
    df["low_satisfaction"] = (df["job_satisfaction"] <= 2).astype(int)
    high_overtime = (df["overtime_hours_per_week"] > overtime_median).astype(int)
    df["high_ot_low_sat"] = high_overtime & df["low_satisfaction"]

    # Buat age_group dari kolom 'age'
    q1 = df['age'].quantile(0.33)
    q2 = df['age'].quantile(0.66)

    # Membagi kelompok usia berdasarkan kuartil
    df['age_group'] = pd.cut(df['age'], bins=[22, q1, q2, 44], labels=['Young', 'Middle', 'Senior'])

    required_features = [
        'performance_efficiency',
        'low_satisfaction',
        'target_achievement',
        'high_ot_low_sat',
        'working_hours_per_week',
        'distance_to_office_km',
        'total_workload',
        'job_satisfaction',
        'long_distance_overwork',
        'company_tenure_years',
        'manager_support_score',
        'is_married'
    ]

    return df[required_features], df


# === Sidebar ===
st.sidebar.header("⚙️ Pengaturan")
mode = st.sidebar.radio("Input Mode:", ["Manual Input", "Upload File"])
threshold = st.sidebar.slider(
    "Threshold klasifikasi churn",
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="Lower threshold → more sensitive to churn"
)

# --- Manual input ---

if mode == "Manual Input":
    st.title("Employee Churn Prediction")
    col1, col2 = st.columns(2)

    with col1:
        target_achievement = st.number_input("Target Achievement", 0.0, 5.0, 1.0)
        working_hours_per_week = st.number_input("Working Hours/Week", 40, 80, 40)
        overtime_hours_per_week = st.number_input("Overtime Hours/Week", 0, 40, 5)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)

    with col2:
        distance_to_office_km = st.number_input("Distance to Office (km)", 0, 100, 10)
        company_tenure_years = st.number_input("Company Tenure (years)", 0, 30, 2)
        manager_support_score = st.slider("Manager Support Score (1-4)", 1, 4, 2)
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        age = st.number_input("Age", 18, 50, 25)

    if st.button("Predict"):
        df = pd.DataFrame([{
            "target_achievement": target_achievement,
            "working_hours_per_week": working_hours_per_week,
            "overtime_hours_per_week": overtime_hours_per_week,
            "job_satisfaction": job_satisfaction,
            "distance_to_office_km": distance_to_office_km,
            "company_tenure_years": company_tenure_years,
            "manager_support_score": manager_support_score,
            "marital_status": marital_status,
            "age": age
        }])

        X, _ = engineer_features(df)
        prob = model.predict_proba(X)[:, 1]
        pred = int(prob >= threshold)

        st.subheader(f"Prediction: {'YES' if pred else 'NO'}")
        st.write(f"Probability: {prob:.2%}")

# --- Upload file ---
else:
    st.title("Employee Churn Prediction — Batch Analysis")
    
    # Buat tab hanya untuk mode upload
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📊 Prediction Summary", "⚖️ Fairness Analysis"])

    with tab1:
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        
        sensitive_cols = ['age', 'gender', 'education', 'marital_status', 'work_location']

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)

            required = {
                "target_achievement",
                "working_hours_per_week",
                "overtime_hours_per_week",
                "job_satisfaction",
                "distance_to_office_km",
                "company_tenure_years",
                "manager_support_score",
                "marital_status",
                "age",
                "churn"
            } | set(sensitive_cols)

            missing = required - set(df_raw.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                try:
                    X, df_with_age = engineer_features(df_raw)
                    probs = model.predict_proba(X)[:, 1]
                    preds = (probs >= threshold).astype(int)

                    st.session_state['results'] = {
                        'df_full': df_with_age,
                        'X': X,
                        'probs': probs,
                        'preds': preds
                    }

                    result = df_raw.copy()
                    result["prediction"] = preds
                    result["probability"] = probs

                    st.dataframe(result.head(10))
                    st.download_button(
                        "📥 Download Full Predictions",
                        result.to_csv(index=False),
                        "predictions_result.csv",
                        "text/csv",
                    )
                except Exception as e:
                    st.error(f"Error processing file: {e}")

    # --- Tab 2: Prediction Summary ---
    with tab2:
        st.subheader("📊 Prediction Summary")

        if 'results' not in st.session_state:
            st.info("Upload data dulu di tab 'Upload CSV'")
        else:
            results = st.session_state['results']
            probs = results['probs']
            preds = results['preds']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Employees", len(probs))
            with col2:
                churn_count = preds.sum()
                st.metric("Predicted Churn", churn_count, f"{churn_count/len(probs):.1%}")
            with col3:
                st.metric("Threshold Used", threshold)

            # Histogram probabilitas
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(probs, bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Distribution of Churn Probabilities')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Count')
            st.pyplot(fig)

            # Dampak threshold
            st.write("### Impact of Threshold:")
            thresholds = np.arange(0.1, 1.0, 0.1)
            churn_rates = [(probs >= t).sum() for t in thresholds]
            impact_df = pd.DataFrame({
                'Threshold': thresholds,
                'Churn Count': churn_rates,
                'Churn Rate (%)': [c/len(probs)*100 for c in churn_rates]
            })
            st.dataframe(impact_df.round(2))

    # --- Tab 3: Fairness Analysis ---
    with tab3:
        st.subheader("⚖️ Fairness Analysis by Age Group")
        sensitive_attr_map = {
            'age_group': 'Age Group',
            'gender': 'Gender',
            'education': 'Education Level',
            'marital_status': 'Marital Status',
            'work_location': 'Work Location'
        }

        # Pilih atribut sensitif (default: age_group)
        selected_sensitive = st.selectbox(
            "Atribut Sensitif (untuk Fairness)",
            options=list(sensitive_attr_map.keys()),
            format_func=lambda x: sensitive_attr_map[x]
        )        
        
        if 'results' not in st.session_state:
            st.info("Upload data dulu di tab 'Upload CSV'")
        else:        

            if 'results' in st.session_state:
                results = st.session_state['results']
                df_full = results['df_full']  # seluruh df asli + age_group
                preds = results['preds']
                
                # Ambil kolom sensitif yang dipilih
                if selected_sensitive == 'age_group':
                    sensitive_series = df_full['age_group']
                else:
                    sensitive_series = df_full[selected_sensitive]

                # Pastikan ini kategori
                sensitive_series = sensitive_series.astype('category')

                # Jika ada label aktual
                if 'churn' in df_full.columns:
                    y_true = df_full['churn']
                    from sklearn.metrics import recall_score, confusion_matrix

                    fairness_data = []
                    for group in sensitive_series.cat.categories:
                        mask = (sensitive_series == group)
                        if mask.sum() == 0:
                            continue
                        y_true_g = y_true[mask]
                        preds_g = preds[mask]

                        recall = recall_score(y_true_g, preds_g, zero_division=0)
                        cm = confusion_matrix(y_true_g, preds_g)
                        if cm.size == 4:
                            tn, fp, fn, tp = cm.ravel()
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        else:
                            fpr = 0

                        fairness_data.append({
                            'Group': str(group),
                            'Recall (TPR)': recall,
                            'FPR': fpr,
                            'Count': int(mask.sum())
                        })

                    if fairness_data:
                        fairness_df = pd.DataFrame(fairness_data)
                        st.dataframe(fairness_df.round(3))

                        # Plot
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        ax[0].bar(fairness_df['Group'], fairness_df['Recall (TPR)'], color='green')
                        ax[0].set_title(f'Recall by {sensitive_attr_map[selected_sensitive]}')
                        ax[0].tick_params(axis='x', rotation=45)
                        ax[0].set_ylim(0, 1)

                        ax[1].bar(fairness_df['Group'], fairness_df['FPR'], color='red')
                        ax[1].set_title(f'False Positive Rate by {sensitive_attr_map[selected_sensitive]}')
                        ax[1].tick_params(axis='x', rotation=45)
                        ax[1].set_ylim(0, 1)

                        st.pyplot(fig)
                    else:
                        st.warning("No groups found for selected attribute.")
                else:
                    st.warning("Fairness analysis requires actual 'churn' labels in the dataset.")

                    # Peringatan disparitas
                    min_recall = fairness_df['Recall (TPR)'].min()
                    max_recall = fairness_df['Recall (TPR)'].max()
                    disparity_ratio = min_recall / max_recall if max_recall > 0 else 0

                    if disparity_ratio < 0.8:
                        st.error(f"⚠️ Significant disparity: Lowest recall ({min_recall:.2f}) is less than 80% of highest ({max_recall:.2f})")
                    else:
                        st.success("✅ Fairness acceptable: No major disparity in recall")