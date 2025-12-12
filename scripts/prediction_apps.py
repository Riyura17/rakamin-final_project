import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# === Konfigurasi ===
sensitive_cols = ['age', 'gender', 'education', 'marital_status', 'work_location']
model = joblib.load("best_model.pkl")
feature_engineering = joblib.load("feature_engineering.pkl")
overtime_median = feature_engineering["overtime_median"]

# === CSS Styling ===
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #173d3b !important;
        color: #FAFAFA;
    }
    [data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stSlider > label {
        color: #FFD166 !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        padding: 0 20px;
        background-color: #E0E0E0;
        color: black;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# === Feature Engineering ===
def engineer_features(df):
    df = df.copy()
    required_for_fe = {
        "marital_status", "working_hours_per_week", "overtime_hours_per_week",
        "target_achievement", "job_satisfaction", "distance_to_office_km",
        "company_tenure_years", "manager_support_score"
    }
    missing_fe = required_for_fe - set(df.columns)
    if missing_fe:
        raise ValueError(f"Missing for feature engineering: {missing_fe}")

    df["is_married"] = (df["marital_status"].str.strip().str.title() == "Married").astype(int)
    df["total_workload"] = df["working_hours_per_week"] + df["overtime_hours_per_week"]
    df["performance_efficiency"] = df["target_achievement"] / (df["total_workload"] + 1e-5)
    df["long_distance_overwork"] = df["distance_to_office_km"] / (df["working_hours_per_week"] + 1)
    df["low_satisfaction"] = (df["job_satisfaction"] <= 2).astype(int)
    high_overtime = (df["overtime_hours_per_week"] > overtime_median).astype(int)
    df["high_ot_low_sat"] = high_overtime & df["low_satisfaction"]

    if "age" in df.columns:
        q1, q2 = df['age'].quantile([0.33, 0.66])
        df['age_group'] = pd.cut(
            df['age'],
            bins=[df['age'].min()-1, q1, q2, df['age'].max()+1],
            labels=['Young', 'Middle', 'Senior']
        )
    else:
        df['age_group'] = None

    required_features = [
        'performance_efficiency', 'low_satisfaction', 'target_achievement',
        'high_ot_low_sat', 'working_hours_per_week', 'distance_to_office_km',
        'total_workload', 'job_satisfaction', 'long_distance_overwork',
        'company_tenure_years', 'manager_support_score', 'is_married'
    ]
    return df[required_features], df

# === Sidebar ===
st.sidebar.header("⚙️ Pengaturan")
mode = st.sidebar.radio("Input Mode:", ["Manual Input", "Upload File"])
threshold = st.sidebar.slider(
    "Threshold klasifikasi churn",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Lower threshold → more sensitive to churn"
)

if threshold > 0.8:
    st.sidebar.warning("Threshold terlalu tinggi\n\nModel tidak sensitif terhadap churn")
elif threshold < 0.2:
    st.sidebar.warning("Threshold terlalu rendah\n\nModel sangat sensitif terhadap churn")

# === Manual Input ===
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
        }])
        X, _ = engineer_features(df)
        prob = model.predict_proba(X)[:, 1].item()
        pred = int(prob >= threshold)
        st.subheader(f"Prediction: {'YES' if pred else 'NO'}")
        st.write(f"Probability: {prob:.2%}")

# === Upload File ===
else:
    st.title("Employee Churn Prediction — Batch Analysis")
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📊 Prediction Summary", "⚖️ Fairness Analysis"])

    # --- Tab 1: Upload & Search ---
    with tab1:
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        if uploaded_file:
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            
            required = {
                "target_achievement", "working_hours_per_week", "overtime_hours_per_week",
                "job_satisfaction", "distance_to_office_km", "company_tenure_years",
                "manager_support_score", "marital_status", "age", "churn"
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

                    # === Pencarian ID ===
                    id_column = "employee_id"
                    if id_column in result.columns:
                        st.markdown("### 🔍 Cari Karyawan Berdasarkan ID")
                        search_id = st.text_input(f"Masukkan {id_column}", placeholder="Masukkan ID")
                        if search_id:
                            filtered = result[result[id_column].astype(str).str.contains(search_id, case=False, na=False)]
                            st.dataframe(filtered if not filtered.empty else result.head(0))
                        else:
                            st.dataframe(result.head(10))
                    else:
                        st.warning(f"Kolom '{id_column}' tidak ditemukan.")
                        st.dataframe(result.head(10))

                    st.download_button(
                        "📥 Download Full Predictions",
                        result.to_csv(index=False),
                        "predictions_result.csv",
                        "text/csv",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    
            # Analisis Per ID
            st.markdown("### 🔍 Analisis Penyebab Churn per Karyawan")
            df_full = st.session_state['results']['df_full']
            id_column = "employee_id"
            if id_column in df_full.columns:
                search_id = st.text_input("Cari ID karyawan untuk analisis penyebab", placeholder="Masukkan ID...")
                if search_id:
                    mask = df_full[id_column].astype(str).str.contains(search_id, case=False, na=False)
                    if mask.any():
                        idx = df_full[mask].index[0]
                        emp_data = df_full.iloc[idx]
                        emp_pred = preds[idx]
                        emp_prob = probs[idx]
                        
                        if emp_pred: st.error(f"**PREDIKSI: CHURN** (Prob: {emp_prob:.2%})")
                        else: st.success(f"**PREDIKSI: TIDAK CHURN** (Prob: {emp_prob:.2%})")
                        
                        interpretable_features = [
                            'job_satisfaction', 'distance_to_office_km', 'working_hours_per_week',
                            'overtime_hours_per_week', 'manager_support_score', 'company_tenure_years',
                            'is_married', 'target_achievement'
                        ]
                        extremes = {}
                        for feat in interpretable_features:
                            if feat in df_full.columns:
                                dev = abs(emp_data[feat] - df_full[feat].median())
                                extremes[feat] = (emp_data[feat], dev)
                        top3 = sorted(extremes.items(), key=lambda x: x[1][1], reverse=True)[:3]

                        st.markdown("### 3 Faktor Utama:")
                        label_map = {
                            'job_satisfaction': 'Job Satisfaction (1-4)',
                            'distance_to_office_km': 'Distance to Office (km)',
                            'working_hours_per_week': 'Working Hours/Week',
                            'overtime_hours_per_week': 'Overtime Hours/Week',
                            'manager_support_score': 'Manager Support Score (1-4)',
                            'company_tenure_years': 'Company Tenure (years)',
                            'is_married': 'Marital Status',
                            'target_achievement': 'Target Achievement'
                        }
                        for i, (feat, (val, _)) in enumerate(top3, 1):
                            name = label_map.get(feat, feat)
                            val_str = "Married" if feat == 'is_married' and val == 1 else ("Single" if feat == 'is_married' else f"{val:.2f}")
                            st.markdown(f"**{i}. {name}** = {val_str}")
                    else:
                        st.warning("ID tidak ditemukan.")

    # --- Tab 2: Prediction Summary ---
    with tab2:
        if 'results' not in st.session_state:
            st.info("Upload data dulu di tab 'Upload Data'")
        else:
            results = st.session_state['results']
            probs = results['probs']
            preds = results['preds']
            churn_count = preds.sum()
            non_churn_count = len(preds) - churn_count

            st.markdown("<h5>Key Metrics</h5>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            card_style = """
            <div style="background:{bg}; color:white; padding:8px 4px; border-radius:20px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                <div style="font-size:min(4.5vw, 20px); font-weight:bold; margin-bottom:4px;">{label}</div>
                <div style="font-size:min(7vw, 32px); font-weight:800;">{value}</div>
            </div>
            """
            with col1: st.markdown(card_style.format(bg="#854ecd", label="Total Employees", value=len(probs)), unsafe_allow_html=True)
            with col2: st.markdown(card_style.format(bg="#cd4e57", label="Churn", value=churn_count), unsafe_allow_html=True)
            with col3: st.markdown(card_style.format(bg="#96cd4e", label="Not Churn", value=non_churn_count), unsafe_allow_html=True)
            with col4: st.markdown(card_style.format(bg="#4ecdc4", label="Avg. Churn Probs", value=f"{np.mean(probs):.2%}"), unsafe_allow_html=True)

            st.markdown("<br><h5>Churn Distribution</h5>", unsafe_allow_html=True)
            col_pie, col_hist = st.columns([1, 2])

            with col_pie:
                fig, ax = plt.subplots(figsize=(3.5, 3))
                fig.patch.set_facecolor('none')
                ax.patch.set_facecolor('none')                
                ax.pie(
                    [churn_count, non_churn_count],
                    labels=['Churn', 'Not Churn'],
                    autopct='%1.1f%%',
                    colors=['#cd4e57', '#4ECDC4'],
                    textprops={'fontsize': 8, 'color': 'white'})
                ax.set_title('Churn Ratio', fontsize=9, pad=5, color='white')
                st.pyplot(fig)

            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor('#E0E0E0')
                ax.patch.set_facecolor('#E0E0E0')
                ax.hist(probs, bins=20, color='#4ECDC4', edgecolor='black')
                ax.set_title('Distribution of Predicted Probabilities', fontsize=10)
                ax.set_xlabel('Churn Probability')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                st.success("Persebaran probabilitas churn berdasarkan hasil prediksi model.")

    # --- Tab 3: Fairness Analysis ---
    with tab3:
        st.subheader("⚖️ Fairness Analysis by Group")
        st.markdown("Analisis metrik utama model berdasarkan atribut sensitif untuk mendeteksi potensi bias pada model <b>(monitoring model)</b>.", unsafe_allow_html=True)
        
        if 'results' not in st.session_state:
            st.info("Upload data dulu di tab 'Upload Data'")
        else:
            sensitive_attr_map = {
                'age_group': 'Age Group',
                'gender': 'Gender',
                'education': 'Education Level',
                'marital_status': 'Marital Status',
                'work_location': 'Work Location'
            }
            selected_sensitive = st.selectbox("Atribut Sensitif", options=list(sensitive_attr_map.keys()), format_func=lambda x: sensitive_attr_map[x])
            results = st.session_state['results']
            df_full = results['df_full']
            preds = results['preds']

            if selected_sensitive == 'age_group':
                sensitive_series = df_full['age_group']
            else:
                sensitive_series = df_full[selected_sensitive]
            sensitive_series = sensitive_series.astype('category')

            if 'churn' not in df_full.columns:
                st.warning("Fairness analysis requires 'churn' labels.")
            else:
                from sklearn.metrics import recall_score, confusion_matrix
                fairness_data = []
                for group in sensitive_series.cat.categories:
                    mask = (sensitive_series == group)
                    if mask.sum() == 0: continue
                    y_true_g = df_full.loc[mask, 'churn']
                    preds_g = preds[mask]
                    recall = recall_score(y_true_g, preds_g, zero_division=0)
                    cm = confusion_matrix(y_true_g, preds_g)
                    fpr = fp / (fp + tn) if (cm.size == 4 and (fp := cm[0,1]) + (tn := cm[0,0]) > 0) else 0
                    fairness_data.append({'Group': str(group), 'Recall (TPR)': recall, 'FPR': fpr, 'Count': int(mask.sum())})

                if fairness_data:
                    fairness_df = pd.DataFrame(fairness_data)
                    st.dataframe(fairness_df.round(3))
                    
                    # Evaluasi recall minimum
                    if fairness_df['Recall (TPR)'].min() < 0.8:
                        st.error(f"⚠️ Critical: One group has recall < 80% (min = {fairness_df['Recall (TPR)'].min():.3f})")
                    else:
                        st.success("✅ All groups have recall ≥ 80%")
                        
                    # Visualisasi
                    st.markdown("<h5>📈 Performance by Group</h5>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        fig.patch.set_facecolor('#E0E0E0')
                        ax.patch.set_facecolor('#E0E0E0')
                        ax.bar(fairness_df['Group'], fairness_df['Recall (TPR)'], color='#4ECDC4', ec='black')
                        ax.set_title(f'Recall by {sensitive_attr_map[selected_sensitive]}', fontsize=9)
                        ax.set_ylim(0, 1)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        st.pyplot(fig)
                    with col2:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        fig.patch.set_facecolor('#E0E0E0')
                        ax.patch.set_facecolor('#E0E0E0')
                        ax.bar(fairness_df['Group'], fairness_df['FPR'], color='#FF6B6B', ec='black')
                        ax.set_title(f'FPR by {sensitive_attr_map[selected_sensitive]}', fontsize=9)
                        ax.set_ylim(0, 1)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        st.pyplot(fig)

                    # Disparitas Recall
                    r_vals = fairness_df['Recall (TPR)']
                    min_r, max_r = r_vals.min(), r_vals.max()
                    if min_r > 0 and max_r / min_r >= 1.5:
                        st.error(f"⚠️ Significant recall disparity: {max_r:.3f} is {max_r/min_r:.3f}× {min_r:.3f}")
                    else:
                        st.success("✅ Recall disparity acceptable (ratio < 1.5)")

                    # Disparitas FPR
                    f_vals = fairness_df['FPR']
                    min_f, max_f = f_vals.min(), f_vals.max()
                    if min_f > 0 and max_f / min_f >= 1.5:
                        st.error(f"⚠️ Significant FPR disparity: {max_f:.3f} is {max_f/min_f:.3f}× {min_f:.3f}")
                    else:
                        st.success("✅ FPR disparity acceptable (ratio < 1.5)")
                else:
                    st.warning("No data for selected group.")