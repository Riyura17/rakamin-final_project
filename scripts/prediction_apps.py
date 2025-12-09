import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sensitive_cols = ['age', 'gender', 'education', 'marital_status', 'work_location']

# === Load model & resources ===
model = joblib.load("best_model.pkl")

feature_engineering = joblib.load("feature_engineering.pkl")
overtime_median = feature_engineering["overtime_median"]

st.markdown("""
<style>
    /* Sidebar styling */
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
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #2D2D3F !important;
        color: #FAFAFA !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #1E1E2F !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6C757D !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
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

# === Feature engineering ===
def engineer_features(df):
    df = df.copy()
    
    required_for_fe = {
        "marital_status",
        "working_hours_per_week",
        "overtime_hours_per_week",
        "target_achievement",
        "job_satisfaction",
        "distance_to_office_km",
        "company_tenure_years",
        "manager_support_score"
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

    # Buat age_group dari kolom 'age'
    if "age" in df.columns:
        q1 = df['age'].quantile(0.33)
        q2 = df['age'].quantile(0.66)
        df['age_group'] = pd.cut(
            df['age'],
            bins=[df['age'].min()-1, q1, q2, df['age'].max()+1],
            labels=['Young', 'Middle', 'Senior']
        )
    else:
        df['age_group'] = None

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
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Lower threshold → more sensitive to churn"
)

if threshold > 0.8:
    st.sidebar.warning(
        f"Threshold terlalu tinggi"
        f"\n\nIni akan mempengaruhi hasil model tidak sensitif terhadap churn"
        )
elif threshold < 0.2:
    st.sidebar.warning(
        f"Threshold terlalu rendah"
        f"\n\nIni akan mempengaruhi hasil model sangat sensitif terhadap churn"
        )
pass

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

# --- Upload file ---
else:
    st.title("Employee Churn Prediction — Batch Analysis")
    
    # Buat tab hanya untuk mode upload
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "📊 Prediction Summary", "⚖️ Fairness Analysis"])
    
# --- Tab 1: Upload Data ---
    with tab1:
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
                
            sensitive_cols = ['age', 'gender', 'education', 'marital_status', 'work_location']
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

                    # kolom pencarian ID 
                    st.markdown("🔍 Cari Karyawan Berdasarkan ID")
                    id_column = "employee_id"
                    
                    if id_column not in result.columns:
                        st.warning(f"Kolom '{id_column}' tidak ditemukan. Pastikan file upload berisi kolom ID.")
                    else:
                        search_id = st.text_input(f"Masukkan {id_column}", placeholder="Masukkan ID karyawan untuk mencari...")
                        st.text(f"Menampilkan {len(result[result[id_column].astype(str).str.contains(search_id, case=False, na=False)])} hasil prediksi untuk karyawan dengan ID yang mengandung '{search_id}'")
                        
                        if search_id:
                            filtered = result[result[id_column].astype(str).str.contains(search_id, case=False, na=False)]
                            if filtered.empty:
                                st.warning(f"Tidak ada karyawan dengan ID yang mengandung '{search_id}'")
                            else:
                                st.dataframe(filtered)
                        else:
                            st.dataframe(result.head(10))
                            
                    st.download_button(
                        "📥 Download Full Predictions",
                        result.to_csv(index=False),
                        "predictions_result.csv",
                        "text/csv",
                    )
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    
            # === Search ID Section ===
            st.markdown("---")
            st.markdown("<h4>🔍 Analisis Penyebab Churn per Karyawan</h4>", unsafe_allow_html=True)
            
            id_column = "employee_id"
            df_full = st.session_state['results']['df_full']
            
            if id_column not in df_full.columns:
                st.warning(f"Kolom '{id_column}' tidak ditemukan. Analisis per ID tidak tersedia.")
            else:
                search_id = st.text_input(f"Masukkan {id_column} untuk lihat penyebab churn", 
                                        placeholder="Masukkan ID karyawan...")
                
                if search_id:
                    mask = df_full[id_column].astype(str).str.contains(search_id, case=False, na=False)
                    if mask.sum() == 0:
                        st.warning(f"ID '{search_id}' tidak ditemukan")
                    else:
                        idx = df_full[mask].index[0]
                        
                        emp_data = df_full.iloc[idx]
                        emp_pred = preds[idx]
                        emp_prob = probs[idx]
                        
                        st.markdown(f"ID Karyawan: **{emp_data[id_column]}**")
                        if emp_pred == 1:
                            st.error(f"**PREDIKSI: CHURN** (Probabilitas: {emp_prob:.2%})")
                        else:
                            st.success(f"**PREDIKSI: TIDAK CHURN** (Probabilitas: {emp_prob:.2%})")
                        
                        # Ambil 3 fitur paling ekstrem
                        interpretable_features = [
                            'job_satisfaction',
                            'distance_to_office_km',
                            'working_hours_per_week',
                            'overtime_hours_per_week',
                            'manager_support_score',
                            'company_tenure_years',
                            'is_married',
                            'target_achievement'
                        ]
                        
                        available_features = [f for f in interpretable_features if f in emp_data.index]
                        extremes = {}
                        for feat in available_features:
                            if feat in df_full.columns:
                                median_val = df_full[feat].median()
                                current_val = emp_data[feat]
                                deviation = abs(current_val - median_val)
                                extremes[feat] = (current_val, deviation)
                        
                        top3 = sorted(extremes.items(), key=lambda x: x[1][1], reverse=True)[:3]
                        
                        st.markdown("3 Faktor Utama Hasil Prediksi:")
                        for i, (feat, (val, _)) in enumerate(top3, 1):
                            readable_name = {
                                'job_satisfaction': 'Job Satisfaction (1-4)',
                                'distance_to_office_km': 'Distance to Office (km)',
                                'working_hours_per_week': 'Working Hours/Week',
                                'overtime_hours_per_week': 'Overtime Hours/Week',
                                'manager_support_score': 'Manager Support Score (1-4)',
                                'company_tenure_years': 'Company Tenure (years)',
                                'is_married': 'Marital Status',
                                'target_achievement': 'Target Achievement'
                            }.get(feat, feat)
                            
                            if feat == 'is_married':
                                val_str = "Married" if val == 1 else "Single"
                            else:
                                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                            
                            st.markdown(f"**{i}. {readable_name}** = {val_str}")
                            
    # --- Tab 2: Prediction Summary ---
    with tab2:
        st.subheader("📊 Prediction Summary")
        
        if 'results' not in st.session_state:
            st.info("Upload data dulu di tab 'Upload CSV'")
        else:
            results = st.session_state['results']
            probs = results['probs']
            preds = results['preds']
            
            churn_count = preds.sum()
            non_churn_count = len(preds) - churn_count
            
            # === Card Metrics & Pie Chart ===
            st.markdown("<h5>Key Metrics</h5>", unsafe_allow_html=True)
            
            col1, col2, col3, col4= st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div style="background:#854ecd; color:white; padding:8px; border-radius:15px; text-align:center;">
                    <text style= "font-size:25px;">Total Employees</text>
                    <br>
                    <text style= "font-size:45px;"><b>{len(probs)}</b></text>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background:#cd4e57; color:white; padding:8px; border-radius:15px; text-align:center;">
                    <text style= "font-size:25px;">Churn</text>
                    <br>
                    <text style= "font-size:45px;"><b>{churn_count}</b></text>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="background:#96cd4e; color:white; padding:8px; border-radius:15px; text-align:center;">
                    <text style= "font-size:25px;">Not Churn</text>
                    <br>
                    <text style= "font-size:45px;"><b>{non_churn_count}</b></text>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                avg_prob = np.mean(probs)
                st.markdown(f"""
                <div style="background:#4ecdc4; color:white; padding:8px; border-radius:15px; text-align:center;">
                    <text style= "font-size:25px;">Avg. Churn Probability</text>
                    <br>
                    <text style= "font-size:45px;"><b>{avg_prob:.2%}</b></text>
                </div>
                """, unsafe_allow_html=True)
                
            # Pie Chart di card terpisah
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
                    textprops={'fontsize': 8, 'color': 'white'}
                )
                
                ax.set_title('Churn Ratio', fontsize=9, pad=5, color='white')
                plt.tight_layout()
                st.pyplot(fig)
                
            with col_hist:
                fig, ax = plt.subplots(figsize=(6, 3))
                
                # Atur transparansi untuk Figure
                fig.patch.set_facecolor('#E0E0E0')
                ax.patch.set_facecolor('#E0E0E0')
                
                sns.histplot(
                    x=probs,
                    bins=20,
                    color='#4ECDC4',
                    edgecolor='black',
                    ax=ax,
                    kde=True
                )
                
                ax.set_xlabel('Churn Probability', fontsize=10, color='black')
                ax.set_ylabel('Count', fontsize=10, color='black')
                ax.set_title('Distribution of Predicted Probabilities', fontsize=10, color='black')
                
                ax.tick_params(axis='x', colors='black')
                ax.tick_params(axis='y', colors='black')
                
                sns.despine(fig=fig, ax=ax)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.success("Persebaran probabilitas churn berdasarkan hasil prediksi model.")   
                
    # --- Tab 3: Fairness Analysis ---
    with tab3:
        st.subheader("⚖️ Fairness Analysis by Group")
        st.markdown("Analisis metrik utama model berdasarkan atribut sensitif untuk mendeteksi potensi bias pada model <b>(monitoring model)</b>.", unsafe_allow_html=True)
        sensitive_attr_map = {
            'age_group': 'Age Group',
            'gender': 'Gender',
            'education': 'Education Level',
            'marital_status': 'Marital Status',
            'work_location': 'Work Location'
        }

        
        if 'results' not in st.session_state:
            st.info("Upload data dulu di tab 'Upload CSV'")
        else:
            
            selected_sensitive = st.selectbox(
                "Atribut Sensitif (untuk Fairness)",
                options=list(sensitive_attr_map.keys()),
                format_func=lambda x: sensitive_attr_map[x]
            )
            
            if 'results' in st.session_state:
                results = st.session_state['results']
                df_full = results['df_full']  # seluruh df asli + age_group
                preds = results['preds']
                
                if selected_sensitive == 'age_group':
                    sensitive_series = df_full['age_group']
                else:
                    sensitive_series = df_full[selected_sensitive]
                    
                sensitive_series = sensitive_series.astype('category')
                st.markdown(f"### Fairness Analysis by {sensitive_attr_map[selected_sensitive]}")
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

                        # === Chart Recall & FPR dalam layout dua kolom ===
                        st.markdown("<h5>📈 Performance by Group</h5>", unsafe_allow_html=True)
                        col_recall, col_fpr = st.columns(2)

                        with col_recall:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            fig.patch.set_facecolor('#E0E0E0')
                            ax.patch.set_facecolor('#E0E0E0')                            
                            ax.bar(
                                fairness_df['Group'], 
                                fairness_df['Recall (TPR)'], 
                                color='#4ECDC4',
                                edgecolor='black'
                            )
                            ax.set_title(f'Recall by {sensitive_attr_map[selected_sensitive]}', fontsize=9)
                            ax.set_ylim(0, 1)
                            ax.tick_params(axis='x', rotation=45, labelsize=8)
                            ax.set_ylabel('Recall (TPR)', fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)

                        with col_fpr:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            fig.patch.set_facecolor('#E0E0E0')
                            ax.patch.set_facecolor('#E0E0E0')                            
                            ax.bar(
                                fairness_df['Group'], 
                                fairness_df['FPR'], 
                                color='#FF6B6B',
                                edgecolor='black'
                            )
                            ax.set_title(f'FPR by {sensitive_attr_map[selected_sensitive]}', fontsize=9)
                            ax.set_ylim(0, 1)
                            ax.tick_params(axis='x', rotation=45, labelsize=8)
                            ax.set_ylabel('False Positive Rate', fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                    else:
                        st.warning("No groups found for selected attribute.")
                else:
                    st.warning("Fairness analysis requires actual 'churn' labels in the dataset.")
                    
                # Peringatan disparitas
                
                if fairness_df['Recall (TPR)'].min() < 0.8:
                    st.error(f"⚠️ Critical: One group has recall < 80% (min = {fairness_df['Recall (TPR)'].min():.2f})")
                else:
                    st.success("✅ All groups have recall ≥ 80%")         
                    
                # Recall
                recall_vals = fairness_df['Recall (TPR)']
                min_r, max_r = recall_vals.min(), recall_vals.max()
                if min_r > 0 and max_r / min_r >= 1.5:
                    st.error(f"⚠️ Significant recall disparity: Highest ({max_r:.3f}) is {max_r/min_r:.1f}× lowest ({min_r:.3f})")
                elif min_r == 0:
                    st.warning("⚠️ At least one group has 0 recall — fairness cannot be assessed reliably")
                else:
                    st.success("✅ Recall disparity acceptable (ratio < 1.5)")
                    
                # FPR
                fpr_vals = fairness_df['FPR']
                min_f, max_f = fpr_vals.min(), fpr_vals.max()
                if min_f > 0 and max_f / min_f >= 1.5:
                    st.error(f"⚠️ Significant FPR disparity: Highest ({max_f:.3f}) is {max_f/min_f:.1f}× lowest ({min_f:.3f})")
                elif min_f == 0 and max_f > 0:
                    st.warning(f"⚠️ Extreme FPR disparity: Some groups have 0 FPR, others have {max_f:.3f}")
                else:
                    st.success("✅ FPR disparity acceptable (ratio < 1.5)")               