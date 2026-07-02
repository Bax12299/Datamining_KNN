import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math

st.set_page_config(
    page_title="Prediksi Diabetes Dini - KNN",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #FFFFFF; }

section[data-testid="stSidebar"] {
    background: #D2DCB6;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #2D3A2A !important; }
section[data-testid="stSidebar"] .stRadio label {
    color: #2D3A2A !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}
section[data-testid="stSidebar"] p { color: #2D3A2A !important; }

.page-header {
    background: #A1BC98;
    border-radius: 16px; padding: 40px 48px; margin-bottom: 32px;
    border-bottom: 6px solid #778873;
    color: #FFFFFF;
}
.page-header h1 { font-family: 'DM Serif Display', serif; color: #ffffff; font-size: 2.4rem; margin: 0 0 8px 0; font-weight: 400; line-height: 1.2; }
.page-header p { color: #F1F3E0; font-size: 1rem; margin: 0; font-weight: 500; }

.section-card { background: #ffffff; border-radius: 12px; padding: 28px; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(119, 136, 115, 0.08); margin-bottom: 20px; }
.section-title { font-size: 1rem; font-weight: 600; color: #778873; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 2px solid #D2DCB6; }

div[data-testid="column"] button {
    background-color: transparent !important; border: none !important; color: #2D3A2A !important; font-weight: 600; font-size: 0.95rem; padding: 10px 16px; border-radius: 8px !important; transition: all 0.2s ease; box-shadow: none !important; text-align: center;
}
div[data-testid="column"] button:hover { background-color: #F1F3E0 !important; color: #778873 !important; transform: translateY(-1px); }

.quiz-shell { max-width: 720px; margin: 0 auto; }
.quiz-progress-bar-wrap { background: #E2E8F0; border-radius: 99px; height: 6px; margin-bottom: 10px; }
.quiz-progress-bar-fill { background: linear-gradient(90deg, #A1BC98, #778873); border-radius: 99px; height: 6px; transition: width 0.4s ease; }
.quiz-step-label { font-size: 0.78rem; color: #778873; text-align: right; margin-bottom: 28px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.quiz-card { background: #ffffff; border-radius: 20px; padding: 40px 48px 32px 48px; border: 1px solid #E2E8F0; box-shadow: 0 4px 24px rgba(161, 188, 152, 0.15); margin-bottom: 24px; }
.quiz-card:hover { background: #FDFFF9; border-color: #A1BC98; box-shadow: 0 8px 32px rgba(161, 188, 152, 0.25); transform: translateY(-2px); transition: all 0.2s ease-in-out; }
.quiz-question { font-size: 1.55rem; font-weight: 700; color: #2D3A2A; line-height: 1.4; margin-bottom: 8px; }

.stRadio > div { display: flex; flex-direction: column; gap: 8px; }
.stRadio div[data-testid="stMarkdownContainer"] p { font-size: 1rem; font-weight: 500; color: #2D3A2A; }

.result-hero { border-radius: 20px; padding: 48px; text-align: center; margin-bottom: 28px; position: relative; overflow: hidden; }
.result-hero-positive { background: linear-gradient(135deg, #FFF1F2 0%, #FEE2E2 100%); border: 2px solid #FCA5A5; }
.result-hero-negative { background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border: 2px solid #86EFAC; }
.result-icon { font-size: 4rem; margin-bottom: 16px; }
.result-title-positive { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #991B1B; margin-bottom: 8px; font-weight: 400; }
.result-title-negative { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #14532D; margin-bottom: 8px; font-weight: 400; }
.result-subtitle { font-size: 1rem; color: #475569; line-height: 1.7; max-width: 520px; margin: 0 auto; }
.result-detail-card { background: #ffffff; border-radius: 16px; padding: 28px 32px; border: 1px solid #E2E8F0; box-shadow: 0 2px 12px rgba(119, 136, 115, 0.05); margin-bottom: 20px; }
.result-detail-title { font-size: 0.82rem; font-weight: 600; color: #778873; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 16px; }
.symptom-chip { display: inline-block; padding: 6px 14px; border-radius: 99px; font-size: 0.82rem; font-weight: 500; margin: 4px; }
.chip-yes { background: #FFE4E6; color: #9F1239; }
.chip-no  { background: #E6F7EC; color: #166534; }

.stButton > button { 
    background: linear-gradient(135deg, #A1BC98, #778873);
    color: white; border: none; border-radius: 8px; 
    padding: 10px 28px; font-weight: 600; font-size: 0.9rem; 
    letter-spacing: 0.02em; transition: all 0.2s ease; 
    width: 100%; 
}
.stButton > button:hover { opacity: 0.85; transform: scale(1.02); }
hr { border: none; border-top: 1px solid #E2E8F0; margin: 20px 0; }

button[kind="primary"][key="start_quiz"] {
    width: 250px !important;
    position: relative !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    margin-top: 15px !important;
    margin-bottom: 15px !important;
}
button[kind="primary"][key="restart_quiz"] {
    width: 250px !important;
    position: relative !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    margin-top: 15px !important;
    margin-bottom: 15px !important;
}
</style>
""", unsafe_allow_html=True)

QUESTIONS = [
    ("age", "", "Berapa usia Anda saat ini?"),
    ("Gender", "", "Apa jenis kelamin Anda?", "", [("Laki-laki", "Male"), ("Perempuan", "Female")]),
    ("Polyuria", "", "Apakah Anda merasa sering kencing?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Polydipsia", "", "Apakah Anda merasa haus berlebihan?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Sudden_weight_loss", "", "Apakah Anda mengalami penurunan berat badan secara tiba-tiba?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Weakness", "", "Apakah Anda mengalami kelelahan secara berlebihan?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Polyphagia", "", "Apakah Anda mengalami rasa lapar yang berlebihan?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Genital_thrush", "", "Apakah Anda mengalami rasa gatal berkepanjangan di area sekitar alat kelamin?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Visual_blurring", "", "Apakah Anda mengalami pandangan yang mengabur?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Itching", "", "Apakah Anda mengalami gatal berlebih pada kulit?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Irritability", "", "Apakah Anda mengalami suasana hati yang tidak stabil, mudah marah, atau sensitif?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Delayed_healing", "", "Apakah Anda mengalami luka yang lama sembuh?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Partial_paresis", "", "Apakah Anda mengalami kelemahan otot sebagian?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Muscle_stiffness", "", "Apakah Anda mengalami kondisi otot yang terasa kaku atau tidak fleksibel?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Alopecia", "", "Apakah Anda mengalami rontok rambut berlebih?", "", [("Ya", "Yes"), ("Tidak", "No")]),
    ("Obesity", "", "Apakah Anda mengalami berat badan yang berlebihan (obesitas)?", "", [("Ya", "Yes"), ("Tidak", "No")]),
]

TOTAL_Q = len(QUESTIONS)

if "quiz_step" not in st.session_state: st.session_state.quiz_step = 0
if "quiz_answers" not in st.session_state: st.session_state.quiz_answers = {}
if "quiz_started" not in st.session_state: st.session_state.quiz_started = False
if "quiz_result" not in st.session_state: st.session_state.quiz_result = None
if "active_tab" not in st.session_state: st.session_state.active_tab = "Skrining Risiko Diabetes"

@st.cache_data
def load_data():
    df = pd.read_excel("Dataset 2 _ Early-stage diabetes risk prediction dataset (ESDRPD) (4).xlsx")
    return df

@st.cache_data
def encode_data(df):
    df_enc = df.copy()
    for col in df_enc.columns:
        if col == "Age": continue
        df_enc[col] = df_enc[col].map({"Male": 1, "Female": 0, "Yes": 1, "No": 0, "Positive": 1, "Negative": 0})
    return df_enc

def euclidean_distance(a, b): return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn_predict(X_train, y_train, X_test_row, k):
    distances = []
    test_vec = list(X_test_row)
    for idx, train_row in enumerate(X_train):
        dist = euclidean_distance(test_vec, list(train_row))
        distances.append((dist, int(y_train[idx]), idx))
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    labels = [n[1] for n in k_neighbors]
    prediction = 1 if sum(labels) > k / 2 else 0
    return prediction, k_neighbors

def run_knn_full(X_train, y_train, X_test, y_test, k):
    predictions, all_neighbors = [], []
    for row in X_test:
        pred, neighbors = knn_predict(X_train, y_train, row, k)
        predictions.append(pred)
        all_neighbors.append(neighbors)
    return np.array(predictions), all_neighbors

def compute_metrics(y_true, y_pred):
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "Specificity": spec}

with st.sidebar:
    st.markdown("### Tentang Aplikasi")
    st.markdown("Prediksi risiko diabetes dini menggunakan algoritma K-Nearest Neighbors")
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#2D3A2A;'>Dataset: Early-Stage Diabetes Risk Prediction<br>Sumber: UCI Machine Learning Repository<br>Total Sampel: 520 pasien</p>", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df_raw = load_data()
    df_enc = encode_data(df_raw)
    
    feature_cols = [c for c in df_enc.columns if c != "Class"]
    X_all = df_enc[feature_cols].values.astype(float)
    y_all = df_enc["Class"].values.astype(int)
    
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, np.arange(len(y_all)), test_size=0.2, random_state=42, stratify=y_all
    )
    
    scaler = MinMaxScaler()
    X_train_norm = X_train_raw.copy()
    X_test_norm  = X_test_raw.copy()
    X_train_norm[:, 0] = scaler.fit_transform(X_train_raw[:, 0].reshape(-1, 1)).flatten()
    X_test_norm[:, 0]  = scaler.transform(X_test_raw[:, 0].reshape(-1, 1)).flatten()
    
    return df_raw, df_enc, feature_cols, X_train_norm, X_test_norm, y_train, y_test, idx_train, idx_test, scaler

df_raw, df_enc, feature_cols, X_train_norm, X_test_norm, y_train, y_test, idx_train, idx_test, scaler = load_and_process_data()

@st.cache_data
def find_best_k(_X_train_norm, _y_train, _X_test_norm, _y_test):
    k_range = list(range(1, 22, 2))
    acc_list = []
    for kk in k_range:
        preds, _ = run_knn_full(_X_train_norm, _y_train, _X_test_norm, _y_test, kk)
        acc_list.append(compute_metrics(_y_test, preds)["Accuracy"])
    
    k_range_filtered = [k for k in k_range if k >= 3]
    acc_list_filtered = [acc_list[i] for i, k in enumerate(k_range) if k >= 3]
    
    max_acc = max(acc_list_filtered)
    best_k_candidates = [k_range_filtered[i] for i, acc in enumerate(acc_list_filtered) if acc == max_acc]
    best_k = max(best_k_candidates)
    
    return best_k

if "best_k" not in st.session_state:
    st.session_state.best_k = find_best_k(X_train_norm, y_train, X_test_norm, y_test)

best_k = st.session_state.best_k

st.markdown("""
<div class="page-header">
    <h1>Prediksi Dini Penyakit Diabetes</h1>
    <p>Klasifikasi risiko diabetes menggunakan algoritma K-Nearest Neighbors (KNN)</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    if st.button("Skrining Risiko Diabetes", use_container_width=True): 
        st.session_state.active_tab = "Skrining Risiko Diabetes"
with c2:
    if st.button("Evaluasi Model", use_container_width=True): 
        st.session_state.active_tab = "Evaluasi Model"
with c3:
    if st.button("Eksplorasi Dataset", use_container_width=True): 
        st.session_state.active_tab = "Eksplorasi Dataset"
with c4:
    if st.button("Insight Diabetes Indonesia", use_container_width=True): 
        st.session_state.active_tab = "Insight Diabetes Indonesia"

st.markdown("---")

active_tab = st.session_state.active_tab

if active_tab == "Skrining Risiko Diabetes":
    step = st.session_state.quiz_step

    if not st.session_state.quiz_started:
        st.markdown('<div class="quiz-shell">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="quiz-card" style="text-align:center;">
            <div style="font-size:3.5rem;margin-bottom:20px;">🩺</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#0F172A;margin-bottom:12px;font-weight:400;">Skrining Risiko Diabetes Dini</div>
            <div style="font-size:0.95rem;color:#475569;line-height:1.8;max-width:500px;margin:0 auto 32px auto;">
                Kuis ini terdiri dari <strong>16 pertanyaan klinis</strong>.
                Jawaban Anda akan dianalisis menggunakan algoritma <strong>K-Nearest Neighbors (KNN)</strong>.
            </div>
            <div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:14px 20px;font-size:0.83rem;color:#92400E;margin-bottom:0;text-align:left;max-width:480px;margin:0 auto;">
                Hasil skrining ini bersifat indikatif dan tidak menggantikan diagnosis klinis oleh tenaga medis profesional.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Mulai Skrining →", key="start_quiz"):
            st.session_state.quiz_started = True; st.session_state.quiz_step = 1; st.session_state.quiz_answers = {}; st.rerun()

    elif step > TOTAL_Q:
        ans = st.session_state.quiz_answers; enc_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        col_map = {
            "Age": ans.get("age", 45),
            "Gender": enc_map.get(ans.get("Gender", "Male"), 1),
            "Polyuria": enc_map.get(ans.get("Polyuria", "No"), 0),
            "Polydipsia": enc_map.get(ans.get("Polydipsia", "No"), 0),
            "Sudden weight loss": enc_map.get(ans.get("Sudden_weight_loss", "No"), 0),
            "Weakness": enc_map.get(ans.get("Weakness", "No"), 0),
            "Polyphagia": enc_map.get(ans.get("Polyphagia", "No"), 0),
            "Genital thrush": enc_map.get(ans.get("Genital_thrush", "No"), 0),
            "Visual blurring": enc_map.get(ans.get("Visual_blurring", "No"), 0),
            "Itching": enc_map.get(ans.get("Itching", "No"), 0),
            "Irritability": enc_map.get(ans.get("Irritability", "No"), 0),
            "Delayed healing": enc_map.get(ans.get("Delayed_healing", "No"), 0),
            "Partial paresis": enc_map.get(ans.get("Partial_paresis", "No"), 0),
            "Muscle stiffness": enc_map.get(ans.get("Muscle_stiffness", "No"), 0),
            "Alopecia": enc_map.get(ans.get("Alopecia", "No"), 0),
            "Obesity": enc_map.get(ans.get("Obesity", "No"), 0),
        }
        patient = [col_map[c] for c in feature_cols]
        patient_norm = patient.copy()
        patient_norm[0] = scaler.transform([[patient[0]]])[0][0]

        pred, neighbors = knn_predict(X_train_norm, y_train, patient_norm, best_k)

        positive_symptoms = [k for k, v in ans.items() if v == "Yes" and k not in ("age", "Gender")]
        negative_symptoms = [k for k, v in ans.items() if v == "No" and k not in ("age", "Gender")]
        symptom_display = {
            "Polyuria": "Sering Kencing", "Polydipsia": "Haus Berlebihan",
            "Sudden_weight_loss": "Penurunan BB Mendadak", "Weakness": "Kelelahan Berlebih",
            "Polyphagia": "Lapar Berlebih", "Genital_thrush": "Gatal pada Kemaluan",
            "Visual_blurring": "Pandangan Kabur", "Itching": "Gatal Kulit",
            "Irritability": "Mudah Marah", "Delayed_healing": "Luka Lambat Sembuh",
            "Partial_paresis": "Kelemahan Otot Parsial", "Muscle_stiffness": "Kekakuan Otot",
            "Alopecia": "Kerontokan Rambut", "Obesity": "Obesitas"
        }

        label_counts = {"Positif": 0, "Negatif": 0}
        for _, lbl, _ in neighbors:
            if lbl == 1: label_counts["Positif"] += 1
            else: label_counts["Negatif"] += 1
        vote_conf = max(label_counts.values()) / best_k * 100

        if pred == 1:
            st.markdown(f"""
            <div class="result-hero result-hero-positive">
                <div class="result-icon">⚠️</div>
                <div class="result-title-positive">Terindikasi Risiko Diabetes</div>
                <div class="result-subtitle">Berdasarkan analisis KNN, profil gejala Anda memiliki kemiripan tinggi dengan pasien yang terdiagnosis <strong>positif diabetes</strong>. Tingkat keyakinan model: <strong>{vote_conf:.0f}%</strong>.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-hero result-hero-negative">
                <div class="result-icon">✅</div>
                <div class="result-title-negative">Risiko Rendah Terdeteksi</div>
                <div class="result-subtitle">Berdasarkan analisis KNN, profil gejala Anda lebih sesuai dengan pasien yang terdiagnosis <strong>negatif diabetes</strong>. Tingkat keyakinan model: <strong>{vote_conf:.0f}%</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"**Gejala dirasakan ({len(positive_symptoms)} Gejala):**" if positive_symptoms else "", unsafe_allow_html=False)
            if positive_symptoms:
                chips = " ".join([f'<span class="symptom-chip chip-yes">{symptom_display.get(s, s)}</span>' for s in positive_symptoms])
                st.markdown(chips, unsafe_allow_html=True)
            
            st.markdown(f"**Gejala yang tidak dirasakan ({len(negative_symptoms)} Gejala):**" if negative_symptoms else "", unsafe_allow_html=False)
            if negative_symptoms:
                chips2 = " ".join([f'<span class="symptom-chip chip-no">{symptom_display.get(s, s)}</span>' for s in negative_symptoms])
                st.markdown(chips2, unsafe_allow_html=True)

        with col_r2:
            st.markdown("### Hasil Voting K-Tetangga Terdekat")
            fig_vote = go.Figure(go.Bar(x=list(label_counts.keys()), y=list(label_counts.values()), marker_color=["#778873", "#D2DCB6"], text=list(label_counts.values()), textposition="outside"))
            fig_vote.update_layout(yaxis_title="Jumlah Suara", yaxis=dict(range=[0, best_k + 1]), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(family="Inter"), height=240, margin=dict(t=10, b=20, l=10, r=10))
            st.plotly_chart(fig_vote, use_container_width=True)

        st.markdown(f"### {best_k} Tetangga Terdekat (Referensi KNN)")
        neighbor_data = []
        for rank, (dist, lbl, train_idx) in enumerate(neighbors, 1):
            orig_idx = idx_train[train_idx]
            neighbor_data.append({"Ranking": rank, "No. Data Training": orig_idx + 1, "Jarak Euclidean": f"{dist:.6f}", "Label Referensi": "Positive" if lbl == 1 else "Negative"})
        st.dataframe(pd.DataFrame(neighbor_data), use_container_width=True, hide_index=True)

        if st.button("Ulangi Skrining", key="restart_quiz"):
            st.session_state.quiz_started = False; st.session_state.quiz_step = 0; st.session_state.quiz_answers = {}; st.session_state.quiz_result = None; st.rerun()

    else:
        q_idx  = step - 1; 
        q_data = QUESTIONS[q_idx]
        q_key = q_data[0]
        q_text = q_data[2]
        
        q_opts = q_data[4] if len(q_data) > 4 else None

        pct = int((step / TOTAL_Q) * 100)
        st.markdown('<div class="quiz-shell">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="quiz-progress-bar-wrap"><div class="quiz-progress-bar-fill" style="width:{pct}%"></div></div>
        <div class="quiz-step-label">Pertanyaan {step} dari {TOTAL_Q}</div>
        """, unsafe_allow_html=True)

        current_answer = st.session_state.quiz_answers.get(q_key, None)

        st.markdown(f"""
        <div class="quiz-card">
            <div class="quiz-question">{q_text}</div>
        </div>
        """, unsafe_allow_html=True)

        if q_key == "age":
            current_age = st.session_state.quiz_answers.get("age", 30)
            selected_value = st.slider(
                "Pilih usia Anda (dalam tahun):",
                min_value=1, max_value=100, value=current_age, step=1,
                key=f"slider_{q_key}_{step}"
            )
            
        else:
            option_labels = [opt[0] for opt in q_opts]; option_values = [opt[1] for opt in q_opts]
            default_idx = 0
            if current_answer is not None:
                for i, v in enumerate(option_values):
                    if v == current_answer: default_idx = i; break

            st.markdown('<div class="stRadio">', unsafe_allow_html=True)
            selected_label = st.radio("Pilih jawaban:", option_labels, index=default_idx, key=f"radio_{q_key}_{step}", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            selected_value = option_values[option_labels.index(selected_label)]

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if step > 1:
                if st.button("← Sebelumnya", key="btn_back"): st.session_state.quiz_answers[q_key] = selected_value; st.session_state.quiz_step -= 1; st.rerun()
            else:
                if st.button("✕ Batalkan Skrining", key="btn_cancel"): st.session_state.quiz_started = False; st.session_state.quiz_step = 0; st.session_state.quiz_answers = {}; st.rerun()
        with col_next:
            is_last = (step == TOTAL_Q); label = "Lihat Hasil →" if is_last else "Selanjutnya →"
            if st.button(label, key="btn_next"):
                st.session_state.quiz_answers[q_key] = selected_value
                st.session_state.quiz_step += 1
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif active_tab == "Evaluasi Model":
    st.caption(f"Nilai K terbaik saat ini untuk Skrining adalah K = {best_k}.")

    st.markdown("### Eksperimen Nilai K")
    k_value_eval = st.slider("Pilih nilai K untuk mengevaluasi performa model", 1, 21, best_k, 2)

    y_pred_eval, _ = run_knn_full(X_train_norm, y_train, X_test_norm, y_test, k_value_eval)
    metrics_eval = compute_metrics(y_test, y_pred_eval)

    col_cm, col_metrics = st.columns([1, 1])
    with col_cm:
        with st.container():
            st.markdown("### Confusion Matrix (K={})".format(k_value_eval))
            st.markdown("---")
            tp, tn, fp, fn = metrics_eval["TP"], metrics_eval["TN"], metrics_eval["FP"], metrics_eval["FN"]
            cm_z = [[tn, fp], [fn, tp]]
            cm_text = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
            fig_cm = go.Figure(go.Heatmap(z=cm_z, text=cm_text, texttemplate="%{text}", colorscale=[[0, "#F1F3E0"], [1, "#778873"]], showscale=False, textfont=dict(size=16, color="white")))
            fig_cm.update_layout(xaxis=dict(tickvals=[0,1], ticktext=["Prediksi: Negative","Prediksi: Positive"]), yaxis=dict(tickvals=[0,1], ticktext=["Aktual: Negative","Aktual: Positive"]), plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=320, margin=dict(t=20,b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_metrics:
        with st.container():
            st.markdown("### Metrik Evaluasi (K={})".format(k_value_eval))
            st.markdown("---")
            
            metric_names = ["Accuracy", "Precision", "TP Rate", "Specificity", "FPR"]
            metric_vals = [
                metrics_eval["Accuracy"]*100, 
                metrics_eval["Precision"]*100,
                metrics_eval["Recall"]*100,
                metrics_eval["Specificity"]*100,
                (1 - (metrics_eval["FP"]/(metrics_eval["TN"]+metrics_eval["FP"]) if (metrics_eval["TN"]+metrics_eval["FP"])>0 else 0))*100
            ]
            fig_radar = go.Figure(go.Scatterpolar(r=metric_vals, theta=metric_names, fill='toself', line_color="#778873", fillcolor="rgba(119, 136, 115, 0.15)", marker=dict(size=6, color="#A1BC98")))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=10), gridcolor="#E2E8F0"), angularaxis=dict(tickfont=dict(size=12, color="#2D3A2A"))), showlegend=False, paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=320, margin=dict(t=20,b=20,l=40,r=40))
            st.plotly_chart(fig_radar, use_container_width=True)

    with st.container():
        st.markdown("### Ringkasan Metrik Evaluasi (K={})".format(k_value_eval))
        st.markdown("---")
        
        total_data = tp + tn + fp + fn
        prevalence = (tp + fn) / total_data
        fpr = metrics_eval["FP"] / (metrics_eval["TN"] + metrics_eval["FP"]) if (metrics_eval["TN"] + metrics_eval["FP"]) > 0 else 0
        misclassification_rate = (metrics_eval["FP"] + metrics_eval["FN"]) / total_data

        metric_df = pd.DataFrame({
            "Metrik": [
                "Accuracy", 
                "Misclassification Rate", 
                "True Positive Rate (TP Rate)", 
                "False Positive Rate (FPR)", 
                "Specificity", 
                "Precision", 
                "Prevalence"
            ],
            "Rumus": [
                "(TP+TN)/(TP+TN+FP+FN)",
                "(FP+FN)/(TP+TN+FP+FN)",
                "TP/(TP+FN)",
                "FP/(TN+FP)",
                "TN/(TN+FP)",
                "TP/(TP+FP)",
                "(TP+FN)/(TP+TN+FP+FN)"
            ],
            "Nilai": [
                f"{metrics_eval['Accuracy']*100:.2f}%",
                f"{misclassification_rate*100:.2f}%",
                f"{metrics_eval['Recall']*100:.2f}%",
                f"{fpr*100:.2f}%",
                f"{metrics_eval['Specificity']*100:.2f}%",
                f"{metrics_eval['Precision']*100:.2f}%",
                f"{prevalence*100:.2f}%"
            ]
        })
        
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

    with st.container():
        st.markdown("### Analisis Sensitivitas Nilai K")
        st.caption("Perbandingan akurasi model pada berbagai nilai K")
        k_range, acc_list = list(range(1,22,2)), []
        for kk in k_range:
            preds, _ = run_knn_full(X_train_norm, y_train, X_test_norm, y_test, kk)
            acc_list.append(compute_metrics(y_test, preds)["Accuracy"]*100)
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=k_range, y=acc_list, mode="lines+markers", line=dict(color="#D2DCB6", width=2.5), marker=dict(size=8, color="#778873", symbol=["star" if k==k_value_eval else "circle" for k in k_range])))
        fig_k.add_vline(x=k_value_eval, line_dash="dash", line_color="#778873", annotation_text=f"K={k_value_eval} (saat ini)", annotation_position="top right")
        fig_k.update_layout(xaxis_title="Nilai K", yaxis_title="Akurasi (%)", xaxis=dict(tickvals=k_range), yaxis=dict(range=[max(0,min(acc_list)-5), 101]), plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=320, margin=dict(t=20,b=40))
        st.plotly_chart(fig_k, use_container_width=True)

    with st.container():
        st.markdown("### Detail Prediksi Data Test (K={})".format(k_value_eval))
        result_rows = [{"No. Data": idx_test[i]+1, "Label Aktual": "Positive" if actual==1 else "Negative", "Label Prediksi": "Positive" if pred==1 else "Negative", "Status": "Benar" if pred==actual else "Salah"} for i,(pred,actual) in enumerate(zip(y_pred_eval, y_test))]
        result_df = pd.DataFrame(result_rows)
        st.dataframe(result_df.style.apply(lambda row: [""]*3 + ["background-color:#F0FDF4;color:#16A34A;font-weight:600" if row["Status"]=="Benar" else "background-color:#FEF2F2;color:#DC2626;font-weight:600"], axis=1), use_container_width=True, hide_index=True)

elif active_tab == "Eksplorasi Dataset":
    col_a, col_b = st.columns(2)
    with col_a:
        with st.container():
            st.markdown("### Distribusi Kelas")
            class_counts = df_raw["Class"].value_counts()
            fig_pie = go.Figure(go.Pie(labels=class_counts.index, values=class_counts.values, hole=0.55, marker=dict(colors=["#778873","#D2DCB6"]), textinfo="percent+label", textfont=dict(size=13)))
            fig_pie.update_layout(paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=280, margin=dict(t=20,b=20), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        with st.container():
            st.markdown("### Distribusi Umur per Kelas")
            pos_ages = df_raw[df_raw["Class"]=="Positive"]["Age"]
            neg_ages = df_raw[df_raw["Class"]=="Negative"]["Age"]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=pos_ages, name="Positive", marker_color="#778873", opacity=0.75, nbinsx=20))
            fig_hist.add_trace(go.Histogram(x=neg_ages, name="Negative", marker_color="#D2DCB6", opacity=0.75, nbinsx=20))      
            fig_hist.update_layout(barmode="overlay", xaxis_title="Umur", yaxis_title="Jumlah Pasien", plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=280, margin=dict(t=20,b=40), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_hist, use_container_width=True)

    with st.container():
        st.markdown("### Prevalensi Gejala per Kelas")
        symptom_cols = [c for c in df_raw.columns if c not in ["Age","Gender","Class"]]
        pos_df = df_raw[df_raw["Class"]=="Positive"]; neg_df = df_raw[df_raw["Class"]=="Negative"]
        pos_pct = [(pos_df[col]=="Yes").mean()*100 for col in symptom_cols]; neg_pct = [(neg_df[col]=="Yes").mean()*100 for col in symptom_cols]
        symptom_labels = ["Polyuria","Polydipsia","Sudden Weight Loss","Weakness","Polyphagia","Genital Thrush","Visual Blur","Itching","Irritability","Delayed Healing","Partial Paresis","Muscle Stiffness","Alopecia","Obesity"]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Positive", x=symptom_labels, y=pos_pct, marker_color="#778873", opacity=0.85))
        fig_bar.add_trace(go.Bar(name="Negative", x=symptom_labels, y=neg_pct, marker_color="#D2DCB6", opacity=0.85))
        fig_bar.update_layout(barmode="group", yaxis_title="Persentase Pasien (%)", plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=380, margin=dict(t=20,b=60), legend=dict(orientation="h",y=1.05), xaxis=dict(tickangle=-30))
        st.plotly_chart(fig_bar, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        with st.container():
            st.markdown("### Distribusi Jenis Kelamin")
            gender_class = df_raw.groupby(["Gender","Class"]).size().reset_index(name="count")
            fig_gender = go.Figure()
            for cls, color in [("Positive","#778873"),("Negative","#D2DCB6")]:
                sub = gender_class[gender_class["Class"]==cls]
                fig_gender.add_trace(go.Bar(x=sub["Gender"], y=sub["count"], name=cls, marker_color=color, opacity=0.85))
            fig_gender.update_layout(barmode="stack", yaxis_title="Jumlah Pasien", plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF", font=dict(family="Inter"), height=260, margin=dict(t=20,b=40), legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig_gender, use_container_width=True)

    with col_d:
        with st.container():
            st.markdown("### Statistik Deskriptif Umur")
            st.dataframe(df_raw.groupby("Class")["Age"].describe().round(1), use_container_width=True)

    with st.container():
        st.markdown("### Preview Dataset")
        st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan 20 dari {len(df_raw)} baris data.")

elif active_tab == "Insight Diabetes Indonesia":
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Apa itu Diabetes?</div>
        <div style="font-size:1rem; color:#475569; line-height:1.8;">
            <p>Diabetes merupakan penyakit kronis yang ditandai oleh peningkatan kadar glukosa dalam darah. Kondisi ini terjadi karena glukosa tidak dapat dimanfaatkan secara efektif oleh tubuh akibat pankreas tidak mampu menghasilkan insulin dalam jumlah yang memadai atau tubuh tidak dapat memanfaatkan insulin secara optimal.</p>
            <p>Jika tidak dikelola dengan baik, diabetes dapat menyebabkan komplikasi serius seperti penyakit jantung, kerusakan saraf, gagal ginjal, dan kebutaan.</p>
            <p style="margin-top:10px; font-size:0.9rem; color:#94A3B8;"><i>Sumber: Alodokter</i></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">Apa Penyebab Diabetes?</div>
        <div style="font-size:1rem; color:#475569; line-height:1.8;">
            <p>Penyebab utama diabetes bergantung pada jenisnya, yang dibagi menjadi <strong>Diabetes Tipe 1</strong> dan <strong>Diabetes Tipe 2</strong>.</p>
            <ul style="padding-left:20px; margin-bottom:10px;">
                <li><strong>Diabetes Tipe 1:</strong> Terjadi akibat sistem kekebalan tubuh (imun) secara keliru menyerang dan menghancurkan sel-sel pankreas yang memproduksi insulin. Penderita Tipe 1 memerlukan suntikan insulin seumur hidup.</li>
                <br>
                <li><strong>Diabetes Tipe 2 (Paling Umum):</strong> Terjadi ketika tubuh menjadi <em>resisten</em> terhadap insulin (tidak responsif), atau pankreas tidak mampu memproduksi cukup insulin. Faktor risiko utamanya meliputi obesitas, pola makan tidak sehat, kurang aktivitas fisik, dan faktor genetik.</li>
            </ul>
            <p style="margin-top:10px; font-size:0.9rem; color:#94A3B8;"><i>Sumber: BB Labkesmas Makassar</i></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="background: #ffffff; border-radius: 12px; padding: 28px; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(119, 136, 115, 0.05); margin-bottom: 20px;">
            <div class="section-title">Diabetes di Indonesia: Gambaran Umum (IDF Atlas 2024)</div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: st.metric("Penderita Diabetes (2024)", "20,4 Juta", help="IDF Diabetes Atlas 2024 (Usia 20-79 tahun)")
        with col_s2: st.metric("Prevalensi Nasional", "11,3%", help="IDF Diabetes Atlas 2024 (Usia 20-79 tahun)")
        with col_s3: st.metric("Peringkat Dunia", "No. 5", help="Jumlah penderita dewasa terbanyak ke-5 di dunia")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="background: #ffffff; border-radius: 12px; padding: 28px; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(119, 136, 115, 0.05); margin-bottom: 20px;">
            <div class="section-title">Profil Diabetes Indonesia 2000 - 2050 (Sumber: IDF Atlas)</div>
        """, unsafe_allow_html=True)
        
        col_head_1, col_head_2, col_head_3, col_head_4 = st.columns(4)
        with col_head_1: st.markdown("**Tahun**")
        with col_head_2: st.markdown("**2000**")
        with col_head_3: st.markdown("**2024**")
        with col_head_4: st.markdown("**2050 (Proyeksi)**")
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1: st.markdown("Penderita (20-79 thn)")
        with col_d2: st.markdown("5.7 Juta")
        with col_d3: st.markdown("**20.4 Juta**")
        with col_d4: st.markdown("28.6 Juta")
        
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1: st.markdown("Prevalensi (Usia 20-79 thn)")
        with col_e2: st.markdown("-")
        with col_e3: st.markdown("**11.3%**")
        with col_e4: st.markdown("12.6%")
        
        st.markdown("<hr style='margin: 15px 0; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
        
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1: st.markdown("Diabetes Tidak Terdiagnosis")
        with col_f2: st.markdown("-")
        with col_f3: st.markdown("**73.2%** (14.9 Juta)")
        with col_f4: st.markdown("-")
        
        st.markdown("""
        <div style="background:#FCF9F2; padding:12px 16px; border-radius:8px; margin-top:12px; border-left:4px solid #778873; font-size:0.9rem; color:#2D3A2A;">
            Catatan Penting: Indonesia merupakan salah satu dari 37 negara di kawasan Pasifik Barat. Indonesia memiliki jumlah penderita diabetes dewasa (usia 20-79 tahun) <strong>terbanyak ke-5 di dunia</strong>.
        </div>
        </div>
        """, unsafe_allow_html=True)

    col_trend, col_province = st.columns(2)
    
    with col_trend:
        with st.container():
            st.markdown("### Tren Prevalensi Diabetes")
            st.markdown("---")
            years = [2000, 2011, 2024, 2050]
            age_prevalence = [0.0, 5.1, 11.3, 13.0]
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=years, y=age_prevalence,
                mode="lines+markers+text",
                text=[f"{v}%" for v in age_prevalence],
                textposition="top center",
                line=dict(color="#778873", width=3),
                marker=dict(size=10, color="#778873"),
                fill="tozeroy",
                fillcolor="rgba(119, 136, 115, 0.15)"
            ))
            fig_trend.update_layout(
                xaxis=dict(tickvals=years, title="Tahun"),
                yaxis=dict(title="Prevalensi (%)", range=[0, 14]),
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF",
                font=dict(family="Inter"), height=300, margin=dict(t=20,b=40)
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            st.caption("Sumber: International Diabetes Federation")

    with col_province:
        with st.container():
            st.markdown("### 10 Provinsi Prevalensi Tertinggi (2023)")
            st.markdown("---")
            provinces = ["DKI Jakarta", "DI Yogyakarta", "Kalimantan Timur", "Jawa Timur", "Bangka Belitung", "Sulawesi Utara", "Banten", "Jawa Tengah", "Jawa Barat", "Bali"]
            prev_prov = [3.1, 2.9, 2.3, 2.2, 2.1, 2.1, 1.9, 1.8, 1.7, 1.7]
            provinces = provinces[::-1]
            prev_prov = prev_prov[::-1]
            fig_prov = go.Figure(go.Bar(
                x=prev_prov, y=provinces, orientation="h",
                marker=dict(color=prev_prov, colorscale=[[0,"#F1F3E0"],[1,"#778873"]], showscale=False),
                text=[f"{v}%" for v in prev_prov], textposition="outside"
            ))
            fig_prov.update_layout(
                xaxis_title="Prevalensi (%)", xaxis=dict(range=[0, 3.5]),
                plot_bgcolor="#F8FAFF", paper_bgcolor="#F8FAFF",
                font=dict(family="Inter"), height=350, margin=dict(t=20,b=40,l=20)
            )
            st.plotly_chart(fig_prov, use_container_width=True)
            st.caption("Sumber: Kementrian Kesehatan Republik Indonesia")

    st.markdown(
        """<div class="section-card">
        <div class="section-title">Data Kesehatan & Beban Ekonomi (IDF Atlas 2024)</div>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between;">
        <div style="flex: 1; min-width: 250px; background: #F8FAFF; padding: 16px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-weight: 700; color: #778873; margin-bottom: 12px;">Glukosa Puasa Terganggu (IFG)</div>
            <ul style="list-style-type: disc; padding-left: 20px; margin: 0; line-height: 1.8; color: #475569;">
                <li><strong>Jumlah:</strong> 14,2 Juta (2024)</li>
                <li><strong>Diproyeksikan:</strong> 18,7 Juta (2050)</li>
            </ul>
        </div>
        <div style="flex: 1; min-width: 250px; background: #F8FAFF; padding: 16px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-weight: 700; color: #778873; margin-bottom: 12px;">Biaya Kesehatan Akibat Diabetes</div>
            <ul style="list-style-type: disc; padding-left: 20px; margin: 0; line-height: 1.8; color: #475569;">
                <li><strong>Total Pengeluaran (2024):</strong> USD 6,3 Miliar</li>
                <li><strong>Diproyeksikan (2050):</strong> USD 8,1 Miliar</li>
                <li><strong>Biaya Per Orang (2024):</strong> USD 308</li>
            </ul>
        </div>
        <div style="flex: 1; min-width: 250px; background: #F8FAFF; padding: 16px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-weight: 700; color: #778873; margin-bottom: 12px;">Diabetes Tipe 1 pada Anak & Remaja</div>
            <ul style="list-style-type: disc; padding-left: 20px; margin: 0; line-height: 1.8; color: #475569;">
                <li><strong>Total (Semua Usia):</strong> 11.713 penderita</li>
                <li><strong>Usia 0-19 tahun:</strong> 3.495 penderita</li>
            </ul>
        </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">Faktor Risiko Utama Diabetes Tipe 2</div>
        <div style="font-size:1rem; color:#475569; line-height:1.8;">
            <ol style="padding-left: 20px; margin: 0;">
                <li><strong>Faktor Genetik:</strong> Memiliki anggota keluarga (seperti orang tua atau saudara) yang mengidap diabetes meningkatkan kemungkinan Anda untuk mengalaminya juga.</li>
                <li><strong>Obesitas atau Kelebihan Berat Badan:</strong> Obesitas adalah salah satu faktor risiko utama untuk diabetes tipe 2, karena lemak perut dapat meningkatkan resistensi insulin.</li>
                <li><strong>Kurang Aktivitas Fisik:</strong> Gaya hidup yang tidak aktif meningkatkan risiko diabetes, karena tubuh membutuhkan olahraga untuk membantu mengontrol kadar gula darah.</li>
                <li><strong>Usia:</strong> Risiko diabetes tipe 2 meningkat seiring bertambahnya usia, terutama setelah usia 45 tahun.</li>
                <li><strong>Riwayat Diabetes Gestasional:</strong> Wanita yang mengalami diabetes gestasional saat hamil lebih berisiko mengembangkan diabetes tipe 2 di masa depan.</li>
                <li><strong>Diet Tidak Sehat:</strong> Pola makan yang tinggi gula, lemak jenuh, dan karbohidrat olahan dapat berkontribusi pada kenaikan berat badan dan resistensi insulin.</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">6 Langkah Sehat Mencegah Diabetes</div>
        <div style="font-size:1rem; color:#475569; line-height:1.8;">
            <p style="margin-bottom: 16px;">Berikut ini adalah 6 langkah sehat yang dapat mencegah seseorang menderita penyakit diabetes melitus, diantaranya adalah:</p>
            <ol style="padding-left: 20px; margin: 0;">
                <li style="margin-bottom: 12px;">
                    <strong>Berhenti Merokok</strong><br>
                    Merokok merupakan salah satu kegiatan yang bukan saja tidak sehat bagi paru-paru, namun juga dapat menyebabkan seseorang terkena penyakit diabetes melitus. Untuk itu, hindari penggunaan tembakau (rokok dan tembakau kunyah) serta berhenti mengkonsumsi minuman beralkohol.
                </li>
                <li style="margin-bottom: 12px;">
                    <strong>Mempertahankan Berat Badan Ideal</strong><br>
                    Mengatur pola makan dengan gizi seimbang untuk mempertahankan berat badan ideal. Kurangi konsumsi karbohidrat dan perbanyak makanan yang kaya akan serat.
                </li>
                <li style="margin-bottom: 12px;">
                    <strong>Melakukan Aktivitas Fisik</strong><br>
                    Aktivitas fisik ringan seperti berjalan, menaiki tangga, hingga melakukan aerobik juga terbukti mampu menurunkan kadar gula dalam tubuh, sehingga tubuh menjadi sehat, berat badan ideal, dan sekaligus meminimalisir seseorang menderita penyakit diabetes melitus.
                </li>
                <li style="margin-bottom: 12px;">
                    <strong>Mengkonsumsi Makanan yang Sehat</strong><br>
                    Salah satu upaya untuk mencegah terkena diabetes melitus dengan konsumsi makanan yang sehat untuk mendapatkan nutrisi. Konsumsi 3-5 porsi buah dan sayur, serta mengurangi asupan gula, garam dan lemak jenuh.
                </li>
                <li style="margin-bottom: 12px;">
                    <strong>Rutin Periksa Gula Darah</strong><br>
                    Memeriksa gula darah atau HbA1c secara rutin merupakan salah satu cara untuk mendeteksi sedini mungkin kandungan gula darah dalam tubuh, sehingga apabila seseorang terpapar diabetes, akan lebih cepat mendapatkan penanganan.
                </li>
                <li style="margin-bottom: 12px;">
                    <strong>Mengelola Stres</strong><br>
                    Stres merupakan salah satu penyebab diabetes yang mungkin jarang diketahui oleh masyarakat. Pasalnya ketika tubuh mengalami stres, produksi serotonin akan terganggu, sehingga menyebabkan kemampuan tubuh dalam menciptakan insulin akan berkurang.
                </li>
            </ol>
            <div style="background:#FFF7ED; border-left: 4px solid #F97316; padding: 14px 20px; margin-top: 24px; border-radius: 8px; color: #92400E; font-weight: 500;">
                Ingat: Lebih baik mencegah daripada mengobati. Mari disiplin menerapkan perilaku hidup bersih dan sehat, serta lakukan pemeriksaan ke fasilitas kesehatan apabila mengalami gejala diabetes melitus seperti buang air kecil lebih dari biasanya terutama saat malam hari, kehilangan berat badan tanpa melakukan apapun, hingga luka yang tidak pernah sembuh. -Atabaik (2026)
            </div>
            <p style="margin-top:16px; font-size:0.9rem; color:#94A3B8;">
                <i>Sumber: Kementerian Kesehatan Republik Indonesia</i>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)