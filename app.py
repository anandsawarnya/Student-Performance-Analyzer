import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Page setup + premium dark navy styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #07111f 0%, #0b1730 55%, #101c38 100%); color: #edf3ff; }
[data-testid="stSidebar"] { background: #081222; border-right: 1px solid #20304d; }
[data-testid="stSidebar"] * { color: #e8efff !important; }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3 { color: #f3f7ff !important; letter-spacing: -0.03em; }
p, label, .stMarkdown { color: #c6d2e7; }
.hero {
  padding: 1.6rem 1.8rem; border-radius: 22px;
  background: linear-gradient(120deg, rgba(28,55,99,.95), rgba(16,29,57,.95));
  border: 1px solid rgba(115,157,230,.28); margin-bottom: 1.2rem;
  box-shadow: 0 18px 45px rgba(0,0,0,.18);
}
.eyebrow { color: #78a9ff; font-size: .78rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { font-size: 2.35rem; margin: .35rem 0 .5rem 0; }
.hero p { color: #c4d4ef; font-size: 1rem; margin-bottom: .1rem; }
.metric-card {
  background: linear-gradient(145deg, #122342, #0d1a31);
  border: 1px solid #263a5e; border-radius: 18px; padding: 1.1rem 1.2rem;
  min-height: 125px; box-shadow: 0 10px 25px rgba(0,0,0,.12);
}
.metric-label { color: #9eb2d3; font-size: .82rem; font-weight: 600; }
.metric-value { color: #f5f8ff; font-size: 1.9rem; font-weight: 800; margin-top: .3rem; }
.metric-sub { color: #77d9bd; font-size: .78rem; margin-top: .2rem; }
.panel {
  background: rgba(13,27,50,.82); border: 1px solid #243858;
  border-radius: 18px; padding: 1.15rem 1.25rem; margin-bottom: 1rem;
}
.small-muted { color: #9eb2d3; font-size: .88rem; }
div[data-testid="stDataFrame"] { border: 1px solid #263a5e; border-radius: 12px; }
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; border: 1px solid #5688e8; background: #2f6fed;
  color: white; font-weight: 700; padding: .55rem 1rem;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: #245bd0; border-color: #7ba4ff; color: white; }
div[data-baseweb="input"] input, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] textarea {
  background-color: #101f3a !important; color: #eef4ff !important; border-color: #2a4167 !important;
}
hr { border-color: #253957; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Constants and data/model utilities
# ------------------------------------------------------------
TARGET = "final_grade"
DROP_FROM_FEATURES = ["overall_score"]  # SRS: avoid target leakage

EXPECTED_COLUMNS = [
    "Age", "Gender", "Schoool_types", "parent_education", "study_hours",
    "attendance_percentage", "internet_access", "travel_time",
    "extra_activities", "study_method", "math_score", "science_score",
    "english_score", "overall_score", "final_grade"
]

NUMERIC_HINTS = [
    "Age", "study_hours", "attendance_percentage", "internet_access",
    "travel_time", "math_score", "science_score", "english_score"
]
CATEGORICAL_HINTS = [
    "Gender", "Schoool_types", "parent_education", "extra_activities", "study_method"
]

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Common spelling variants mapped to the SRS names
    aliases = {
        "school_types": "Schoool_types",
        "school_type": "Schoool_types",
        "School_types": "Schoool_types",
        "School_type": "Schoool_types",
        "age": "Age",
        "gender": "Gender",
        "final grade": "final_grade",
        "Final_Grade": "final_grade",
        "overall score": "overall_score",
        "Overall_Score": "overall_score",
        "Study_Hours": "study_hours",
        "Attendance_Percentage": "attendance_percentage",
        "Math_Score": "math_score",
        "Science_Score": "science_score",
        "English_Score": "english_score",
    }
    return df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return normalize_columns(pd.read_csv(uploaded_file))
    local_path = "Student_Performance.csv"
    if os.path.exists(local_path):
        return normalize_columns(pd.read_csv(local_path))
    return None

def prepare_data(raw_df):
    df = normalize_columns(raw_df)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset mein '{TARGET}' target column nahi mila.")
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET]).copy()
    # Remove rows with an empty target and keep target labels as strings
    y = df[TARGET].astype(str).str.strip()
    X = df.drop(columns=[TARGET] + [c for c in DROP_FROM_FEATURES if c in df.columns], errors="ignore")
    # Drop columns that are entirely empty
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("Target ke alawa koi usable feature column nahi mila.")
    return X, y, df

def build_pipeline(X, estimator):
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

@st.cache_resource(show_spinner=False)
def train_models(csv_bytes):
    from io import BytesIO
    raw = pd.read_csv(BytesIO(csv_bytes))
    X, y, clean_df = prepare_data(raw)

    # Keep a holdout split for honest evaluation.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=None),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight=None),
    }
    trained = {}
    rows = []
    predictions = {}
    for name, estimator in estimators.items():
        pipe = build_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        trained[name] = pipe
        predictions[name] = pred
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1-Score (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
        })
    metrics = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    return {
        "X": X, "y": y, "clean_df": clean_df, "X_test": X_test, "y_test": y_test,
        "models": trained, "metrics": metrics, "best_name": best_name,
        "predictions": predictions,
    }

def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

def styled_chart(fig):
    fig.patch.set_facecolor("#0d1b32")
    for ax in fig.axes:
        ax.set_facecolor("#0d1b32")
        ax.tick_params(colors="#c6d2e7")
        ax.xaxis.label.set_color("#c6d2e7")
        ax.yaxis.label.set_color("#c6d2e7")
        ax.title.set_color("#f3f7ff")
        for spine in ax.spines.values():
            spine.set_color("#2b4062")
        ax.grid(axis="y", alpha=.18, color="#90a9d0")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 SPA")
    st.caption("STUDENT PERFORMANCE ANALYZER")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Predict Grade", "Analytics", "Model Lab", "About Project"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader("Upload Student_Performance.csv", type=["csv"])
    st.caption("CSV mein `final_grade` target column hona chahiye.")
    st.markdown("---")
    st.markdown('<div class="small-muted">HCL-Tech guided academic project · 2026–27</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Dataset handling
# ------------------------------------------------------------
data = load_dataset(uploaded)
if data is None:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Machine Learning · Academic Insights</div>
      <h1>Student Performance Analyzer</h1>
      <p>Predict final grades, explore academic trends, and compare classification models in one place.</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("Dataset abhi load nahi hua. Sidebar mein apni `Student_Performance.csv` file upload karo, ya CSV ko `app.py` ke same folder mein rakho.")
    st.markdown('<div class="panel"><h3>🚀 What this app includes</h3><p>Grade prediction · Interactive analytics · Model comparison · Confusion matrix · Evaluation KPIs</p></div>', unsafe_allow_html=True)
    st.info("Real ML prediction ke liye actual dataset zaroori hai. App sample/fake predictions generate nahi karta.")
    st.stop()

try:
    model_bundle = train_models(data.to_csv(index=False).encode("utf-8"))
except Exception as exc:
    st.error(f"Dataset/model setup mein problem aayi: {exc}")
    st.info("CSV headers check karo. SRS ke mutabik target column `final_grade` hona chahiye, aur har grade class ke examples hone chahiye.")
    st.stop()

clean_df = model_bundle["clean_df"]
metrics = model_bundle["metrics"]
best_name = model_bundle["best_name"]
best_model = model_bundle["models"][best_name]
X = model_bundle["X"]
y = model_bundle["y"]
n_records = len(clean_df)
n_features = X.shape[1]
n_grades = y.nunique()
best_acc = metrics.iloc[0]["Accuracy"]

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">Student Intelligence Platform</div>
      <h1>Academic Performance, <span style="color:#78a9ff">Decoded.</span></h1>
      <p>Turn student data into useful insights with machine learning, visual analytics, and grade prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("STUDENT RECORDS", f"{n_records:,}", "After duplicate removal")
    with c2: metric_card("INPUT FEATURES", str(n_features), "Leakage-prone overall_score excluded")
    with c3: metric_card("GRADE CLASSES", str(n_grades), "Target classes found in CSV")
    with c4: metric_card("BEST MODEL ACCURACY", f"{best_acc:.1%}", best_name)

    st.markdown("### Model performance at a glance")
    left, right = st.columns([1.25, 1])
    with left:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar(metrics["Model"], metrics["Accuracy"] * 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Model accuracy comparison")
        ax.tick_params(axis="x", rotation=12)
        styled_chart(fig)
    with right:
        st.markdown('<div class="panel"><h3>🧭 Quick insights</h3>', unsafe_allow_html=True)
        st.write(f"**Best-performing model:** {best_name}")
        st.write(f"**Test accuracy:** {best_acc:.2%}")
        st.write(f"**Evaluation split:** 80% train / 20% test")
        st.write("**Random state:** 42")
        st.write("**Target:** `final_grade`")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Evaluation metrics")
    show_metrics = metrics.copy()
    for col in show_metrics.columns[1:]:
        show_metrics[col] = show_metrics[col].map(lambda v: f"{v:.2%}")
    st.dataframe(show_metrics, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
elif page == "Predict Grade":
    st.markdown('<div class="hero"><div class="eyebrow">Prediction Studio</div><h1>Predict a Student Grade</h1><p>Enter student attributes and let the selected trained model estimate the final grade.</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-muted">Active model: <b>{best_name}</b> · trained from the uploaded dataset</div>', unsafe_allow_html=True)
    st.markdown("### Student details")

    input_values = {}
    cols = st.columns(3)
    for idx, colname in enumerate(X.columns):
        with cols[idx % 3]:
            series = X[colname]
            if pd.api.types.is_numeric_dtype(series):
                finite = pd.to_numeric(series, errors="coerce").dropna()
                default = float(finite.median()) if not finite.empty else 0.0
                min_val = float(finite.min()) if not finite.empty else 0.0
                max_val = float(finite.max()) if not finite.empty else max(default + 1, 1.0)
                if min_val == max_val:
                    max_val = min_val + 1.0
                if colname == "attendance_percentage":
                    min_val, max_val = min(0.0, min_val), max(100.0, max_val)
                    default = min(max(default, min_val), max_val)
                input_values[colname] = st.number_input(
                    colname.replace("_", " ").title(),
                    min_value=min_val, max_value=max_val, value=min(max(default, min_val), max_val),
                    key=f"input_{colname}"
                )
            else:
                vals = series.dropna().astype(str).unique().tolist()
                vals = sorted(vals)
                if not vals:
                    vals = ["Unknown"]
                input_values[colname] = st.selectbox(
                    colname.replace("_", " ").title(), vals, key=f"input_{colname}"
                )

    if st.button("✨ Predict Final Grade", use_container_width=True):
        one_student = pd.DataFrame([input_values], columns=X.columns)
        prediction = best_model.predict(one_student)[0]
        st.markdown("---")
        st.success(f"Predicted Final Grade: **{prediction}**")
        st.caption("This is an ML estimate based on the training dataset, not an official academic grade.")
        if hasattr(best_model, "predict_proba"):
            try:
                probabilities = best_model.predict_proba(one_student)[0]
                classes = best_model.classes_
                confidence = float(np.max(probabilities))
                st.metric("Model confidence (highest class probability)", f"{confidence:.1%}")
                prob_df = pd.DataFrame({"Grade": classes, "Probability": probabilities})
                prob_df = prob_df.sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Grade"))
            except Exception:
                pass

# ------------------------------------------------------------
# Analytics
# ------------------------------------------------------------
elif page == "Analytics":
    st.markdown('<div class="hero"><div class="eyebrow">Data Explorer</div><h1>Academic Analytics</h1><p>Explore grade distribution, subject scores, attendance, and study patterns from your dataset.</p></div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel"><h3>Final grade distribution</h3></div>', unsafe_allow_html=True)
        counts = clean_df[TARGET].astype(str).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(counts.index, counts.values)
        ax.set_xlabel("Final grade")
        ax.set_ylabel("Number of students")
        ax.set_title("Students by final grade")
        styled_chart(fig)
    with b:
        numeric_cols = [c for c in ["math_score", "science_score", "english_score"] if c in clean_df.columns]
        if numeric_cols:
            st.markdown('<div class="panel"><h3>Subject score averages</h3></div>', unsafe_allow_html=True)
            means = clean_df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean()
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.bar([c.replace("_score", "").title() for c in means.index], means.values)
            ax.set_ylabel("Average score")
            ax.set_title("Average subject scores")
            styled_chart(fig)
        else:
            st.info("Subject score columns were not found in this CSV.")
    c, d = st.columns(2)
    with c:
        if "attendance_percentage" in clean_df.columns:
            st.markdown('<div class="panel"><h3>Attendance distribution</h3></div>', unsafe_allow_html=True)
            vals = pd.to_numeric(clean_df["attendance_percentage"], errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(vals, bins=15)
            ax.set_xlabel("Attendance percentage")
            ax.set_ylabel("Student count")
            ax.set_title("Attendance distribution")
            styled_chart(fig)
        else:
            st.info("attendance_percentage column not found.")
    with d:
        if "study_hours" in clean_df.columns:
            st.markdown('<div class="panel"><h3>Study hours distribution</h3></div>', unsafe_allow_html=True)
            vals = pd.to_numeric(clean_df["study_hours"], errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(vals, bins=15)
            ax.set_xlabel("Study hours")
            ax.set_ylabel("Student count")
            ax.set_title("Study hours distribution")
            styled_chart(fig)
        else:
            st.info("study_hours column not found.")
    st.markdown("### Dataset preview")
    st.dataframe(clean_df.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_cleaned.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# Model lab
# ------------------------------------------------------------
elif page == "Model Lab":
    st.markdown('<div class="hero"><div class="eyebrow">Model Evaluation Lab</div><h1>Compare the Models</h1><p>Review test-set performance and inspect the confusion matrix for a selected classifier.</p></div>', unsafe_allow_html=True)
    formatted = metrics.copy()
    for col in formatted.columns[1:]:
        formatted[col] = formatted[col].map(lambda v: f"{v:.2%}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    selected = st.selectbox("Select model for confusion matrix", list(model_bundle["models"].keys()), index=list(model_bundle["models"].keys()).index(best_name))
    pred = model_bundle["predictions"][selected]
    actual = model_bundle["y_test"]
    labels = sorted(pd.Series(actual).astype(str).unique().tolist())
    cm = confusion_matrix(actual, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(f"Confusion Matrix — {selected}")
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Actual grade")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] < cm.max() * .6 else "#07111f", fontsize=9)
    styled_chart(fig)

    st.markdown("### Classification report")
    report = classification_report(actual, pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ------------------------------------------------------------
# About
# ------------------------------------------------------------
elif page == "About Project":
    st.markdown('<div class="hero"><div class="eyebrow">Project Overview</div><h1>About Student Performance Analyzer</h1><p>Machine-learning based academic grade estimation and performance analysis.</p></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
    <h3>🎯 Objective</h3>
    Analyze student-related data such as study hours, attendance, and subject scores to estimate the final grade.
    </div>
    <div class="panel">
    <h3>🧠 Algorithms</h3>
    Logistic Regression, Decision Tree, and Random Forest classification.
    </div>
    <div class="panel">
    <h3>🧹 Preprocessing</h3>
    Duplicate removal, missing-value imputation, categorical one-hot encoding, and an 80:20 train-test split with random_state=42.
    </div>
    <div class="panel">
    <h3>📏 Evaluation</h3>
    Accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrix.
    </div>
    <div class="panel">
    <h3>⚠️ Limitations</h3>
    Predictions depend on the dataset and are estimates, not official grades. The SRS excludes `overall_score` from model inputs to reduce target leakage.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Important:** The model metrics shown by this app are calculated from the dataset you upload. They may differ from figures in the SRS.")
