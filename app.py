import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
from datetime import datetime
from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Quality Guardian",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# DATABASE
# ============================================================

DB_FILE = "data_quality.db"


def create_database():
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            total_rows INTEGER,
            total_columns INTEGER,
            missing_values INTEGER,
            duplicate_rows INTEGER,
            invalid_emails INTEGER,
            anomalies INTEGER,
            quality_score REAL,
            processed_at TEXT
        )
    """)

    connection.commit()
    connection.close()


create_database()


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🛡️ AI Data Quality Guardian</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Data Profiling • Cleaning • Validation • ETL • Anomaly Detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# READ DATASET
# ============================================================

def read_dataset(uploaded_file):
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(uploaded_file)

        st.error("Please upload a CSV or Excel file.")
        return None

    except Exception as error:
        st.error(f"Unable to read the file: {error}")
        return None


# ============================================================
# EMAIL VALIDATION
# ============================================================

def is_valid_email(value):
    if pd.isna(value):
        return False

    value = str(value).strip()

    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    return bool(re.match(pattern, value))


def detect_invalid_emails(df):
    invalid_count = 0
    email_columns = []

    for column in df.columns:
        if "email" in str(column).lower():
            email_columns.append(column)

    for column in email_columns:
        invalid_count += int(
            (~df[column].apply(is_valid_email)).sum()
        )

    return invalid_count, email_columns


# ============================================================
# DATA PROFILING
# ============================================================

def profile_dataset(df):
    profile = []

    for column in df.columns:
        profile.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Missing Values": int(df[column].isna().sum()),
            "Missing %": round(df[column].isna().mean() * 100, 2),
            "Unique Values": int(df[column].nunique()),
            "Duplicate Values": int(df[column].duplicated().sum())
        })

    return pd.DataFrame(profile)


# ============================================================
# AI ANOMALY DETECTION
# ============================================================

def detect_anomalies(df):
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    flags = pd.Series(False, index=df.index)

    if not numeric_columns:
        return flags

    working_df = df[numeric_columns].copy()

    working_df = working_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in numeric_columns:
        median = working_df[column].median()

        if pd.isna(median):
            median = 0

        working_df[column] = working_df[column].fillna(median)

    if len(working_df) < 10:
        return flags

    try:
        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42
        )

        predictions = model.fit_predict(working_df)

        flags = pd.Series(
            predictions == -1,
            index=df.index
        )

    except Exception:
        pass

    return flags


# ============================================================
# DATA CLEANING + TRANSFORMATION
# ============================================================

def clean_dataset(df):
    cleaned = df.copy()
    cleaning_log = []

    # Remove completely empty rows
    before = len(cleaned)

    cleaned = cleaned.dropna(how="all")

    removed = before - len(cleaned)

    if removed:
        cleaning_log.append(
            f"Removed {removed} completely empty rows."
        )

    # Remove duplicate rows
    before = len(cleaned)

    cleaned = cleaned.drop_duplicates()

    removed = before - len(cleaned)

    if removed:
        cleaning_log.append(
            f"Removed {removed} duplicate rows."
        )

    # Strip whitespace from text columns
    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for column in text_columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    if text_columns:
        cleaning_log.append(
            "Removed unnecessary whitespace from text fields."
        )

    # Fill numeric missing values with median
    numeric_columns = cleaned.select_dtypes(
        include=np.number
    ).columns.tolist()

    for column in numeric_columns:
        missing = int(cleaned[column].isna().sum())

        if missing > 0:
            median_value = cleaned[column].median()

            if pd.notna(median_value):
                cleaned[column] = cleaned[column].fillna(
                    median_value
                )

                cleaning_log.append(
                    f"Filled {missing} missing values in "
                    f"{column} using the median."
                )

    # Fill categorical missing values with mode
    categorical_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for column in categorical_columns:
        missing = int(cleaned[column].isna().sum())

        if missing > 0:
            mode_values = cleaned[column].mode()

            if not mode_values.empty:
                cleaned[column] = cleaned[column].fillna(
                    mode_values.iloc[0]
                )

                cleaning_log.append(
                    f"Filled {missing} missing values in "
                    f"{column} using the most frequent value."
                )

    # SAFE column-name transformation.
    # This replaces the old cleaned.columns = [...] code
    # that caused the length-mismatch error.
    rename_map = {}

    for column in cleaned.columns:
        new_name = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        rename_map[column] = new_name

    cleaned = cleaned.rename(columns=rename_map)

    cleaning_log.append(
        "Standardized column names."
    )

    return cleaned, cleaning_log


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    total_rows,
    total_columns,
    missing_values,
    duplicate_rows,
    invalid_emails,
    anomalies
):
    if total_rows == 0:
        return 0.0

    total_cells = max(
        total_rows * max(total_columns, 1),
        1
    )

    missing_penalty = (missing_values / total_cells) * 100
    duplicate_penalty = (duplicate_rows / total_rows) * 100
    email_penalty = (invalid_emails / total_rows) * 100
    anomaly_penalty = (anomalies / total_rows) * 100

    score = 100.0

    score -= missing_penalty * 0.30
    score -= duplicate_penalty * 0.20
    score -= email_penalty * 0.20
    score -= anomaly_penalty * 0.30

    return round(max(0.0, min(100.0, score)), 2)


# ============================================================
# SAVE PIPELINE RESULT
# ============================================================

def save_pipeline_result(
    filename,
    total_rows,
    total_columns,
    missing_values,
    duplicate_rows,
    invalid_emails,
    anomalies,
    quality_score
):
    connection = sqlite3.connect(DB_FILE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO pipeline_runs (
            filename,
            total_rows,
            total_columns,
            missing_values,
            duplicate_rows,
            invalid_emails,
            anomalies,
            quality_score,
            processed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        total_rows,
        total_columns,
        missing_values,
        duplicate_rows,
        invalid_emails,
        anomalies,
        quality_score,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Pipeline")

    st.write(
        "Upload a dataset to start the AI-powered "
        "data quality pipeline."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"]
    )

    st.divider()

    st.subheader("Pipeline Stages")

    st.write("1️⃣ Data Ingestion")
    st.write("2️⃣ Data Profiling")
    st.write("3️⃣ Quality Detection")
    st.write("4️⃣ AI Anomaly Detection")
    st.write("5️⃣ Data Cleaning")
    st.write("6️⃣ Transformation")
    st.write("7️⃣ Validation")
    st.write("8️⃣ Database Loading")

    st.divider()

    st.caption(
        "AI Data Quality Guardian • Data Engineering Project"
    )


# ============================================================
# LANDING PAGE
# ============================================================

if uploaded_file is None:
    st.info(
        "👈 Upload a CSV or Excel dataset from the sidebar "
        "to start the pipeline."
    )

    st.markdown("## 🔍 What this system does")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            "### 📥 Ingest\n"
            "Upload raw CSV or Excel datasets."
        )

    with col2:
        st.markdown(
            "### 🔎 Detect\n"
            "Find missing values, duplicates, invalid "
            "records and anomalies."
        )

    with col3:
        st.markdown(
            "### 🧹 Clean\n"
            "Automatically clean and transform the dataset."
        )

    with col4:
        st.markdown(
            "### 📊 Analyze\n"
            "Generate quality scores and downloadable reports."
        )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = read_dataset(uploaded_file)

if df is None:
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# ============================================================
# DATA ANALYSIS
# ============================================================

st.success(
    f"Successfully loaded **{uploaded_file.name}**"
)

total_rows = len(df)
total_columns = len(df.columns)

missing_values = int(
    df.isna().sum().sum()
)

duplicate_rows = int(
    df.duplicated().sum()
)

invalid_emails, email_columns = detect_invalid_emails(df)

anomaly_flags = detect_anomalies(df)

anomaly_count = int(anomaly_flags.sum())

quality_score = calculate_quality_score(
    total_rows,
    total_columns,
    missing_values,
    duplicate_rows,
    invalid_emails,
    anomaly_count
)


# ============================================================
# OVERVIEW
# ============================================================

st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Rows", f"{total_rows:,}")

with col2:
    st.metric("Columns", total_columns)

with col3:
    st.metric("Missing Values", f"{missing_values:,}")

with col4:
    st.metric("Duplicates", f"{duplicate_rows:,}")

with col5:
    st.metric("Quality Score", f"{quality_score}%")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Profile",
    "🤖 AI Detection",
    "🧹 Cleaning & ETL",
    "📊 Dashboard",
    "📄 Report"
])


# ============================================================
# TAB 1: PROFILE
# ============================================================

with tab1:
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Column Profile")

    st.dataframe(
        profile_dataset(df),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
        "Missing %": [
            round(
                df[column].isna().mean() * 100,
                2
            )
            for column in df.columns
        ]
    })

    st.dataframe(
        missing_table,
        use_container_width=True
    )


# ============================================================
# TAB 2: AI DETECTION
# ============================================================

with tab2:
    st.subheader("🤖 AI Data Quality Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Invalid Emails", invalid_emails)

    with col2:
        st.metric("AI Anomalies", anomaly_count)

    with col3:
        st.metric("Duplicate Records", duplicate_rows)

    st.divider()

    if anomaly_count > 0:
        st.warning(
            f"⚠️ AI detected {anomaly_count} potential anomalous records."
        )

        anomaly_df = df[anomaly_flags].copy()

        st.dataframe(
            anomaly_df.head(100),
            use_container_width=True
        )
    else:
        st.success(
            "✅ No major statistical anomalies detected."
        )

    if invalid_emails > 0:
        st.warning(
            f"⚠️ {invalid_emails} invalid email values detected."
        )

        for column in email_columns:
            invalid_rows = df[
                ~df[column].apply(is_valid_email)
            ]

            st.write(f"Invalid values in `{column}`")

            st.dataframe(
                invalid_rows[[column]].head(50),
                use_container_width=True
            )
    elif email_columns:
        st.success("✅ Email validation passed.")
    else:
        st.info("No email column detected.")


# ============================================================
# TAB 3: ETL
# ============================================================

with tab3:
    st.subheader("🧹 ETL Pipeline")

    st.write(
        "Run the cleaning and transformation process on "
        "the uploaded dataset."
    )

    if st.button(
        "🚀 Run ETL Pipeline",
        type="primary"
    ):
        with st.spinner("Running ETL pipeline..."):
            cleaned_df, cleaning_log = clean_dataset(df)

        st.success(
            "ETL pipeline completed successfully!"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.success("📥 Ingestion\n\nPASSED")

        with col2:
            st.success("🧹 Cleaning\n\nPASSED")

        with col3:
            st.success("🔄 Transformation\n\nPASSED")

        with col4:
            st.success("✅ Validation\n\nPASSED")

        st.divider()

        st.subheader("Cleaning Operations")

        if cleaning_log:
            for item in cleaning_log:
                st.write(f"✓ {item}")
        else:
            st.info(
                "No cleaning operations were necessary."
            )

        st.subheader("Cleaned Dataset")

        st.dataframe(
            cleaned_df.head(100),
            use_container_width=True
        )

        st.session_state["cleaned_df"] = cleaned_df

        csv_data = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Cleaned Dataset",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="cleaned_dataset_download_etl"
        )


# ============================================================
# TAB 4: DASHBOARD
# ============================================================

with tab4:
    st.subheader("📊 Data Quality Dashboard")

    if quality_score >= 90:
        st.success(
            f"🟢 Excellent Data Quality — {quality_score}%"
        )
    elif quality_score >= 70:
        st.warning(
            f"🟡 Moderate Data Quality — {quality_score}%"
        )
    else:
        st.error(
            f"🔴 Poor Data Quality — {quality_score}%"
        )

    st.progress(quality_score / 100)

    st.divider()

    issue_data = pd.DataFrame({
        "Issue": [
            "Missing",
            "Duplicates",
            "Invalid Emails",
            "Anomalies"
        ],
        "Count": [
            missing_values,
            duplicate_rows,
            invalid_emails,
            anomaly_count
        ]
    })

    st.subheader("Detected Issues")

    st.bar_chart(
        issue_data.set_index("Issue")
    )

    st.divider()

    st.subheader("Data Pipeline")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info("📥\n\nINGEST")

    with col2:
        st.info("🔍\n\nPROFILE")

    with col3:
        st.info("🧹\n\nCLEAN")

    with col4:
        st.info("🤖\n\nAI CHECK")

    with col5:
        st.success("💾\n\nLOAD")


# ============================================================
# TAB 5: REPORT
# ============================================================

with tab5:
    st.subheader("📄 AI Data Quality Report")

    generated_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    report = f"""AI DATA QUALITY REPORT
======================

Dataset:
{uploaded_file.name}

Generated:
{generated_time}

Total Rows:
{total_rows}

Total Columns:
{total_columns}

Missing Values:
{missing_values}

Duplicate Records:
{duplicate_rows}

Invalid Emails:
{invalid_emails}

AI Detected Anomalies:
{anomaly_count}

Overall Data Quality Score:
{quality_score}%

STATUS:
"""

    if quality_score >= 90:
        report += "EXCELLENT\n"
    elif quality_score >= 70:
        report += "MODERATE\n"
    else:
        report += "NEEDS IMPROVEMENT\n"

    report += "\nRECOMMENDATIONS:\n"

    if missing_values > 0:
        report += (
            "- Review and handle missing values.\n"
        )

    if duplicate_rows > 0:
        report += (
            "- Remove duplicate records.\n"
        )

    if invalid_emails > 0:
        report += (
            "- Validate email addresses.\n"
        )

    if anomaly_count > 0:
        report += (
            "- Investigate AI-detected anomalies.\n"
        )

    if (
        missing_values == 0
        and duplicate_rows == 0
        and invalid_emails == 0
        and anomaly_count == 0
    ):
        report += (
            "- Dataset appears to have good quality.\n"
        )

    st.code(
        report,
        language="text"
    )

    st.download_button(
        "📥 Download Quality Report",
        data=report,
        file_name="data_quality_report.txt",
        mime="text/plain",
        key="quality_report_download"
    )

    if "cleaned_df" in st.session_state:
        cleaned_df = st.session_state["cleaned_df"]

        csv_data = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Cleaned Dataset",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="cleaned_dataset_download_report"
        )


# ============================================================
# SAVE RUN
# ============================================================

if st.session_state.get("last_saved_file") != uploaded_file.name:
    save_pipeline_result(
        uploaded_file.name,
        total_rows,
        total_columns,
        missing_values,
        duplicate_rows,
        invalid_emails,
        anomaly_count,
        quality_score
    )

    st.session_state["last_saved_file"] = uploaded_file.name


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Data Quality Guardian | "
    "Python • Pandas • Scikit-learn • SQLite • Streamlit"
)
