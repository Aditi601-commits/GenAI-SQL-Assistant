import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import time
from fpdf import FPDF
import re

def run_ml_pipeline(df):
    df = df.copy()

    # Drop missing values
    df = df.fillna(df.mean(numeric_only=True))

    # -------- AUTO TARGET DETECTION --------
    target_col = None
    for col in df.columns:
        name = col.lower()
        if any(x in name for x in ['target', 'label', 'class', 'sentiment', 'output']):
            target_col = col
            break

    if target_col is None:
        return None, None
    # ✅ Reduce classes (VERY IMPORTANT)
    df[target_col] = df[target_col].replace({
        "Very Positive": "Positive",
        "Very Negative": "Negative"
    })

    st.write("Target column:", target_col)
    st.write("Unique values:", df[target_col].unique())
    
    # -------- CLEAN TARGET COLUMN --------
    df[target_col] = df[target_col].astype(str).str.lower().str.strip()
    
    y = df[target_col]
    X = df.drop(columns=[target_col])

    df[target_col] = df[target_col].str.lower().str.strip()
    
    # -------- ENCODING --------
    from sklearn.preprocessing import LabelEncoder

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

    # -------- HANDLE DATETIME --------
    for col in X.columns:
        try:
            X[col] = pd.to_datetime(X[col])
            X[col] = X[col].astype('int64')
        except:
            pass

    # -------- FEATURE ENGINEERING --------
    for col in X.columns[:2]:
        X[f"{col}_squared"] = X[col] ** 2

    # -------- SCALING --------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------- TRAIN TEST SPLIT --------
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- MODEL --------
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')

    return acc, f1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sales Intelligence Hub", page_icon="📊", layout="centered")
st.title("📊 Sales Intelligence Hub")
st.markdown("Analyze your sales data with AI. Click a quick option or type your own question.")

# --- API SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("⚠️ API Key missing! Check your .streamlit/secrets.toml")
    st.stop()

# --- HELPER: FIXED PDF REPORT ---
def create_pdf_report(user_question, sql_query, insights, df_preview):
    class PDF(FPDF):
        def header(self):
            # 1. Draw the Background Rectangle (Height 20mm)
            self.set_fill_color(73, 7, 83)
            self.rect(0, 0, 210, 20, 'F')
            
            # 2. FORCE cursor position to start inside the rect
            self.set_y(5) 
            
            # 3. Write Text
            self.set_font('Arial', 'B', 16)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, 'Executive Analysis Report', 0, 1, 'C')
            self.ln(10) # Add space after header

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Universal Data Assistant | Page {self.page_no()}', 0, 0, 'C')

        def section_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(240, 240, 240)
            self.set_text_color(0, 0, 0)
            self.cell(0, 8, f"  {title}", 0, 1, 'L', fill=True)
            self.ln(4)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 6, body)
            self.ln(5)

    def clean_text(text):
        if text: return text.encode('latin-1', 'replace').decode('latin-1')
        return ""

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Question
    pdf.section_title("1. Analysis Request")
    pdf.chapter_body(clean_text(f'Question: "{user_question}"'))

    # 2. Insights
    if insights:
        pdf.section_title("2. Key Strategic Insights")
        pdf.chapter_body(clean_text(insights.replace('*', '').replace('#', '')))

    # 3. Data Table (Fixed Overlap)
    pdf.section_title("3. Data Evidence (Top 10 Rows)")
    pdf.set_font("Courier", size=8) # Smaller font for table
    
    cols = df_preview.columns.tolist()
    if cols:
        page_width = 190
        col_width = page_width / len(cols)
        
        # Header
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Arial", 'B', 8)
        for col in cols:
            # Truncate to 12 chars to prevent overlap
            pdf.cell(col_width, 8, clean_text(str(col)[:12]), 1, 0, 'C', fill=True)
        pdf.ln()
        
        # Rows
        pdf.set_font("Arial", '', 8)
        for _, row in df_preview.head(10).iterrows():
            for col in cols:
                pdf.cell(col_width, 8, clean_text(str(row[col])[:12]), 1, 0, 'C')
            pdf.ln()
    
    pdf.ln(10)

    # 4. SQL
    pdf.section_title("4. Technical Appendix (SQL)")
    pdf.set_font("Courier", size=9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, clean_text(sql_query))

    return pdf.output(dest='S').encode('latin-1')

# --- UPDATED HELPER: LOAD & CLEAN & MASK ---
def process_uploaded_file(uploaded_file):
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        header_row = 0
        if file_ext in ['csv', 'xls', 'xlsx']:
            try:
                if file_ext == 'csv': df_peek = pd.read_csv(uploaded_file, header=None, nrows=20)
                else: df_peek = pd.read_excel(uploaded_file, header=None, nrows=20, engine='openpyxl')
                max_filled = 0
                for i, row in df_peek.iterrows():
                    if row.count() > max_filled:
                        max_filled = row.count()
                        header_row = i
            except: header_row = 0
            
            uploaded_file.seek(0)
            if file_ext == 'csv': df = pd.read_csv(uploaded_file, header=header_row)
            else: df = pd.read_excel(uploaded_file, header=header_row, engine='openpyxl')
        elif file_ext == 'json':
            df = pd.read_json(uploaded_file)
        else:
            return None, "Unsupported file format."
        
        # Basic Cleaning
        
        df.dropna(axis=1, how='all', inplace=True)
        new_cols = []
        for i, col in enumerate(df.columns):
            c_str = str(col).strip()
            if "Unnamed" in c_str or c_str == "" or c_str.lower() == "nan": new_cols.append(f"Column_{i+1}")
            else: new_cols.append(c_str.replace(' ', '_').replace('.', '').replace('-', '_').replace('\n', ''))
        df.columns = new_cols
        
        for col in df.columns:
            if df[col].dtype == 'object': df[col] = df[col].astype(str)
            
        return df, None
    except Exception as e:
        return None, str(e)

# --- NEW FUNCTION: PII MASKING ---
def mask_pii(df):
    df_masked = df.copy()
    # Regex patterns for Email and Phone (Basic)
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    masked_count = 0
    
    for col in df_masked.columns:
        if df_masked[col].dtype == 'object': # Only check text columns
            # Check if column looks like PII
            sample = " ".join(df_masked[col].astype(str).head(10).tolist())
            if re.search(email_pattern, sample) or re.search(phone_pattern, sample) or "email" in col.lower() or "phone" in col.lower():
                df_masked[col] = "*****" # REDACT
                masked_count += 1
                
    return df_masked, masked_count

# --- HELPER: SQL SYNC ---
def push_to_sqlite(df):
    conn = sqlite3.connect(':memory:')
    df.to_sql('uploaded_data', conn, index=False, if_exists='replace')
    return conn

# --- GEMINI & INSIGHTS HELPERS ---
def get_gemini_response(question, schema_info, previous_context=None):
    cols = ', '.join(schema_info[:50]) 
    schema_str = f"Table: uploaded_data. Columns: {cols}"
    history_str = ""
    if previous_context:
        history_str = f"PREVIOUS: Q='{previous_context['question']}' SQL='{previous_context['sql']}'. If follow-up, modify SQL."

# --- HELPER FUNCTIONS ---
def analyze_query_results(df, question):
    data_summary = df.head(10).to_string()
    prompt = f"""
    You are a Data Analyst. User asked: "{question}"
    Data found:
    {data_summary}
    
    TASK: Provide 3 short, sharp business insights based on this data.
    Format as bullet points.
    """
    for _ in range(3):
        try:
            model = genai.GenerativeModel('gemini-flash-lite-latest')
            response = model.generate_content([prompt])
            sql = response.text.strip().replace("```sql", "").replace("```sqlite", "").replace("```", "")
            if "SELECT" in sql.upper() or "WITH" in sql.upper(): return sql
        except: time.sleep(1)
    return "API Quota Exceeded or Error Generating SQL."

def generate_insights(df, question):
    data_preview = df.head(20).to_string(index=False)
    prompt = f"""
    Role: Senior Analyst. Question: "{question}"
    Task: Write crisp bullet points for each of the following 3 headings : 1. 3 Key Trends. 2. Outliers. 3. 1 Strategic Action.
    """
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    response = model.generate_content([prompt, question])
    return response.text.strip().replace("```sql", "").replace("```", "")

def execute_query(sql_query):
    if not os.path.exists('database.db'):
        return "⚠️ Database not found. Run setup_database.py."
    conn = sqlite3.connect('database.db')
    try:
        model = genai.GenerativeModel('gemini-flash-lite-latest')
        return model.generate_content([prompt]).text
    except Exception as e: return f"Error: {e}"

# --- INITIALIZE STATE ---
if "user_question" not in st.session_state: st.session_state.user_question = ""
if "last_result" not in st.session_state: st.session_state.last_result = None
if "last_sql" not in st.session_state: st.session_state.last_sql = None
if "last_question" not in st.session_state: st.session_state.last_question = None
if "last_insights" not in st.session_state: st.session_state.last_insights = ""
if "show_chart" not in st.session_state: st.session_state.show_chart = False
if "active_df" not in st.session_state: st.session_state.active_df = None
if "file_id" not in st.session_state: st.session_state.file_id = None

def set_q(q): st.session_state.user_question = q

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🤖 **DATA**-ASSISTANT") 
    st.caption("Pro Edition - Powered by Gemini")
    st.markdown("---")
    
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")
    
    if uploaded_file:
        if st.session_state.file_id != uploaded_file.file_id:
            df_new, error = process_uploaded_file(uploaded_file)
            if df_new is not None:
                st.session_state.active_df = df_new.copy()
                st.session_state.raw_df = df_new.copy()
                st.session_state.file_id = uploaded_file.file_id
                st.session_state.last_result = None
            else:
                st.error(error)
        
        st.success(f"Loaded: {uploaded_file.name}")
        st.caption("✅ Database Active")
    else:
        st.info("👆 Upload a file to start.")
        
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **ETL Layer:** Uploads & cleans raw data (CSV/XLS).
        2. **SQL Layer:** Gemini AI converts Natural Language -> SQL.
        3. **Execution:** Runs query on in-memory SQLite DB.
        4. **Visualization:** Plotly & Seaborn for dynamic charts.
        5. **Reporting:** FPDF generates downloadable insights.
        """)


# --- MAIN APP ---
conn = None
schema_info = []

if st.session_state.active_df is not None:
    conn = push_to_sqlite(st.session_state.active_df)
    schema_info = st.session_state.active_df.columns.tolist()
    
    # --- DATA DOCTOR & PREVIEW ---
    with st.expander("🛠️ Data Doctor & Preview (Click to Expand)", expanded=False):
        tab_doc, tab_raw, tab_clean = st.tabs([
            "🩺 Data Health Check",
            "🔍 Raw Data",
            "✨ Cleaned Data"
        ])
        
    with tab_raw:
        st.dataframe(st.session_state.raw_df, use_container_width=True)

    with tab_clean:
        st.dataframe(st.session_state.active_df, use_container_width=True)

    with tab_doc:
        # your existing health check code
            
        with tab_doc:
            df = st.session_state.active_df
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", len(df))
            c2.metric("Missing", df.isnull().sum().sum(), delta_color="inverse")
            c3.metric("Duplicates", df.duplicated().sum(), delta_color="inverse")
            
            st.markdown("### 🛡️ Privacy & Quality Tools")
            
            fx1, fx2, fx3, fx4 = st.columns(4) # Added 4th column
            
            with fx1:
                if st.button("🧼 Remove Duplicates"):
                    st.session_state.active_df = df.drop_duplicates()
                    st.toast("Duplicates removed!", icon="✅")
                    time.sleep(1.5)
                    st.rerun()
            with fx2:
                if st.button("🩹 Fill missing rows"):
                    num_cols = df.select_dtypes(include=['number']).columns
                    df[num_cols] = df[num_cols].fillna(0)
                    st.session_state.active_df = df
                    st.toast("Missing rows filled!", icon="✅")
                    time.sleep(1.5)
                    st.rerun()
            with fx3:
                if st.button("✂️ Drop missing rows"):
                    st.session_state.active_df = df.dropna()
                    st.toast("Missing rows dropped!", icon="✅")
                    time.sleep(1.5)
                    st.rerun()
            
            # --- NEW PRIVACY BUTTON ---
            with fx4:
                if st.button("🕵️ Mask PII"):
                    masked_df, count = mask_pii(df)
                    if count > 0:
                        st.success(f"🔒 Redacted {count} sensitive columns!")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.info("No PII detected.")

    # --- KPI HEADER ---
    st.subheader("📊 Data Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    curr_df = st.session_state.active_df
    kpi1.metric("Rows", f"{len(curr_df):,}")
    kpi2.metric("Cols", len(curr_df.columns))
    kpi3.metric("Missing", f"{curr_df.isnull().sum().sum():,}")
    kpi4.metric("Duplicates", curr_df.duplicated().sum())
    st.markdown("---")
    
    # --- CHAT & RESULTS ---
    col_chat, col_results = st.columns([1, 2])

    with col_chat:
        st.subheader("💬 Ask Your Data")
        qa1, qa2, qa3 = st.columns(3)
        if len(schema_info) > 0:
            try:
                cat_col = next((col for col in schema_info if "ID" not in col.upper()), schema_info[0])
                with qa1: 
                    if st.button(f"🔢 Count"): set_q(f"Count records by {cat_col}")
                with qa2: 
                    if st.button("👀 Sample"): set_q("Show 5 random rows")
                with qa3: 
                    if st.button("📑 Summary"): set_q("Show count of all rows")
            except: pass
        
        question = st.text_input("Type your question:", key="user_question")
        
        if st.button("🚀 Run Analysis", type="primary"):
            if not question:
                st.warning("Enter a question first.")
            else:
                st.session_state.show_chart = False
                with st.spinner("Analyzing..."):
                    try:
                        prev_ctx = None
                        if st.session_state.last_sql and st.session_state.last_question:
                            prev_ctx = {"question": st.session_state.last_question, "sql": st.session_state.last_sql}
                        
                        sql = get_gemini_response(question, schema_info, prev_ctx)
                        
                        # B. Show Chart (if applicable)
                        if len(result.columns) == 2:
                            st.bar_chart(result.set_index(result.columns[0]))

                        # C. Insights (Closed by default)
                        with st.spinner("Generating insights..."):
                            insights = analyze_query_results(result, question)
                            
                        # 'expanded=False' keeps it closed
                        with st.expander(f"💡 View AI Insights for: '{question}'", expanded=False):
                            st.markdown(insights)
                            
                        # D. SQL (Closed by default)
                        with st.expander("🛠️ View Technical Details (SQL)", expanded=False):
                            st.code(sql, language="sql")
                else:
                    st.info("Visualizations need at least 2 columns.")
            
            with tab_insight:
                # Insights are already generated, just display them
                st.markdown("### 🧠 AI Analysis")
                st.markdown(insights)
            
            with t2:
                st.subheader("📊 Standard Charts")
                if len(result.columns) >= 2:
                    plot_df = result.copy()
                    num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
                    cat_cols = plot_df.select_dtypes(exclude=['number']).columns.tolist()
                    
                    vc1, vc2, vc3 = st.columns(3)
                    with vc1: c_type = st.selectbox("Type", ["Bar", "Line", "Scatter", "Pie", "Histogram"], key="vt")
                    with vc2: 
                        dx = cat_cols[0] if cat_cols else plot_df.columns[0]
                        x_ax = st.selectbox("X-Axis", plot_df.columns, index=plot_df.columns.get_loc(dx), key="vx")
                    with vc3: 
                        dy = num_cols[0] if num_cols else plot_df.columns[1]
                        if c_type in ["Pie", "Histogram"]: y_ax = st.selectbox("Y-Axis", [None] + list(plot_df.columns), key="vy_opt")
                        else: y_ax = st.selectbox("Y-Axis", plot_df.columns, index=plot_df.columns.get_loc(dy), key="vy")
                    
                    if st.button("Generate Chart"): st.session_state.show_chart = True
                    
                    if st.session_state.show_chart:
                        try:
                            if c_type == "Bar": fig = px.bar(plot_df, x=x_ax, y=y_ax, template="plotly_dark")
                            elif c_type == "Line": fig = px.line(plot_df, x=x_ax, y=y_ax, template="plotly_dark")
                            elif c_type == "Scatter": fig = px.scatter(plot_df, x=x_ax, y=y_ax, template="plotly_dark")
                            elif c_type == "Pie": fig = px.pie(plot_df, names=x_ax, values=y_ax, template="plotly_dark")
                            elif c_type == "Histogram": fig = px.histogram(plot_df, x=x_ax, y=y_ax, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e: st.error(f"Viz Error: {e}")
                else: st.info("Need 2+ columns.")

                st.markdown("---")
                st.subheader("🔥 Advanced: Heatmap")
                if len(result.select_dtypes(include=['number']).columns) > 1:
                    if st.button("Generate Heatmap"):
                        try:
                            corr = result.select_dtypes(include=['number']).corr()
                            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                            st.plotly_chart(fig, use_container_width=True)
                        except: st.error("Heatmap failed.")
                else: st.warning("Need 2+ numeric columns.")

            with t3: st.markdown(st.session_state.last_insights)
            with t4: st.code(st.session_state.last_sql, language="sql")
            
            # --- ML VALIDATION SECTION ---
            st.markdown("---")
            st.subheader("🧪 ML Validation (Data Quality Check)")

            if st.button("Run ML Validation"):
                raw_df = st.session_state.get("raw_df")
                processed_df = st.session_state.get("active_df")
                
                if raw_df is None:
                    st.warning("Upload data first.")
                else:
                    with st.spinner("Running ML validation..."):
                        results = []
                        
                        raw_acc, raw_f1 = run_ml_pipeline(raw_df)
                        if raw_acc is not None:
                            results.append(("Raw Dataset", raw_acc, raw_f1))
                            
                        clean_acc, clean_f1 = run_ml_pipeline(processed_df)
                        if clean_acc is not None:
                            results.append(("Cleaned Dataset", clean_acc, clean_f1))
                                
                        if results:
                            df_results = pd.DataFrame(results, columns=["Dataset", "Accuracy", "F1 Score"])
                            st.dataframe(df_results, use_container_width=True)
                            st.bar_chart(df_results.set_index("Dataset"))
                        if len(results) == 2:
                            improvement = (clean_acc - raw_acc) * 100
                            st.success(f"📈 Improvement after cleaning: {improvement:.2f}%")
                        else:
                            st.error("No valid target column found (need target/label/sentiment column).")
                
else:              
    # --- MODERN LANDING PAGE ---
    st.markdown("""
    <style>
        .landing-header {
            text-align: center;
            padding: 4rem 1rem;
            background: linear-gradient(180deg, rgba(73,7,83,0.2) 0%, rgba(14,17,23,0) 100%);
            border-radius: 20px;
            margin-bottom: 2rem;
        }
        .landing-title {
            font-size: 3rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #FAFAFA, #D900FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .landing-subtitle {
            font-size: 1.2rem;
            color: #b0b0b0;
            max-width: 600px;
            margin: 0 auto;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 3rem;
        }
        .feature-card {
            background-color: #262730;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #464b5f;
            text-align: left;
            transition: transform 0.2s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: #D900FF;
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            display: block;
        }
        .feature-title {
            font-weight: bold;
            color: #fff;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .feature-desc {
            color: #a0a0a0;
            font-size: 0.9rem;
            line-height: 1.5;
        }
    </style>

    <div class="landing-header">
        <div class="landing-title">Data Analysis, Reimagined.</div>
        <div class="landing-subtitle">Stop wrestling with spreadsheets. Upload your data and let AI handle the cleaning, SQL querying, and visualization for you.</div>
    </div>

    <div class="feature-grid">
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">AI-Powered Analytics</div>
            <div class="feature-desc">Powered by Gemini 2.0. Just ask questions in plain English and get SQL-accurate answers instantly.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🩺</span>
            <div class="feature-title">Smart Data Doctor</div>
            <div class="feature-desc">Automatically detect missing values and duplicates. Fix messy datasets with a single click.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <div class="feature-title">Dynamic Visualization</div>
            <div class="feature-desc">Create Bar Charts, Line Graphs, and Heatmaps on the fly without writing a single line of code.</div>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📑</span>
            <div class="feature-title">Executive Reporting</div>
            <div class="feature-desc">Turn your insights into professional PDF reports ready for your next management meeting.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if conn: conn.close()