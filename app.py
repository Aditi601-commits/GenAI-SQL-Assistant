import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal Data Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Universal Data Assistant")

# --- API SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("⚠️ API Key missing! Check your .streamlit/secrets.toml")
    st.stop()

# --- HELPER: LOAD CUSTOM DATA (ALL FORMATS) ---
def load_custom_data(uploaded_file):
    conn = sqlite3.connect(':memory:')
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # 1. READ DATA BASED ON FILE TYPE
        if file_ext == 'csv':
            # Peek to find header (Smart Detection from before)
            df_peek = pd.read_csv(uploaded_file, header=None, nrows=10)
            header_row = 0
            max_text = 0
            for i, row in df_peek.iterrows():
                cnt = row.apply(lambda x: isinstance(x, str)).sum()
                if cnt > max_text:
                    max_text = cnt
                    header_row = i
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=header_row)

        elif file_ext in ['xls', 'xlsx']:
            df_peek = pd.read_excel(uploaded_file, header=None, nrows=10, engine='openpyxl')
            header_row = 0
            max_text = 0
            for i, row in df_peek.iterrows():
                cnt = row.apply(lambda x: isinstance(x, str)).sum()
                if cnt > max_text:
                    max_text = cnt
                    header_row = i
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, header=header_row, engine='openpyxl')
            
        elif file_ext == 'json':
            # JSON is often nested, we try to flatten it
            df = pd.read_json(uploaded_file)
            
        elif file_ext == 'parquet':
            # Needs 'pip install pyarrow'
            df = pd.read_parquet(uploaded_file)
            
        elif file_ext in ['tsv', 'txt']:
            # Assume tab-separated for txt/tsv
            df = pd.read_csv(uploaded_file, sep='\t')
            
        else:
            return None, "Unsupported file format."

        # 2. CLEANUP: Remove Empty Cols & Rows
        # Remove columns named "Unnamed" or empty ones
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False, na=False)]
        df.dropna(axis=1, how='all', inplace=True)
        
        # 3. CLEAN COLUMN NAMES
        cleaned_columns = []
        for c in df.columns:
            clean_c = str(c).strip().replace(' ', '_').replace('.', '_').replace('-', '_')
            cleaned_columns.append(clean_c)
        df.columns = cleaned_columns
        
        # 4. SAFETY: Force objects/dates to strings to prevent SQL crashes
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        df.to_sql('uploaded_data', conn, index=False, if_exists='replace')
        return conn, df.columns.tolist()

    except Exception as e:
        return None, str(e)

# --- HELPER: INSIGHTS ---
def analyze_query_results(df, question):
    data_summary = df.head(10).to_string()
    prompt = f"""
    You are a Data Analyst. User asked: "{question}"
    Data found:
    {data_summary}
    
    TASK: Provide 3 short, sharp business insights.
    Format as bullet points.
    """
    model = genai.GenerativeModel('gemini-flash-latest')
    response = model.generate_content(prompt)
    return response.text

# --- HELPER: SQL GENERATION (WITH SURGICAL CLEANING) ---
def get_gemini_response(question, mode, schema_info):
    if mode == "default":
        context = """Tables: products, customers, sales. Revenue = SUM(total_amount)."""
    else:
        context = f"""Table: uploaded_data. Columns: {schema_info}"""

    prompt = f"""
    You are an expert SQL Assistant.
    {context}
    STRICT RULES:
    1. Return ONLY valid SQLite SQL.
    2. Use `LOWER(col) LIKE '%val%'` for text matching.
    3. Return "NO_SQL" if unrelated.
    4. Return ONLY code.
    """
    model = genai.GenerativeModel('gemini-flash-latest')
    response = model.generate_content([prompt, question])
    sql = response.text.strip()
    
    # 1. Remove Markdown (standard)
    sql = sql.replace("```sql", "").replace("```", "")
    
    # 2. SURGICAL CLEANUP (The Fix for "ite")
    # This finds exactly where "SELECT" starts and deletes everything before it.
    upper_sql = sql.upper()
    if "SELECT" in upper_sql:
        start_idx = upper_sql.find("SELECT")
        sql = sql[start_idx:]  # Keep only from SELECT onwards
    elif "WITH" in upper_sql:
        start_idx = upper_sql.find("WITH")
        sql = sql[start_idx:]

    return sql

# --- MAIN UI ---
st.sidebar.header("📂 Data Source")
# UPDATE: Added more file types here
uploaded_file = st.sidebar.file_uploader(
    "Upload Data File", 
    type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"]
)

mode = "default"
conn = None
schema_info = []

if uploaded_file:
    mode = "custom"
    st.info(f"Using: **{uploaded_file.name}**")
    conn, schema_or_error = load_custom_data(uploaded_file)
    if conn is None:
        st.error(f"Error: {schema_or_error}")
        st.stop()
    schema_info = schema_or_error 
else:
    mode = "default"
    if os.path.exists('database.db'):
        conn = sqlite3.connect('database.db')
    else:
        st.error("⚠️ Default database.db not found.")
        st.stop()

# --- SMART BUTTONS ---
st.write("### ⚡ Quick Actions")
col1, col2, col3 = st.columns(3)
def set_q(q): st.session_state.user_question = q

if mode == "default":
    with col1:
        if st.button("💰 Total Revenue", use_container_width=True): set_q("What is the total revenue?")
    with col2:
        if st.button("🏆 Top Products", use_container_width=True): set_q("Show top 5 expensive products")
    with col3:
        if st.button("📉 Sales by Category", use_container_width=True): set_q("Count sales by category")
else:
    # INTELLIGENT BUTTON LOGIC
    try:
        # Find a categorical column (Text based, not ID)
        cat_col = None
        for col in schema_info:
            if "ID" not in col.upper() and "DATE" not in col.upper() and "URL" not in col.upper():
                cat_col = col
                break
        if not cat_col: cat_col = schema_info[0]
        
        # Find a value column (last column usually)
        val_col = schema_info[-1] 

        with col1:
            if st.button(f"📊 Count by {cat_col}", use_container_width=True): 
                set_q(f"Count records for each {cat_col}")
        with col2:
            if st.button(f"📑 Top 5 by {val_col}", use_container_width=True): 
                set_q(f"Show top 5 records with highest {val_col}")
        with col3:
            if st.button("👀 Sample Data", use_container_width=True): 
                set_q("Show 5 random rows")
    except:
        st.warning("Could not generate buttons.")

# --- INPUT & ANALYSIS ---
if "user_question" not in st.session_state: st.session_state.user_question = ""
question = st.text_input("Ask a question:", key="user_question")

if st.button("Run Analysis", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                sql = get_gemini_response(question, mode, schema_info)
                if sql == "NO_SQL":
                    st.error("Cannot answer this.")
                    st.session_state.last_result = None
                else:
                    if "DROP" in sql.upper() or "DELETE" in sql.upper():
                        st.error("Read-only mode.")
                    else:
                        result = pd.read_sql_query(sql, conn)
                        if result.empty:
                            st.warning("No data found.")
                            st.session_state.last_result = None
                        else:
                            st.session_state.last_result = result
                            st.session_state.last_sql = sql
                            st.session_state.last_question = question
            except Exception as e:
                st.error(f"Error: {e}")

# --- DISPLAY ---
if "last_result" in st.session_state and st.session_state.last_result is not None:
    res = st.session_state.last_result
    st.success("Analysis Complete")
    st.dataframe(res)
    
    # Chart Logic
    if len(res.columns) == 2:
        try:
            clean = res.copy()
            clean.columns = ["Category", "Value"]
            clean["Value"] = pd.to_numeric(clean["Value"], errors='coerce')
            st.write("### 📊 Visualization")
            st.bar_chart(clean.set_index("Category"))
        except: pass

    st.markdown("---")
    if st.button("✨ Generate AI Insights"):
        with st.spinner("Analyzing..."):
            try:
                insights = analyze_query_results(res, st.session_state.last_question)
                st.markdown(insights)
            except: st.error("Quota reached.")

    with st.expander("🛠️ View SQL"): st.code(st.session_state.last_sql, language="sql")

if mode == "default" and conn: conn.close()