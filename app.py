import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal Data Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Universal Data Assistant")

# --- API SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("⚠️ API Key missing! Check your .streamlit/secrets.toml")
    st.stop()

# --- HELPER: LOAD CUSTOM DATA (Widest Row Strategy) ---
def load_custom_data(uploaded_file):
    conn = sqlite3.connect(':memory:')
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # 1. SMART HEADER DETECTION
        header_row = 0
        if file_ext in ['csv', 'xls', 'xlsx']:
            try:
                if file_ext == 'csv':
                    df_peek = pd.read_csv(uploaded_file, header=None, nrows=20)
                else:
                    df_peek = pd.read_excel(uploaded_file, header=None, nrows=20, engine='openpyxl')
                
                # Find the row with the MOST non-empty cells
                max_filled_cols = 0
                for i, row in df_peek.iterrows():
                    filled_count = row.count()
                    if filled_count > max_filled_cols:
                        max_filled_cols = filled_count
                        header_row = i
            except:
                header_row = 0
            
            uploaded_file.seek(0)
            
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file, header=header_row)
            else:
                df = pd.read_excel(uploaded_file, header=header_row, engine='openpyxl')
                
        elif file_ext == 'json':
            df = pd.read_json(uploaded_file)
        else:
            return None, "Unsupported file format."

        # 2. CLEANUP COLUMN NAMES
        new_cols = []
        for i, col in enumerate(df.columns):
            c_str = str(col).strip()
            if "Unnamed" in c_str or c_str == "" or c_str.lower() == "nan":
                new_cols.append(f"Col_{i+1}")
            else:
                clean = c_str.replace(' ', '_').replace('.', '').replace('-', '_').replace('\n', '')
                new_cols.append(clean)
        
        df.columns = new_cols
        
        # 3. SAFETY: Convert objects to strings
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        df.to_sql('uploaded_data', conn, index=False, if_exists='replace')
        return conn, df.columns.tolist()

    except Exception as e:
        return None, str(e)

# --- HELPER: SQL GENERATION ---
def get_gemini_response(question, schema_info):
    # Limit schema context to first 50 columns to save tokens
    cols = ', '.join(schema_info[:50]) 
    context = f"Table: uploaded_data. Columns: {cols}"

    prompt = f"""
    Context: {context}
    User Question: "{question}"
    
    Task: Write a valid SQLite SQL query.
    1. Return ONLY the SQL code. No markdown.
    2. If unsure, SELECT first 10 rows.
    3. Use LOWER(col) LIKE '%val%' for text search.
    """
    
    # Retry Logic
    for attempt in range(3):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([prompt])
            sql = response.text.strip()
            
            # Cleaning
            sql = sql.replace("```sql", "").replace("```sqlite", "").replace("```", "")
            
            if "SELECT" in sql.upper():
                return sql[sql.upper().find("SELECT"):]
            elif "WITH" in sql.upper():
                return sql[sql.upper().find("WITH"):]
            
            if len(sql) > 10: return sql
            
        except:
            time.sleep(1)
            
    return "SELECT * FROM uploaded_data LIMIT 10"

# --- SIDEBAR UI ---
st.sidebar.header("📂 Data Source")

st.sidebar.markdown(
    """<div style="margin-bottom: 10px;">
        <p style="font-weight: bold; margin-bottom: 5px;">Upload Data File</p>
        <p style="font-size: 0.8em; color: #888;">Accepted: CSV, Excel, JSON</p>
    </div>""", unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")

# --- MAIN LOGIC ---
conn = None
schema_info = []

if uploaded_file:
    st.info(f"Using: **{uploaded_file.name}**")
    conn, schema_or_error = load_custom_data(uploaded_file)
    
    if conn is None:
        st.error(f"Error loading file: {schema_or_error}")
        st.stop()
    else:
        schema_info = schema_or_error
else:
    # LANDING PAGE (Shown when no file is uploaded)
    st.info("👋 Welcome! Please upload a CSV or Excel file in the sidebar to begin analyzing your data.")
    st.stop() # Stops the rest of the app from running

# --- QUICK ACTIONS ---
st.write("### ⚡ Quick Actions")
col1, col2, col3 = st.columns(3)

if "user_question" not in st.session_state:
    st.session_state.user_question = ""

def set_q(q): st.session_state.user_question = q

# Only generate buttons if we have schema info
if len(schema_info) > 0:
    try:
        # Smart detection: Find the first text column that isn't an ID or Number
        cat_col = next((col for col in schema_info if "ID" not in col.upper() and "NUM" not in col.upper()), schema_info[0])
        
        with col1:
            if st.button(f"📊 Count by {cat_col}", use_container_width=True): 
                set_q(f"Count records by {cat_col}")
        with col2:
            if st.button("👀 Sample Data", use_container_width=True): 
                set_q("Show 5 random rows")
        with col3:
            if st.button("📑 Data Summary", use_container_width=True): 
                set_q("Show count of all rows")
    except: pass

# --- ANALYSIS SECTION ---
question = st.text_input("Ask a question about your data:", value=st.session_state.user_question)

if st.button("Run Analysis", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state.user_question = question 
        with st.spinner("Analyzing..."):
            try:
                # 1. Get SQL
                sql = get_gemini_response(question, schema_info)
                
                # 2. Run SQL
                result = pd.read_sql_query(sql, conn)
                
                if result.empty:
                    st.warning("No data found for that query.")
                else:
                    st.success("Analysis Complete!")
                    
                    # TABS
                    tab1, tab2 = st.tabs(["📊 Data & Visuals", "📜 SQL Query"])
                    
                    with tab1:
                        st.dataframe(result, use_container_width=True)
                        
                        # Download CSV
                        csv = result.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Results (CSV)", csv, "analysis_results.csv", "text/csv")

                        # Auto-Chart
                        if len(result.columns) == 2:
                            clean = result.copy()
                            col_x = clean.columns[0]
                            col_y = clean.columns[1]
                            clean[col_y] = pd.to_numeric(clean[col_y], errors='coerce')
                            
                            st.markdown("### 📈 Visual Trends")
                            # If X axis looks like a date/year, use Line Chart
                            if any(x in col_x.lower() for x in ["date", "year", "time", "month"]):
                                st.line_chart(clean.set_index(col_x))
                            else:
                                st.bar_chart(clean.set_index(col_x))

                    with tab2:
                        st.code(sql, language="sql")
                            
            except Exception as e:
                st.error(f"Error: {e}")

if conn: conn.close()