import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import time

# --- PAGE CONFIGURATION (Full Width, Custom Title) ---
st.set_page_config(
    page_title="Universal Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN DASHBOARD LOOK ---
st.markdown("""
    <style>
    /* Adjust main container padding to fix title cut-off */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b5f;
        padding: 10px;
        border-radius: 10px;
        color: white;
    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
    /* Section Headers */
    h3 {
        margin-top: 0px !important;
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- API SETUP ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except:
    st.error("⚠️ API Key missing! Check your .streamlit/secrets.toml")
    st.stop()

# --- HELPER: LOAD CUSTOM DATA ---
def load_custom_data(uploaded_file):
    conn = sqlite3.connect(':memory:')
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        header_row = 0
        if file_ext in ['csv', 'xls', 'xlsx']:
            try:
                if file_ext == 'csv':
                    df_peek = pd.read_csv(uploaded_file, header=None, nrows=20)
                else:
                    df_peek = pd.read_excel(uploaded_file, header=None, nrows=20, engine='openpyxl')
                
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
            return None, None, "Unsupported file format."

        df.dropna(axis=1, how='all', inplace=True)

        new_cols = []
        for i, col in enumerate(df.columns):
            c_str = str(col).strip()
            if "Unnamed" in c_str or c_str == "" or c_str.lower() == "nan":
                new_cols.append(f"Column_{i+1}")
            else:
                clean = c_str.replace(' ', '_').replace('.', '').replace('-', '_').replace('\n', '')
                new_cols.append(clean)
        
        df.columns = new_cols
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        df.to_sql('uploaded_data', conn, index=False, if_exists='replace')
        return conn, df, df.columns.tolist()

    except Exception as e:
        return None, None, str(e)

# --- HELPER: SQL GENERATION ---
def get_gemini_response(question, schema_info, previous_context=None):
    cols = ', '.join(schema_info[:50]) 
    schema_str = f"Table: uploaded_data. Columns: {cols}"

    history_str = ""
    if previous_context:
        history_str = f"""
        PREVIOUS QUERY CONTEXT:
        - User's Last Question: "{previous_context['question']}"
        - Last SQL Generated: "{previous_context['sql']}"
        
        INSTRUCTION: 
        If the current question is a follow-up (e.g., "now filter by...", "only show...", "remove..."), 
        MODIFY the Last SQL to satisfy the new request.
        If it is a new unrelated question, IGNORE the previous context and write a new query.
        """

    prompt = f"""
    Database Schema: {schema_str}
    
    {history_str}
    
    Current User Question: "{question}"
    
    Task: Write a valid SQLite SQL query.
    1. Return ONLY the SQL code. No markdown.
    2. STRICTLY FOLLOW any row limits requested (e.g., "5 rows").
    3. If the user does NOT specify a limit, default to LIMIT 10.
    4. Use LOWER(col) LIKE '%val%' for text search.
    """
    
    last_error = ""
    for attempt in range(3):
        try:
            model = genai.GenerativeModel('gemini-flash-latest')
            response = model.generate_content([prompt])
            sql = response.text.strip()
            sql = sql.replace("```sql", "").replace("```sqlite", "").replace("```", "")
            
            if "SELECT" in sql.upper():
                return sql[sql.upper().find("SELECT"):]
            elif "WITH" in sql.upper():
                return sql[sql.upper().find("WITH"):]
            
            if len(sql) > 10: return sql
            
        except Exception as e:
            last_error = str(e)
            time.sleep(1)
            
    return f"ERROR: {last_error}"

# --- HELPER: BUSINESS INTELLIGENCE ---
def generate_insights(df, question):
    data_preview = df.head(20).to_string(index=False)
    
    prompt = f"""
    You are a Senior Business Analyst. 
    User Question: "{question}"
    Data Result (First 20 rows):
    {data_preview}
    
    Task: Provide a brief, professional analysis.
    1. Identify 3 Key Trends or Patterns.
    2. Highlight any Outliers or Anomalies.
    3. Suggest 1 Strategic Action based on this data.
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# --- INITIALIZE SESSION STATE ---
if "user_question" not in st.session_state: st.session_state.user_question = ""
if "last_result" not in st.session_state: st.session_state.last_result = None
if "last_sql" not in st.session_state: st.session_state.last_sql = None
if "last_question" not in st.session_state: st.session_state.last_question = None
if "show_chart" not in st.session_state: st.session_state.show_chart = False

def set_q(q): st.session_state.user_question = q

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🤖 Universal Data Assistant")
    st.markdown("---")
    
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")
    
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")
        st.markdown("---")
        st.caption("✅ Gemini AI Connected")
        st.caption("✅ Database In-Memory Active")
    else:
        st.info("👆 Upload a file to start.")

# --- MAIN PAGE LOGIC ---
conn = None
schema_info = []
df_full = None

if uploaded_file:
    conn, df_full, schema_or_error = load_custom_data(uploaded_file)
    if conn is None:
        st.error(f"Error: {schema_or_error}")
        st.stop()
    else:
        schema_info = schema_or_error
    
    # --- NEW: DATA PREVIEW SECTION ---
    with st.expander("🔍 Preview Raw Data (Click to Expand)"):
        st.dataframe(df_full, use_container_width=True)
        
    # --- DASHBOARD HEADER: KPI METRICS ---
    st.subheader("📊 Data Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    row_count = len(df_full)
    col_count = len(df_full.columns)
    missing_val = df_full.isnull().sum().sum()
    duplicate_count = df_full.duplicated().sum()

    kpi1.metric("Total Rows", f"{row_count:,}")
    kpi2.metric("Total Columns", col_count)
    kpi3.metric("Missing Values", f"{missing_val:,}")
    kpi4.metric("Duplicates", duplicate_count)
    
    st.markdown("---")

    # --- MAIN INTERFACE: SPLIT LAYOUT ---
    col_chat, col_results = st.columns([1, 2])

    with col_chat:
        st.subheader("💬 Ask Your Data")
        
        # Quick Actions
        st.markdown("**Quick Actions:**")
        qa1, qa2, qa3 = st.columns(3)
        if len(schema_info) > 0:
            try:
                cat_col = next((col for col in schema_info if "ID" not in col.upper() and "NUM" not in col.upper()), schema_info[0])
                with qa1:
                    if st.button(f"🔢 Count by {cat_col}"): set_q(f"Count records by {cat_col}")
                with qa2:
                    if st.button("👀 Show Sample"): set_q("Show 5 random rows")
                with qa3:
                    if st.button("📑 Summarize"): set_q("Show count of all rows")
            except: pass
        
        # Input Area (Bound to session state)
        question = st.text_input("Type your question here:", key="user_question")
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not question:
                st.warning("Please enter a question.")
            else:
                st.session_state.show_chart = False
                with st.spinner("Analyzing..."):
                    try:
                        # Prepare Context
                        previous_context = None
                        if st.session_state.last_sql is not None and st.session_state.last_question is not None:
                            previous_context = {
                                "question": st.session_state.last_question,
                                "sql": st.session_state.last_sql
                            }

                        # AI Logic
                        sql = get_gemini_response(question, schema_info, previous_context)
                        
                        if sql.startswith("ERROR:"):
                            st.error(sql)
                            st.session_state.last_result = None
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
                        st.session_state.last_result = None

    # --- RESULTS COLUMN ---
    with col_results:
        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            sql = st.session_state.last_sql
            
            # TABS for organized viewing
            tab_table, tab_viz, tab_insight, tab_sql = st.tabs(["📄 Data Table", "📊 Visualization", "🧠 AI Insights", "📜 SQL"])
            
            with tab_table:
                st.dataframe(result, use_container_width=True)
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
            
            with tab_viz:
                if len(result.columns) >= 2:
                    plot_df = result.copy()
                    numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = plot_df.select_dtypes(exclude=['number']).columns.tolist()
                    
                    vc1, vc2, vc3 = st.columns(3)
                    with vc1:
                        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram"], key="viz_type")
                    with vc2:
                        default_x = categorical_cols[0] if categorical_cols else plot_df.columns[0]
                        x_axis = st.selectbox("X-Axis", plot_df.columns, index=plot_df.columns.get_loc(default_x), key="viz_x")
                    with vc3:
                        default_y = numeric_cols[0] if numeric_cols else plot_df.columns[1]
                        if chart_type in ["Pie", "Histogram"]:
                            y_axis = st.selectbox("Y-Axis", [None] + list(plot_df.columns), key="viz_y_opt")
                        else:
                            y_axis = st.selectbox("Y-Axis", plot_df.columns, index=plot_df.columns.get_loc(default_y), key="viz_y")

                    if st.button("Generate Chart"):
                        st.session_state.show_chart = True

                    if st.session_state.show_chart:
                        try:
                            if chart_type == "Bar":
                                fig = px.bar(plot_df, x=x_axis, y=y_axis, template="plotly_dark")
                            elif chart_type == "Line":
                                fig = px.line(plot_df, x=x_axis, y=y_axis, template="plotly_dark")
                            elif chart_type == "Scatter":
                                fig = px.scatter(plot_df, x=x_axis, y=y_axis, template="plotly_dark")
                            elif chart_type == "Pie":
                                fig = px.pie(plot_df, names=x_axis, values=y_axis, template="plotly_dark")
                            elif chart_type == "Histogram":
                                fig = px.histogram(plot_df, x=x_axis, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Viz Error: {e}")
                else:
                    st.info("Visualizations need at least 2 columns.")
            
            with tab_insight:
                if st.button("✨ Generate AI Analysis"):
                    with st.spinner("Thinking..."):
                        insights = generate_insights(result, st.session_state.last_question)
                        st.markdown(insights)
            
            with tab_sql:
                st.code(sql, language="sql")
        
        else:
            st.info("👈 Use the chat on the left to query your data. Results will appear here.")
            st.markdown("""
                **Tips for better results:**
                * Be specific (e.g., "Show top 10 sales by country")
                * Use filters (e.g., "Only where quantity > 5")
                * Ask follow-up questions!
            """)

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🤖 Universal Data Assistant</h1>
        <p>Your AI-powered partner for data analysis.</p>
        <p>Upload a CSV or Excel file on the left to get started.</p>
    </div>
    """, unsafe_allow_html=True)

if conn: conn.close()