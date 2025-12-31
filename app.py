import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import time
from fpdf import FPDF  # <--- NEW IMPORT FOR PDF

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
    /* FIX: Tab Hover Visibility */
    button[data-baseweb="tab"] {
        color: #e0e0e0 !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: #490753 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background-color: transparent !important;
        border-top: 2px solid #D900FF;
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

# --- HELPER: PDF GENERATION ---
def create_pdf_report(user_question, sql_query, insights, df_preview):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Universal Data Assistant - Analysis Report', 0, 1, 'C')
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # Helper to clean text for FPDF (handles special chars/emojis)
    def clean_text(text):
        if text:
            return text.encode('latin-1', 'replace').decode('latin-1')
        return ""

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # 1. User Question
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Analysis Request:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, clean_text(f'"{user_question}"'))
    pdf.ln(5)

    # 2. AI Insights
    if insights:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "2. AI Strategic Insights:", ln=True)
        pdf.set_font("Arial", size=11)
        # Strip Markdown * characters for cleaner PDF text
        clean_insights = insights.replace('*', '').replace('#', '')
        pdf.multi_cell(0, 8, clean_text(clean_insights))
        pdf.ln(5)

    # 3. SQL Query
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Technical Query (SQL):", ln=True)
    pdf.set_font("Courier", size=10)
    pdf.multi_cell(0, 8, clean_text(sql_query))
    pdf.ln(5)

    # 4. Data Snapshot
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "4. Data Snapshot (Top 10 Rows):", ln=True)
    pdf.set_font("Courier", size=8)
    
    # Table Header
    cols = df_preview.columns.tolist()
    if cols:
        col_width = 190 / len(cols)
        for col in cols:
            pdf.cell(col_width, 8, clean_text(str(col)[:15]), border=1) # Truncate long headers
        pdf.ln()
        
        # Table Rows
        for _, row in df_preview.head(10).iterrows():
            for col in cols:
                val = str(row[col])
                pdf.cell(col_width, 8, clean_text(val[:15]), border=1) # Truncate long values
            pdf.ln()

    return pdf.output(dest='S').encode('latin-1')

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
            model = genai.GenerativeModel('gemini-2.5-flash')
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
    
    Task: Provide a crisp, brief, professional analysis in 3 bullet points for each heading given below.
    1. Identify 3 Key Trends or Patterns.
    2. Highlight any Outliers or Anomalies.
    3. Suggest 1 Strategic Action based on this data.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# --- INITIALIZE SESSION STATE ---
if "user_question" not in st.session_state: st.session_state.user_question = ""
if "last_result" not in st.session_state: st.session_state.last_result = None
if "last_sql" not in st.session_state: st.session_state.last_sql = None
if "last_question" not in st.session_state: st.session_state.last_question = None
if "last_insights" not in st.session_state: st.session_state.last_insights = "" # Store insights for PDF
if "show_chart" not in st.session_state: st.session_state.show_chart = False

def set_q(q): st.session_state.user_question = q

# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown("## 🤖 **DATA** ASSISTANT") 
    st.caption("Pro Edition")
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
                                
                                # --- AUTO-GENERATE INSIGHTS FOR PDF ---
                                st.session_state.last_insights = generate_insights(result, question)
                                # --------------------------------------
                                
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.last_result = None

    # --- RESULTS COLUMN ---
    with col_results:
        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            sql = st.session_state.last_sql
            insights = st.session_state.last_insights
            
            # --- NEW: DOWNLOAD BUTTONS ROW ---
            d_col1, d_col2 = st.columns([1, 1])
            with d_col1:
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Result", csv, "data.csv", "text/csv", use_container_width=True)
            with d_col2:
                # Generate PDF on the fly
                pdf_bytes = create_pdf_report(st.session_state.last_question, sql, insights, result)
                st.download_button("📄 Download PDF Report", pdf_bytes, "analysis_report.pdf", "application/pdf", use_container_width=True)
            # ---------------------------------
            
            # TABS for organized viewing
            tab_table, tab_viz, tab_insight, tab_sql = st.tabs(["📄 Data Table", "📊 Visualization", "🧠 AI Insights", "📜 SQL"])
            
            with tab_table:
                st.dataframe(result, use_container_width=True)
            
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
                                fig = px.histogram(plot_df, x=x_axis, y=y_axis, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Viz Error: {e}")
                else:
                    st.info("Visualizations need at least 2 columns.")
            
            with tab_insight:
                # Insights are already generated, just display them
                st.markdown("### 🧠 AI Analysis")
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