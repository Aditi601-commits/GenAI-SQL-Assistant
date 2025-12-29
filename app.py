import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import plotly.express as px
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

# --- HELPER: LOAD CUSTOM DATA ---
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
        return conn, df.columns.tolist()

    except Exception as e:
        return None, str(e)

# --- HELPER: SQL GENERATION ---
def get_gemini_response(question, schema_info):
    cols = ', '.join(schema_info[:50]) 
    context = f"Table: uploaded_data. Columns: {cols}"

    prompt = f"""
    Context: {context}
    User Question: "{question}"
    
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
    
    Keep it concise (bullet points).
    """
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# --- SIDEBAR ---
st.sidebar.header("📂 Data Source")
st.sidebar.markdown(
    """<div style="margin-bottom: 10px;">
        <p style="font-weight: bold; margin-bottom: 5px;">Upload Data File</p>
        <p style="font-size: 0.8em; color: #888;">Accepted: CSV, Excel, JSON</p>
    </div>""", unsafe_allow_html=True
)
uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")

conn = None
schema_info = []

if uploaded_file:
    st.info(f"Using: **{uploaded_file.name}**")
    conn, schema_or_error = load_custom_data(uploaded_file)
    if conn is None:
        st.error(f"Error: {schema_or_error}")
        st.stop()
    else:
        schema_info = schema_or_error
else:
    st.info("👋 Welcome! Please upload a CSV or Excel file to begin.")
    st.stop()

# --- QUICK ACTIONS ---
st.write("### ⚡ Quick Actions")
col1, col2, col3 = st.columns(3)

if "user_question" not in st.session_state:
    st.session_state.user_question = ""

def set_q(q): st.session_state.user_question = q

if len(schema_info) > 0:
    try:
        cat_col = next((col for col in schema_info if "ID" not in col.upper() and "NUM" not in col.upper()), schema_info[0])
        with col1:
            if st.button(f"📊 Count by {cat_col}", use_container_width=True): set_q(f"Count records by {cat_col}")
        with col2:
            if st.button("👀 Sample Data", use_container_width=True): set_q("Show 5 random rows")
        with col3:
            if st.button("📑 Data Summary", use_container_width=True): set_q("Show count of all rows")
    except: pass

# --- ANALYSIS SECTION ---
question = st.text_input("Ask a question about your data:", value=st.session_state.user_question)

# Initialize Session State
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

if st.button("Run Analysis", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state.user_question = question 
        st.session_state.show_chart = False # Reset chart on new query
        
        with st.spinner("Analyzing..."):
            try:
                # 1. SQL Generation
                sql = get_gemini_response(question, schema_info)
                
                if sql.startswith("ERROR:"):
                    st.error(sql)
                    st.session_state.last_result = None
                else:
                    # 2. Execution
                    result = pd.read_sql_query(sql, conn)
                    
                    if result.empty:
                        st.warning("No data found.")
                        st.session_state.last_result = None
                    else:
                        st.session_state.last_result = result
                        st.session_state.last_sql = sql
                        st.session_state.last_question = question
                        st.success("Analysis Complete!")
                            
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.last_result = None

# --- RESULTS DISPLAY ---
if st.session_state.last_result is not None:
    result = st.session_state.last_result
    sql = st.session_state.last_sql
    
    st.dataframe(result, use_container_width=True)
    
    st.markdown("---")
    if st.button("✨ Explain this Result (AI Insights)"):
        with st.spinner("Generating Business Intelligence..."):
            insights = generate_insights(result, st.session_state.last_question)
            st.markdown("### 🧠 AI Analysis")
            st.markdown(insights)
            st.info("⚠️ Analysis based on the top 20 rows of the result.")

    col_actions1, col_actions2 = st.columns(2)
    
    with col_actions1:
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
    
    # --- INTERACTIVE VISUALIZATION SECTION ---
    with st.expander("📊 Visualize Data", expanded=True):
        if len(result.columns) >= 2:
            st.caption("Select options and click 'Generate Chart'")
            
            # Prepare Data
            plot_df = result.copy()
            numeric_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = plot_df.select_dtypes(exclude=['number']).columns.tolist()
            
            # Chart Controls
            c1, c2, c3 = st.columns(3)
            with c1:
                chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram"])
            with c2:
                default_x = categorical_cols[0] if categorical_cols else plot_df.columns[0]
                x_axis = st.selectbox("X-Axis", plot_df.columns, index=plot_df.columns.get_loc(default_x))
            with c3:
                default_y = numeric_cols[0] if numeric_cols else plot_df.columns[1]
                if chart_type in ["Pie", "Histogram"]:
                    y_axis = st.selectbox("Values / Y-Axis", [None] + list(plot_df.columns))
                else:
                    y_axis = st.selectbox("Y-Axis", plot_df.columns, index=plot_df.columns.get_loc(default_y))

            # The "Show" Button
            if st.button("Generate Chart"):
                st.session_state.show_chart = True

            # Only render if button was clicked
            if st.session_state.show_chart:
                try:
                    if chart_type == "Bar":
                        fig = px.bar(plot_df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", template="plotly_dark")
                    elif chart_type == "Line":
                        fig = px.line(plot_df, x=x_axis, y=y_axis, title=f"{y_axis} Trend", template="plotly_dark")
                    elif chart_type == "Scatter":
                        fig = px.scatter(plot_df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", template="plotly_dark")
                    elif chart_type == "Pie":
                        fig = px.pie(plot_df, names=x_axis, values=y_axis, title=f"Distribution of {x_axis}", template="plotly_dark")
                    elif chart_type == "Histogram":
                        fig = px.histogram(plot_df, x=x_axis, title=f"Distribution of {x_axis}", template="plotly_dark")
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate chart: {e}")

        else:
            st.info("⚠️ Charts require at least 2 columns (e.g., Category & Value).")

    with st.expander("📜 View SQL Query"):
        st.code(sql, language="sql")

if conn: conn.close()