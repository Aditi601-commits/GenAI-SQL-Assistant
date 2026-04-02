import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import time
from fpdf import FPDF

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

# --- STATE MANAGEMENT (For Quick Buttons) ---
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

def set_question(question_text):
    st.session_state.user_question = question_text

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
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    response = model.generate_content(prompt)
    return response.text

def get_gemini_response(question):
    prompt = """
    You are an expert SQL Assistant for a Sales Database.
    Tables: products, customers, sales.
    
    RULES:
    1. Return ONLY valid SQL.
    2. If asking for Revenue, use SUM(total_amount).
    3. Use LOWER(col) LIKE '%val%' for text.
    4. Return "NO_SQL" if off-topic.
    5. No markdown, just code.
    """
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    response = model.generate_content([prompt, question])
    return response.text.strip().replace("```sql", "").replace("```", "")

def execute_query(sql_query):
    if not os.path.exists('database.db'):
        return "⚠️ Database not found. Run setup_database.py."
    conn = sqlite3.connect('database.db')
    try:
        if "DROP" in sql_query.upper() or "DELETE" in sql_query.upper():
            return "SAFETY ALERT: Read-Only Mode."
        return pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

# --- MAIN UI ---

# 1. QUICK ACTION BUTTONS
st.write("### ⚡ Quick Actions")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💰 Total Revenue", use_container_width=True):
        set_question("What is the total revenue generated?")
with col2:
    if st.button("🏆 Top Products", use_container_width=True):
        set_question("Show me the top 5 most expensive products")
with col3:
    if st.button("📉 Sales by Category", use_container_width=True):
        set_question("Count how many sales happened for each category")

# 2. INPUT SECTION
# We bind the value to session_state so the buttons can update it
question = st.text_input(
    "Or type your specific question here:",
    key="user_question"
)

# 3. ANALYSIS LOGIC
if st.button("Run Analysis", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            sql = get_gemini_response(question)
            
            if sql == "NO_SQL":
                st.error("I can only answer questions about Sales data.")
            else:
                result = execute_query(sql)
                
                if isinstance(result, pd.DataFrame):
                    if result.empty:
                        st.warning("No data found for this query.")
                    else:
                        # A. Show Data
                        st.success("Analysis Complete")
                        st.dataframe(result)
                        
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
            st.info("👈 Use the chat to query data.")
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