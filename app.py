import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import os

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
                    st.error(result)