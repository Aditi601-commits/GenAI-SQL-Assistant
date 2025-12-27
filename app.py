import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sales Database Bot", page_icon="🛒", layout="centered")

# --- TITLE ---
st.title("Sales Intelligence Bot")
st.markdown("Ask questions about the **Sales Database** (Revenue, Customers, Products).")

# LOAD API KEY FROM SECRETS 
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ API Key missing! Please add it to .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_response(question):
    """
    Prompt tailored strictly for the 'database.db' schema.
    """
    prompt = [
        """
        You are an expert SQL Assistant for a Sales Database.
        
        The user has a SQLite database named 'database.db' with these tables:
        
        1. products (product_id, name, category, price, stock_quantity)
        2. customers (customer_id, name, city, email, signup_date)
        3. sales (sale_id, customer_id, product_id, quantity, total_amount, sale_date)
        
        STRICT RULES:
        1. Return ONLY valid SQL.
        2. If the user asks for "Revenue", calculate SUM(total_amount).
        3. Use `LOWER(name) LIKE '%val%'` for case-insensitive text matching.
        4. If the question is not about sales data, return "NO_SQL".
        5. Return ONLY the SQL code. No markdown (no ```sql).
        """
    ]
    
    model = genai.GenerativeModel('gemini-flash-lite-latest')
    response = model.generate_content([prompt[0], question])
    return response.text.strip().replace("```sql", "").replace("```", "")

def execute_query(sql_query):
    # HARDCODED DATABASE CONNECTION
    db_file = 'database.db'
    
    if not os.path.exists(db_file):
        return f"Error: The file '{db_file}' does not exist. Please run setup_database.py first."

    conn = sqlite3.connect(db_file)
    try:
        if "DROP" in sql_query.upper() or "DELETE" in sql_query.upper():
            return "SAFETY ALERT: Read-Only Mode."
        
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()

# --- UI ---
question = st.text_input("Ask a question:", placeholder="e.g., What is the total revenue from Electronics?")

if st.button("Analyze"):
    if not GOOGLE_API_KEY:
        st.error("API Key Missing.")
    else:
        with st.spinner("Thinking..."):
            sql = get_gemini_response(question)
            
            if sql == "NO_SQL":
                st.error("I can only answer questions about the Sales Database.")
            else:
                with st.expander("See SQL Query"):
                    st.code(sql, language="sql")
                
                result = execute_query(sql)
                
                if isinstance(result, pd.DataFrame):
                    st.success("Results:")
                    st.dataframe(result)
                    
                    # Auto-Chart logic
                    if len(result.columns) == 2:
                        st.bar_chart(result.set_index(result.columns[0]))
                else:
                    st.error(result)