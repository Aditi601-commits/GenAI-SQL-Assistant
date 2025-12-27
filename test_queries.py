import sqlite3
import pandas as pd

def run_query(query_text):
    
    conn = sqlite3.connect('database.db')
    
    try:
        # Run the SQL and load it into a nice table (DataFrame)
        df = pd.read_sql_query(query_text, conn)
        print(f"\n--- Query: {query_text} ---")
        print(df)
        print("-" * 50)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# --- TEST QUESTIONS ---

# 1. Sanity Check: Show me 5 random sales
run_query("SELECT * FROM sales LIMIT 5;")

# 2. Total Revenue: How much money did our fake Flipkart make?
# (Write this number down! You will ask the AI this later.)
run_query("SELECT SUM(total_amount) as Total_Revenue FROM sales;")

# 3. Best Selling Category: What should we stock more of?
sql_3 = """
SELECT p.category, SUM(s.total_amount) as Category_Revenue
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY Category_Revenue DESC;
"""
run_query(sql_3)