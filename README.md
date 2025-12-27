# AI-Powered SQL Sales Assistant

A Natural Language to SQL generation tool that allows users to query a Sales Database using plain English. Powered by **Google Gemini 2.0 Flash** and **Streamlit**.

## 🚀 Features
- **Natural Language Querying:** Ask "What is the total revenue from Electronics?" and get instant answers.
- **Automated SQL Generation:** The AI converts your question into a valid SQL query specific to the schema.
- **Safety Guardrails:** Includes logic to reject non-data questions (e.g., "Write a poem").
- **Dynamic Visualization:** Automatically generates bar charts for comparative data.
- **Transparent Logic:** View the generated SQL query in an expandable dropdown.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **AI Model:** Gemini flash-lite latest
- **Database:** SQLite (Pre-loaded with 1,000+ sales records)
- **Language:** Python

## 📂 Project Structure
- `app.py`: The main application logic.
- `setup_database.py`: A script that generates dummy Sales, Product, and Customer data.
- `database.db`: The SQLite database file.
- `requirements.txt`: List of dependencies.