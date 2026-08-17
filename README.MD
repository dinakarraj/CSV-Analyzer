📊 AI CSV Analyser

Ask your CSV questions in plain English — powered by Google's Gemini API and pandas. Upload a CSV, type a question like "average salary by department", and get your answer back as a table. No SQL, no pandas syntax required.

How it works
You upload a CSV and ask a question in plain English.
Gemini parses your question into structured JSON (intent, filters, aggregation, sort, limit).
That JSON drives real pandas operations on your data.
The result is displayed as a table in a Streamlit UI.

Features
Natural language to structured query parsing via Gemini
Supports filtering, aggregation (sum, average, median, count, highest, lowest), grouping, sorting, and row limits
Case-insensitive text matching for filters
Clean error handling — mismatched columns, unsupported operators, and malformed AI responses all show as readable messages instead of crashes
Simple, responsive Streamlit interface
Tech stack
Python
Streamlit — UI
pandas — data operations
Google Gemini API — natural language parsing
Setup
Clone the repo:
bash
git clone https://github.com/dinakarraj/CSV-Analyzer.git
cd CSV-Analyzer
Install dependencies:
bash
pip install -r requirements.txt
Add your Gemini API key. Create a .env file in the project root:
GEMINI_API_KEY="YOUR_API_KEY_HERE"
Run the app:
bash
streamlit run app.py
Example questions
"How many employees are there?"
"Average salary by department"
"Show employees in Hyderabad with salary above 50000"
"Sort by experience descending, top 5"

Project structure
├── app.py              # Streamlit UI
├── Analyser_core.py    # Core logic: Gemini parsing + query execution
├── requirements.txt    # Python dependencies
├── data.csv            # Sample dataset
└── .gitignore
