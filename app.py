import streamlit as st 
import pandas as pd 
import json
import os
import re

from google import genai 
from google.genai import types 
from dotenv import load_dotenv
from Analyser_core import parse_with_gemini, clean_text, run_query

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key= api_key)


st.title("AI CSV Analyser📊")
st.write("Upload a csv file and ask questions about your data")

file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    with st.expander("Dataset Preview"):
         st.dataframe(data)

    question = st.text_input("Ask a question about your data")

   

    if st.button("Analyse"):

        if not question:
                st.warning("Try enetering a question")
        else:  
                try:
                    with st.spinner("Thinking....."):
                        raw_response = parse_with_gemini(client, data.columns, question)
                        cleaned = clean_text(raw_response)
                        parsed = json.loads(cleaned)
                        result = run_query(data, parsed)
                        with st.expander("Click to see how gemini understood your response"):
                            st.json(parsed)
                        st.subheader("Result of your query is shown below")
                        st.dataframe(result)
                except json.JSONDecodeError:
                    st.error("Gemini's response wasn't valid JSON. Try rephrasing your question.")
                except KeyError as e:
                    st.error(f"column {e} does not exist in your data. Available columns : {list(data.columns)} ")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
        

else:
    st.write("Upload a csv to get started")