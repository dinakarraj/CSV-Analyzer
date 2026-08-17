from google import genai 
from google.genai import types
from dotenv import load_dotenv
import os 
import requests 
import pandas as pd, numpy as np
from pandas.api.types import is_numeric_dtype
import re
import json 

AGGREGRATION_HANDLERS = {
     "sum" : lambda g, col: g[col].sum(),
     "average" : lambda g, col: g[col].mean(),
     "median" : lambda g, col:g[col].median(),
     "highest" : lambda g, col:g[col].max(),
     "lowest" : lambda g, col:g[col].min()
}

NO_COLUMN_AGGREGATIONS= {"count"}

SUPPORTED_AGGREGATIONS = set(AGGREGRATION_HANDLERS) | NO_COLUMN_AGGREGATIONS

def equals(series, value):
     if isinstance(value, str) and not is_numeric_dtype(series):
          return series.str.lower() == value.lower()
     return series == value

def not_equals(series, value):
     if isinstance(value, str) and not is_numeric_dtype(series):
               return series.str.lower() != value.lower()
     return series != value

OPERATION_HANDLERS ={
     "=" : equals,
     "!=" : not_equals,
     ">" : lambda series, value : series > value ,
     "<" : lambda series, value : series < value ,
     ">=" : lambda series, value : series >= value , 
     "<=" : lambda series, value : series <= value 
}

#this function is primarily used to clean the json markdown 
def clean_text(text):
     text = text.strip()
     text = re.sub(r'^```(?:json)?\s*', '', text)
     text = re.sub(r'\s*```$', '', text)
     return text

def apply_aggregation(data, aggregation_type, group_by_fields, parsed):

     if not aggregation_type:
          return data

     if aggregation_type not in SUPPORTED_AGGREGATIONS:
          raise ValueError(f"{aggregation_type} type is not supported yet")

     source = data.groupby(group_by_fields) if group_by_fields else data 


     if aggregation_type in NO_COLUMN_AGGREGATIONS:
          #aggregation count 
          if group_by_fields:
               return source.size().reset_index(name = "count")
          else:
               return pd.DataFrame({"count" : [len(data)]})

    #  if control came here it means aggregation is in no column and it should be in aggregation_handlers
     handler = AGGREGRATION_HANDLERS[aggregation_type]
     target_column_list = parsed["target_column"]
     results =  {}

     if target_column_list:
          for target_column in target_column_list:
               value = handler(source, target_column)
               if not group_by_fields:
                    value = pd.Series([value], name=target_column)
               results[target_column]=value 

     result = pd.DataFrame(results)
     return result

def parse_with_gemini(client, dataset, question):
#parameters client object, list of columns, user_question

    prompt = f"""You are an expert data query parser.

                Your ONLY job is to convert the user's question into structured JSON.

                DO NOT answer the question.

                DO NOT calculate anything.

                DO NOT explain anything.

                Use ONLY the list of dataset columns provided below.

                If a column is not present in the dataset, return null for that field.

                Return ONLY valid JSON.

        Schema:

        {{
            "intent": "",
            "aggregation": "",
            "target_column": [],
            "filters": [
                {{
                    "column": "",
                    "value": "", 
                    "operator": ""
                }}
            ],
            "group_by": [],
            "sort": 
            [{{
                "column": "",
                "order": ""
            }}],
            "limit": null
        }}

        Definitions:

        intent
        Possible values include:
        - aggregate
        - select
        - filter
        - sort
        - unknown

        aggregation
        Possible values include:
        - sum
        - average
        - count
        - median
        - highest
        - lowest
        - max
        - min
        - null

        target_column

        Return a list of dataset columns on which the requested operation should be applied.

        Examples

        Question:
        Average salary

        Output:
        "target_column": ["salary"]

        Question:
        Sum of salary and bonus

        Output:
        "target_column": ["salary", "bonus"]

        Question:
        Show employee_name and salary

        Output:
        "target_column": ["employee_name", "salary"]

        Question:
        Count employees

        Output:
        "target_column": []

        group_by
        Return the column name if the user requests grouping.
        Otherwise return null.

        Return every sorting instruction found in the user's question.

        Each sort item must contain:

        - "column"
        - "order"

        order can only be:
        - "asc" or "ascending" or "increasing order" or "increasing"
        - "desc" or "decreasing" or "decreasing order" or "decreasing"

        Examples

        Question:
        Sort by salary descending

        Output:

        "sort": [
            {{
                "column": "salary",
                "order": "desc"
            }}
        ]

            Question:
            Sort by department ascending then salary descending

            Output:

            "sort": [
                {{
                    "column": "department",
                    "order": "asc"
                }},
                {{
                    "column": "salary",
                    "order": "desc"
                }}
            ]

            If the user does not request sorting, return an empty list.

        filters
        Return every filter found in the user's question.
        Each filter has "column" and "value".
        If the value is a whole number, return it as a JSON number, not a string. Example: 20, not "20".
        If the value is a decimal number, return it as a JSON number. Example: 19.99, not "19.99".
        If the value is text, return it as a JSON string. Example: "Widget".
        Match the value's type to how it would naturally appear in that column — numeric columns get numeric values, text columns get string values.


        operator

        Return the comparison operator implied by the question.
        Possible values: "=", "!=", ">", "<", ">=", "<="

        Examples:
        - "equal to", "is", "equals"        -> "="
        - "not equal to", "different from"  -> "!="
        - "greater than", "more than", "above" -> ">"
        - "less than", "under", "below"     -> "<"
        - "at least", "minimum"             -> ">="
        - "at most", "maximum", "up to"     -> "<="

        If no operator is implied, default to "=".

        limit

        Return the maximum number of rows requested.

        Examples

        top 5
        first 5
        show 10
        return 20 rows

        → 5
        → 5
        → 10
        → 20

        If no limit is mentioned, return null.

        

        1. COUNT counts rows/records/entities.
        COUNT does NOT require a target_column.

        2. If the user asks:
        - how many employees
        - number of employees
        - employee count
        - count employees
        - number of records
        - how many records

        return:

        "aggregation": "count"
        "target_column": []

        3. For COUNT, target_column MUST always be an empty list
        unless the user explicitly asks to count a specific column.

        4. SUM, AVERAGE, MEDIAN, HIGHEST and LOWEST require a target column.

        Examples:

        Question: How many employees are there?

        Output:
        {{
            "intent": "aggregate",
            "aggregation": "count",
            "target_column": [],
            "filters": [],
            "group_by": [],
            "sort": [],
            "limit": null
        }}

        Question: How many employees are in each department?

        Output:
        {{
            "intent": "aggregate",
            "aggregation": "count",
            "target_column": [],
            "filters": [],
            "group_by": ["Department"],
            "sort": [],
            "limit": null
        }}

        Dataset Columns:
        {dataset}

        User Question:
        {question}

        Rules:

        Return ONLY JSON.

        Never wrap JSON inside markdown.

        Do not use ```json.

        Do not explain your reasoning.

        If a value is missing, return null.

        If no filters exist, return an empty list. """

    response = client.models.generate_content(model="gemini-2.5-flash", 
                                             contents=prompt,
                                             config= types.GenerateContentConfig(response_mime_type="application/json")
                                             )
    return response.text

def run_query(data, parsed):
     
    

    filters=parsed["filters"]
    if filters:
          
          for filter_rule in filters:
               column=filter_rule["column"]
               value=filter_rule["value"]
               operator = filter_rule["operator"]


               if operator not in OPERATION_HANDLERS:
                    raise ValueError(f"This {operator} is not supported!!!")
                    return

               column_data= data[column]
               handler = OPERATION_HANDLERS[operator]
               mask = handler(column_data, value)
               data=data[mask] 
    
    group_by_fields = parsed["group_by"]

    aggregation_type = parsed["aggregation"]

    result = apply_aggregation(data, aggregation_type, group_by_fields, parsed)
  
    sort_list = parsed["sort"]

    if sort_list:
         columns = []
         orders  = []

         for rule in sort_list:

            columns.append(rule["column"])
            orders.append(rule["order"].lower() == "asc")

         result=result.sort_values(by=columns, ascending=orders) 

        

    limit_rows = parsed["limit"]

    if limit_rows:
         result = result.head(limit_rows)

    return result
         