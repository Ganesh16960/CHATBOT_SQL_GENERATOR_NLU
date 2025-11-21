import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "defog/sqlcoder-7b-2"  # change if you want another open-source model

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return tokenizer, model

tokenizer, model = load_model()

def build_prompt(schema: str, question: str) -> str:
    return f"""You are an expert SQL generator.

Database schema:
{schema}

Write a single SQL query that answers the question.
Return ONLY the SQL query, without explanation, markdown, or backticks.

Question: {question}
SQL:
"""

def generate_sql(schema: str, question: str, max_new_tokens: int = 256) -> str:
    prompt = build_prompt(schema, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = full.split("SQL:")[-1].strip()
    return sql

st.title("💬 SQL Generator Chatbot (Open-Source LLM)")

default_schema = """TABLE customers (
  customer_id INT PRIMARY KEY,
  name VARCHAR(100),
  city VARCHAR(100),
  signup_date DATE
);

TABLE orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  amount DECIMAL(10,2),
  order_date DATE,
  status VARCHAR(20)
);
"""

schema = st.text_area("Database schema", value=default_schema, height=220)
question = st.text_input("Ask a question about your data")

if st.button("Generate SQL"):
    if not schema.strip() or not question.strip():
        st.warning("Please enter both schema and question.")
    else:
        sql = generate_sql(schema, question)
        st.code(sql, language="sql")
