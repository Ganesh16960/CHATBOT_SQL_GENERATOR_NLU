import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Lightweight text-to-SQL model (T5-small based, good for Streamlit Cloud)
MODEL_NAME = "cssupport/t5-small-awesome-text-to-sql"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer from base t5-small as per model card
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Text-to-SQL model (small, CPU-friendly)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


def build_prompt(schema: str, question: str) -> str:
    """
    Model expects:
    tables:
    CREATE TABLE ...;
    CREATE TABLE ...;
    query for: <question>
    """
    schema_clean = schema.strip()
    question_clean = question.strip()

    prompt = (
        "tables:\n"
        + schema_clean
        + "\n"
        + "query for:"
        + question_clean
    )
    return prompt


def generate_sql(schema: str, question: str, max_length: int = 256) -> str:
    prompt = build_prompt(schema, question)

    inputs = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
        )

    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_sql.strip()


# ----------------------- Streamlit UI -----------------------

st.title("💬 SQL Generator Chatbot (Lightweight Open-Source LLM)")

st.markdown(
    """
Paste your **CREATE TABLE** schema below and ask a natural language question.
The model will generate a SQL query.
"""
)

default_schema = """CREATE TABLE customers (
  customer_id INT PRIMARY KEY,
  name VARCHAR(100),
  city VARCHAR(100),
  signup_date DATE
);

CREATE TABLE orders (
  order_id INT PRIMARY KEY,
  customer_id INT,
  amount DECIMAL(10,2),
  order_date DATE,
  status VARCHAR(20)
);"""

schema = st.text_area("Database schema (as CREATE TABLE statements)", value=default_schema, height=220)
question = st.text_input("Ask a question about your data")

if st.button("Generate SQL"):
    if not schema.strip():
        st.warning("Please provide a database schema.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            sql = generate_sql(schema, question)
        st.subheader("Generated SQL")
        st.code(sql, language="sql")
