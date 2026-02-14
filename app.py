import streamlit as st
import difflib
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Q&A from JSON file
def load_dataset():
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
            return data["questions"]
    except Exception as e:
        st.error("Error loading data.json: " + str(e))
        return []

DATASET = load_dataset()
# Helper: Find best match
def find_best_answer(user_query):
    questions = [item["question"] for item in DATASET]
    match = difflib.get_close_matches(user_query.lower(), questions, n=1, cutoff=0.5)

    if match:
        for item in DATASET:
            if item["question"].lower() == match[0].lower():
                return item["answer"]

    return None

# Fallback LLM response
def llm_fallback(user_query):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                ChatCompletionMessageParam(role="system", content="You are a helpful AI support agent."),
                ChatCompletionMessageParam(role="user", content=user_query)
            ]
        )
        return response.choices[0].message.content
    except Exception:
        return "Sorry, I couldn't process that request."

# Streamlit UI
st.set_page_config(page_title="Thoughtful AI Support Agent", page_icon="🤖")

st.title("🤖 Thoughtful AI – Customer Support Agent")
st.write("Ask me anything about Thoughtful AI's automation agents!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    answer = find_best_answer(user_input)

    if answer:
        final_answer = answer
    else:
        final_answer = llm_fallback(user_input)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    with st.chat_message("assistant"):
        st.write(final_answer)
