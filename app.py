import streamlit as st
from main import get_answer


st.title("RAG Chatbot")
question = st.text_input("Enter your question here:")
submit = st.button("Submit",type="primary")

if question and submit:
    with st.spinner("Processing..."):
        answer = get_answer(question)
        st.subheader("Answer")
        st.write(answer)