import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import altair as alt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained GPT-2 model
chatbot_model = pipeline("text-generation", model="distilgpt2")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'query_log' not in st.session_state:
    st.session_state.query_log = []

if 'login' not in st.session_state:
    st.session_state.login = False

# ---------------------------
# Healthcare Rule-based Logic (Updated)
# ---------------------------
def rule_based_response(user_input):
    if "fever" in user_input:
        return "You might have an infection. Please monitor your temperature and consult a doctor."
    elif "cough" in user_input:
        return "For cough, stay hydrated and consult a physician if it persists."
    elif "sore throat" in user_input:
        return "A sore throat could be caused by a viral infection, strep throat, or allergies. Please consult a healthcare provider for a diagnosis."
    elif "symptom" in user_input:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "Take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        return None

# ---------------------------
# GPT-2 AI-based Response (Updated)
# ---------------------------
def gpt2_response(user_input):
    # Filter out low-quality generations
    response = chatbot_model(user_input, max_length=100, num_return_sequences=1)
    generated = response[0]['generated_text']
    
    # Refined health-related keyword filter
    health_keywords = ["doctor", "health", "symptom", "treatment", "medicine", "cough", "fever", "sore throat", "pain", "disease", "illness"]
    
    # Enhanced check: If a relevant health keyword is found, use the generated response
    if any(word in generated.lower() for word in health_keywords):
        return generated
    else:
        return "I'm here to help with your healthcare queries. Could you please elaborate?"

# ---------------------------
# Modular Hybrid Chatbot Logic (Updated)
# ---------------------------
def healthcare_chatbot(user_input, method="hybrid"):
    rule_response = rule_based_response(user_input.lower())
    ai_response = gpt2_response(user_input)
    
    if method == "rule":
        return rule_response or "Sorry, I don't have information on that."
    elif method == "ai":
        return ai_response
    else:  # Hybrid logic
        return rule_response if rule_response else ai_response



# ---------------------------
# Login Page
# ---------------------------
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "user" and password == "pass":  # Simple login
            st.session_state.login = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials.")

# ---------------------------
# Analytics Dashboard
# ---------------------------
def show_dashboard():
    st.subheader("📊 Query Analytics Dashboard")
    df = pd.DataFrame(st.session_state.query_log, columns=["Query", "Method", "Rating"])
    
    if not df.empty:
        st.dataframe(df)

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Method'),
            y=alt.Y('count()', title='Number of Queries'),
            color='Method'
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No queries yet to display.")

# Main Chat Interface
def main_app():
    st.title("💬 AI-Powered Healthcare Assistant")
    
    st.sidebar.title("Settings")
    method = st.sidebar.radio("Choose Response Mode:", ("hybrid", "rule", "ai"))
    
    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            response = healthcare_chatbot(user_input, method)

            # Save to chat history
            st.session_state.chat_history.append((user_input, response))
            st.session_state.query_log.append((user_input, method, None))  # Do not add rating yet

            st.write(f"**User:** {user_input}")
            st.write(f"**Healthcare Assistant:** {response}")

            # Show feedback slider
            rating = st.slider("Rate this response (1-5 stars):", 1, 5, 3)

            # Update query log with rating only when user interacts with slider
            if rating:
                st.session_state.query_log[-1] = (user_input, method, rating)

        else:
            st.warning("Please enter a query.")

    # Show chat history
    if st.checkbox("Show Chat History"):
        for i, (q, r) in enumerate(st.session_state.chat_history):
            st.write(f"**{i+1}. You:** {q}")
            st.write(f"**Assistant:** {r}")

    # Show dashboard
    if st.checkbox("Show Query Analytics"):
        show_dashboard()


# ---------------------------
# Run App
# ---------------------------

def main():
    if not st.session_state.login:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
