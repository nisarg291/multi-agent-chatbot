import streamlit as st
import requests
from datetime import datetime
import numpy as np
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Concordia Chatbot", page_icon=":robot_face:", layout="wide")

# Apply custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 60vh;
            overflow-y: auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #dcf8c6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 70%;
            align-self: flex-end;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .bot-message {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 70%;
            align-self: flex-start;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .follow-up {
            font-style: italic;
            color: #555;
            margin-top: 5px;
        }
        .agent-label {
            font-weight: bold;
            color: #2c3e50;
        }
        .stTextArea > div > div > textarea {
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 10px;
            background-color: #ffffff;
            height: 100px;
            width: 100%;
            resize: vertical;
            font-family: 'Arial', sans-serif;
        }
        .error-message {
            color: #e74c3c;
            font-style: italic;
        }
        .stButton > button {
            border-radius: 5px;
            background-color: #3498db;
            color: white;
            font-weight: bold;
            padding: 8px 16px;
        }
        .stButton > button:hover {
            background-color: #2980b9;
        }
        .sidebar .sidebar-content {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metrics-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Backend URL configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if "localhost" in BACKEND_URL or "0.0.0.0" in BACKEND_URL:
    logger.warning("Using local BACKEND_URL. Set 'BACKEND_URL' environment variable for production.")

# Initialize session state
def initialize_session_state():
    defaults = {
        "history": [],
        "feedback_submitted": set(),
        "current_topic": None,
        "feedback_ratings": {},
        "follow_up_input": "",
        "eval_metrics": None,
        "sentence_model": None  # Lazy-loaded model
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Lazy load SentenceTransformer
def load_sentence_model():
    if st.session_state.sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            st.session_state.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {str(e)}. Using string matching for topic similarity.")
            st.session_state.sentence_model = False  # Indicate failure
    return st.session_state.sentence_model

def is_same_topic(prev_topic, current_topic):
    if not prev_topic or not current_topic:
        return False
    model = load_sentence_model()
    if model is False:  # Failed to load
        return prev_topic.lower() == current_topic.lower()
    elif model:
        try:
            prev_embedding = model.encode([prev_topic])[0]
            curr_embedding = model.encode([current_topic])[0]
            similarity = cosine_similarity([prev_embedding], [curr_embedding])[0][0]
            return similarity >= 0.8
        except Exception as e:
            logger.error(f"Error checking topic similarity: {str(e)}. Falling back to string match.")
            return prev_topic.lower() == current_topic.lower()
    return prev_topic.lower() == current_topic.lower()

def fetch_evaluation_metrics():
    try:
        response = requests.get(f"{BACKEND_URL}/evaluation_metrics/", timeout=120)
        response.raise_for_status()
        eval_metrics = response.json()
        st.session_state.eval_metrics = eval_metrics
        logger.info(f"Evaluation metrics fetched: {eval_metrics}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch evaluation metrics: {str(e)}")
        st.session_state.eval_metrics = {
            "total_conversations": 0,
            "active_users": 0,
            "avg_session_length": 0.0,
            "avg_response_time": 0.0,
            "resolution_rate": 0.0,
            "fallback_rate": 0.0,
            "accuracy": 0.0,
            "coherence": 0.0,
            "satisfaction": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

def send_message(message, user_id="user"):
    try:
        with st.spinner("Processing your query..."):
            response = requests.post(
                f"{BACKEND_URL}/chat/",
                json={"user_id": user_id, "query": message},
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            required_fields = ["agent", "response", "metrics", "benchmark_comparison", "doc_id", "topic", "follow_up_question"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")
            logger.info(f"Backend response: {data}")
            return data
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Please try again."
        logger.error(error_msg)
        st.error(error_msg)
        return {
            "agent": "Error",
            "response": error_msg,
            "metrics": {"accuracy": 0.0, "coherence": 0.0, "satisfaction": 0.0},
            "benchmark_comparison": {"accuracy": "Fail", "coherence": "Fail", "satisfaction": "Fail"},
            "doc_id": None,
            "topic": None,
            "follow_up_question": "Can you try again?"
        }
    except (requests.exceptions.RequestException, ValueError) as e:
        error_msg = f"Failed to communicate with backend: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return {
            "agent": "Error",
            "response": error_msg,
            "metrics": {"accuracy": 0.0, "coherence": 0.0, "satisfaction": 0.0},
            "benchmark_comparison": {"accuracy": "Fail", "coherence": "Fail", "satisfaction": "Fail"},
            "doc_id": None,
            "topic": None,
            "follow_up_question": "Can you tell me more?"
        }

# Display chatbot title
st.title("🤖 Concordia Chatbot")

# Fetch metrics on initial load
if st.session_state.eval_metrics is None:
    fetch_evaluation_metrics()

# Chat container
chat_html = '<div class="chat-container" id="chat-container">'
for chat in st.session_state.history:
    timestamp = chat.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if chat["role"] == "user":
        chat_html += f'<div class="user-message">**You ({timestamp}):** {chat["message"]}</div>'
    else:
        if chat["agent"] == "Error":
            chat_html += f'<div class="bot-message error-message">**Error ({timestamp}):** {chat["message"]}</div>'
        else:
            message_parts = chat["message"].split("\n\nWould you like to know more? How about: '")
            main_message = message_parts[0]
            follow_up = message_parts[1][:-1] if len(message_parts) > 1 else chat.get("follow_up_question", "")
            chat_html += f'<div class="bot-message"><span class="agent-label">{chat["agent"]} ({timestamp}):</span> {main_message}'
            if follow_up:
                chat_html += f'<div class="follow-up">Would you like to know more? How about: \'{follow_up}\'</div>'
            chat_html += '</div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Auto-scroll
st.components.v1.html(
    """
    <script>
        var chatContainer = document.getElementById('chat-container');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    </script>
    """,
    height=0
)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your message:",
        value=st.session_state.follow_up_input,
        key="user_input",
        placeholder="Type your multi-line query here (e.g., 'What is AI?' or 'How do I apply to Concordia?')",
        help="Ask about AI, Concordia admissions, or anything else!"
    )
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.form_submit_button("Send")
    with col2:
        clear_button = st.form_submit_button("Clear Chat")
    with col3:
        if st.session_state.history and st.session_state.history[-1]["role"] == "bot" and st.session_state.history[-1].get("follow_up_question"):
            if st.form_submit_button("Use Follow-Up"):
                st.session_state.follow_up_input = st.session_state.history[-1]["follow_up_question"]
                logger.info(f"Follow-up selected: {st.session_state.follow_up_input}")
                st.rerun()

# Handle message submission
if submit_button and user_input.strip():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({"role": "user", "message": user_input.strip(), "timestamp": timestamp})
    response_data = send_message(user_input.strip())
    
    new_topic = response_data.get("topic")
    if st.session_state.current_topic and new_topic and not is_same_topic(st.session_state.current_topic, new_topic):
        st.session_state.feedback_ratings[new_topic] = []
    elif new_topic and new_topic not in st.session_state.feedback_ratings:
        st.session_state.feedback_ratings[new_topic] = []

    response_text = response_data.get("response", "No response received.")
    if not isinstance(response_text, str):
        logger.error(f"Invalid response type: {type(response_text)} - {response_text}")
        response_text = "Error: Invalid response format from backend."

    st.session_state.history.append({
        "role": "bot",
        "agent": response_data.get("agent", "Unknown"),
        "message": response_text,
        "metrics": response_data.get("metrics", {"accuracy": 0.0, "coherence": 0.0, "satisfaction": 0.0}),
        "benchmark_comparison": response_data.get("benchmark_comparison", {"accuracy": "Fail", "coherence": "Fail", "satisfaction": "Fail"}),
        "doc_id": response_data.get("doc_id"),
        "timestamp": timestamp,
        "topic": new_topic,
        "follow_up_question": response_data.get("follow_up_question", "Can you tell me more?")
    })
    st.session_state.current_topic = new_topic
    if response_data.get("doc_id"):
        st.session_state.feedback_submitted.discard(response_data["doc_id"])
    st.session_state.follow_up_input = ""
    fetch_evaluation_metrics()
    st.rerun()

# Handle chat clearing
if clear_button:
    st.session_state.history = []
    st.session_state.feedback_submitted = set()
    st.session_state.current_topic = None
    st.session_state.feedback_ratings = {}
    st.session_state.follow_up_input = ""
    fetch_evaluation_metrics()
    st.rerun()

# Feedback for last response
if st.session_state.history and st.session_state.history[-1]["role"] == "bot":
    last_response = st.session_state.history[-1]
    if last_response["agent"] != "Error" and last_response["doc_id"]:
        feedback_key = f"feedback_{last_response['doc_id']}"
        feedback = st.slider(
            "Rate the last response (1 = Poor, 5 = Excellent):",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            format="%d",
            key=feedback_key
        )
        if st.button("Submit Feedback"):
            if last_response["doc_id"] not in st.session_state.feedback_submitted:
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/feedback/",
                        json={"doc_id": last_response["doc_id"], "rating": feedback},
                        timeout=120
                    )
                    response.raise_for_status()
                    data = response.json()
                    if data.get("message") != "Feedback submitted successfully":
                        raise ValueError(f"Unexpected response: {data}")
                    st.session_state.feedback_submitted.add(last_response["doc_id"])
                    last_response["rating"] = feedback
                    current_topic = st.session_state.current_topic
                    if current_topic:
                        st.session_state.feedback_ratings.setdefault(current_topic, []).append(feedback)
                    st.success("Feedback submitted successfully!")
                    logger.info(f"Feedback submitted for doc_id: {last_response['doc_id']}, rating: {feedback}")
                    fetch_evaluation_metrics()
                    st.rerun()
                except (requests.exceptions.RequestException, ValueError) as e:
                    error_msg = f"Feedback submission failed: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
            else:
                st.info("Feedback already submitted for this response.")
    elif last_response["agent"] == "Error":
        st.warning("Feedback unavailable due to an error in the last response.")
    else:
        st.warning("Feedback unavailable: No valid response ID.")

# Sidebar: Metrics Display
st.sidebar.header("Chatbot Performance Metrics")

eval_metrics = st.session_state.eval_metrics or {
    "total_conversations": 0,
    "active_users": 0,
    "avg_session_length": 0.0,
    "avg_response_time": 0.0,
    "resolution_rate": 0.0,
    "fallback_rate": 0.0,
    "accuracy": 0.0,
    "coherence": 0.0,
    "satisfaction": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0
}

st.sidebar.markdown('<div class="metrics-header">Overall Model Performance</div>', unsafe_allow_html=True)
st.sidebar.write(f"**Accuracy:** {eval_metrics['accuracy']:.2f}")
st.sidebar.write(f"**Coherence:** {eval_metrics['coherence']:.2f}")
st.sidebar.write(f"**Satisfaction:** {eval_metrics['satisfaction']:.2f}")
st.sidebar.write(f"**Precision:** {eval_metrics['precision']:.2f}")
st.sidebar.write(f"**Recall:** {eval_metrics['recall']:.2f}")
st.sidebar.write(f"**F1 Score:** {eval_metrics['f1_score']:.2f}")

st.sidebar.markdown('<div class="metrics-header">Operational Metrics</div>', unsafe_allow_html=True)
st.sidebar.write(f"**Total Conversations:** {eval_metrics['total_conversations']}")
st.sidebar.write(f"**Active Users:** {eval_metrics['active_users']}")
st.sidebar.write(f"**Avg Session Length:** {eval_metrics['avg_session_length']:.2f}")
st.sidebar.write(f"**Avg Response Time:** {eval_metrics['avg_response_time']:.2f}s")
st.sidebar.write(f"**Resolution Rate:** {eval_metrics['resolution_rate']:.2%}")
st.sidebar.write(f"**Fallback Rate:** {eval_metrics['fallback_rate']:.2%}")

if st.session_state.history:
    bot_responses = [chat for chat in st.session_state.history if chat["role"] == "bot"]
    if bot_responses:
        if st.session_state.current_topic and len(bot_responses) > 1:
            topic_responses = [r for r in bot_responses if r.get("topic") == st.session_state.current_topic]
            responses_to_use = topic_responses if topic_responses else [bot_responses[-1]]
        else:
            responses_to_use = [bot_responses[-1]]
        
        avg_accuracy = np.mean([float(r["metrics"]["accuracy"]) for r in responses_to_use])
        avg_coherence = np.mean([float(r["metrics"]["coherence"]) for r in responses_to_use])
        avg_satisfaction = np.mean([float(r["metrics"]["satisfaction"]) for r in responses_to_use])

        st.sidebar.markdown('<div class="metrics-header">Session Metrics</div>', unsafe_allow_html=True)
        st.sidebar.write(f"**Avg Accuracy:** {avg_accuracy:.2f}")
        st.sidebar.write(f"**Avg Coherence:** {avg_coherence:.2f}")
        st.sidebar.write(f"**Implicit Satisfaction:** {avg_satisfaction:.2f}")

        if st.session_state.current_topic in st.session_state.feedback_ratings:
            ratings = st.session_state.feedback_ratings[st.session_state.current_topic]
            if ratings:
                normalized_ratings = [(r - 1) / 4.0 for r in ratings]
                csat = np.mean(normalized_ratings)
                st.sidebar.write(f"**CSAT:** {csat:.2f} (based on {len(ratings)} ratings)")
            else:
                st.sidebar.write("**CSAT:** N/A (no ratings yet)")
        else:
            st.sidebar.write("**CSAT:** N/A (no ratings yet)")

        if st.session_state.current_topic:
            st.sidebar.markdown(f"**Current Topic:** {st.session_state.current_topic}")
else:
    st.sidebar.write("No interactions yet.")

if __name__ == "__main__":
    st.write("Running Concordia Chatbot...")
