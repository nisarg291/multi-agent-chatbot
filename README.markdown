# Concordia Multi-Agent Chatbot

## Overview

The **Concordia Multi-Agent Chatbot** is a sophisticated conversational platform designed to provide accurate and contextually relevant responses to a wide range of user queries. Leveraging the power of **Ollama** for local language model processing, this chatbot employs a multi-agent architecture to specialize in three key domains: Artificial Intelligence (AI), Concordia University admissions, and general knowledge. Built with **FastAPI** for the backend and **Streamlit** for an intuitive frontend, the system integrates advanced natural language processing (NLP) techniques, external APIs, and a persistent memory store to deliver a seamless user experience.

This project aims to assist users—whether students, researchers, or curious individuals—by offering tailored responses, follow-up suggestions, and performance metrics to ensure continuous improvement. The chatbot is designed for local deployment, making it ideal for educational institutions, research labs, or personal use cases requiring privacy and customization.

## Features

- **Multi-Agent Architecture**:
  - **AI Agent**: Handles queries related to machine learning, neural networks, and other AI topics with technical precision.
  - **Admissions Agent**: Provides detailed information on Concordia University’s programs, requirements, and processes, referencing official sources.
  - **General Agent**: Addresses a broad spectrum of topics, from lifestyle to academics, ensuring versatility.
- **Topic Detection**: Uses Ollama’s language model to classify queries into AI, Admissions, or General categories, ensuring the most relevant agent responds.
- **Context-Aware Responses**: Maintains conversation history via **ChromaDB**, enabling contextual understanding and personalized follow-up questions.
- **External API Integration**:
  - Google Custom Search for real-time web data.
  - Wikipedia for structured knowledge retrieval.
  - Gemini API for supplementary responses when needed.
- **Frontend Interface**: A user-friendly **Streamlit** application with a clean chat UI, real-time metrics display, and feedback submission.
- **Performance Monitoring**:
  - Tracks metrics like accuracy, coherence, and user satisfaction.
  - Logs interactions and feedback for evaluation and improvement.
- **Local Deployment**: Runs entirely on local infrastructure with Ollama, ensuring data privacy and low latency.

## Technologies Used

- **Backend**:
  - **FastAPI**: High-performance API framework for handling chat and feedback endpoints.
  - **Ollama**: Local LLM for topic detection, response generation, and follow-up suggestions.
  - **ChromaDB**: Persistent vector database for storing conversation history.
  - **Sentence Transformers**: For embedding-based similarity checks and relevance scoring.
  - **NLP Libraries**: SpaCy, NLTK, and CrossEncoder for text processing and sentiment analysis.
  - **External APIs**: Google Custom Search, Wikipedia, and Gemini for enriched responses.
- **Frontend**:
  - **Streamlit**: Interactive web interface for chat, metrics, and feedback.
  - **Custom CSS**: For a polished and professional UI design.
- **Other**:
  - **Python**: Core programming language for all components.
  - **Logging**: Comprehensive logging for debugging and performance tracking.
  - **AsyncIO**: For efficient handling of concurrent API requests.

## Installation

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- Ollama installed locally (see Ollama documentation)
- API keys for:
  - Google Custom Search (API Key and CSE ID)
  - Gemini API
  - Hugging Face (optional, for additional models)

### Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/concordia-multi-agent-chatbot.git
   cd concordia-multi-agent-chatbot
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Ollama**:

   - Install Ollama and pull the `llama3` model:

     ```bash
     ollama pull llama3
     ollama serve
     ```

4. **Configure Environment Variables**: Create a `.env` file in the root directory:

   ```
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_google_cse_id
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   BACKEND_URL=http://127.0.0.1:8000
   ```

   Load the variables:

   ```bash
   source .env
   ```

5. **Download SpaCy Model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Run the Application**:

   - Start the FastAPI backend:

     ```bash
     uvicorn main:app --host 0.0.0.0 --port 8000
     ```

   - In a separate terminal, launch the Streamlit frontend:

     ```bash
     streamlit run app.py
     ```

7. **Access the Chatbot**:

   - Open your browser and navigate to `http://localhost:8501` to interact with the Streamlit interface.
   - The backend API is available at `http://localhost:8000/docs` for testing endpoints.

## Usage

1. **Chat Interface**:

   - Enter your query in the text area (e.g., “What are Concordia’s admission requirements?” or “How does machine learning work?”).
   - Submit to receive a response from the appropriate agent (AI, Admissions, or General).
   - View follow-up suggestions to continue the conversation.
   - Use the “Clear Chat” button to reset the session.

2. **Feedback**:

   - Rate responses on a 1-5 scale to help improve the chatbot.
   - Feedback is stored and used to refine response selection.

3. **Metrics**:

   - The sidebar displays real-time performance metrics, including accuracy, coherence, satisfaction, and operational stats like response time and resolution rate.
   - Session-specific metrics show average performance for the current conversation topic.

4. **Logging**:

   - Interaction logs are saved to `evaluation_log.txt`.
   - Feedback logs are stored in `feedback_log.txt` for analysis.

## Example Queries

- **AI**: “What is reinforcement learning, and how is it used in robotics?”
- **Admissions**: “What are the GPA requirements for Concordia’s graduate programs?”
- **General**: “How can I improve my time management skills?”

## Project Structure

```
concordia-multi-agent-chatbot/
├── app.py                # Streamlit frontend
├── main.py               # FastAPI backend
├── chroma_customer_db/   # ChromaDB storage
├── evaluation_log.txt    # Interaction logs
├── feedback_log.txt      # Feedback logs
├── reward_memory.json    # Reward storage
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Contributing

I welcome contributions to enhance the chatbot's functionality, UI, or performance. To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description.

Please adhere to the code style and include tests for new features.

## Contact

For questions or support, contact the project maintainers at:

- Email: nadalja26@gmail.com
- GitHub Issues: Submit an issue

---

*Built with ❤️ for the Concordia University community and for all Students*