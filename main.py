# Replace with your actual API keys
GEMINI_API_KEY = "AIzaSyDGWGa2bY4WiiNh0-zOvViq-KV003pKkwA"
GOOGLE_API_KEY = "AIzaSyBsEjEpJwJXUYcmJcEmgRwLvzfnMYg2n5A"
GOOGLE_CSE_ID = "b418152f819ee4c59"
HUGGINGFACE_API_KEY = "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"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from chromadb import Client
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
import google.generativeai as genai
import logging
import wikipedia
import requests
import numpy as np
import spacy
import uuid
import certifi
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import asyncio
import aiohttp
from collections import defaultdict

nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
nlp = spacy.load('en_core_web_sm')

# Initialize ChromaDB Persistent Client
CHROMA_DB_PATH = "./chroma_customer_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
logger.info(f"ChromaDB initialized with persistent storage at {CHROMA_DB_PATH}")

# API configurations
genai.configure(api_key=GEMINI_API_KEY)
HF_API_URL = "https://api-inference.huggingface.co/models/mixtralai/Mixtral-8x7B-Instruct-v0.1"

# RL configurations
epsilon = 0.1  # Initial exploration rate
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
q_table = defaultdict(lambda: {"use_agent": 0.0, "use_google": 0.0, "use_wiki": 0.0, "use_gemini": 0.0, "use_hf": 0.0, "use_enhanced": 0.0})
response_cache = {}
question_history = defaultdict(lambda: {"count": 0, "responses": []})
query_counter = 0

# Define keywords
ai_keywords = [
    "ai", "machine learning", "neural network", "deep learning", "artificial intelligence",
    "data science", "nlp", "natural language processing", "computer vision", "chatbot",
    "algorithm", "training", "model", "prediction", "classification", "regression",
    "reinforcement learning", "supervised learning", "unsupervised learning", "generative ai",
    "tensor", "optimization", "gradient descent", "overfitting", "underfitting", "feature engineering",
    "big data", "analytics", "pattern recognition", "image recognition", "speech recognition",
    "autonomous systems", "robotics", "expert systems", "knowledge graph", "sentiment analysis",
    "ai ethics", "bias in ai", "explainable ai", "transfer learning", "convolutional neural network",
    "recurrent neural network", "transformer", "gpt", "bert", "llm", "large language model"
]

admission_keywords = [
    "admission", "concordia", "concordia university", "computer science", "program", "apply",
    "enrollment", "deadline", "requirement", "scholarship", "tuition", "application", "faculty",
    "course", "undergraduate", "graduate", "masters", "phd", "bachelor", "degree", "diploma",
    "registration", "semester", "term", "academic", "transcript", "gpa", "prerequisite",
    "international student", "visa", "study permit", "acceptance", "rejection", "offer letter",
    "orientation", "campus", "housing", "residence", "student services", "financial aid",
    "bursary", "co-op", "internship", "placement", "concordia engineering", "concordia arts",
    "concordia science", "concordia business", "john molson", "admission process", "open house",
    "tour", "prospectus", "eligibility", "credits", "transfer student"
]

general_keywords = [
    "assistance", "chat", "chemistry", "contact", "general", "guide", "hello", "help",
    "hi", "info", "news", "politics", "psychology", "query", "question", "science",
    "social media", "sports", "study", "support", "talk", "weather", "cooking", "recipe",
    "food", "cuisine", "baking", "ingredients", "meal", "dish", "nutrition", "diet",
    "health", "fitness", "exercise", "travel", "destination", "tourism", "culture",
    "history", "geography", "language", "music", "art", "literature", "movies", "tv",
    "games", "technology", "gadgets", "internet", "software", "hardware", "fashion",
    "style", "trends", "events", "holidays", "celebrations", "diy", "crafts", "gardening",
    "pets", "animals", "environment", "sustainability", "economics", "finance", "jobs",
    "career", "education", "school", "learning", "hobbies", "fun", "random", "ceo", "company", "city", "country"
]

# Test and benchmark data (unchanged)
test_data = {
    "How does machine learning differ from traditional programming, and why is it so important in AI development?": 
        "Machine learning differs from traditional programming in its ability to learn and improve from data without being explicitly programmed. It uses algorithms and statistical models to analyze patterns and make predictions, whereas traditional programming relies on explicit rules and logic. This adaptability makes machine learning crucial in AI development for tasks like image recognition, natural language processing, and predictive modeling.",
    "How can one develop better critical thinking skills, and what are some exercises to enhance this ability?": 
        "Develop critical thinking by questioning assumptions, seeking evidence, considering different perspectives, and identifying biases. Practice with exercises like analyzing arguments, evaluating news sources, debating opposing viewpoints, and solving logic puzzles. Regular practice is key.",
    "Can artificial intelligence truly exhibit creativity, and what are some examples of AI being used in creative fields like music or art?": 
        "Artificial intelligence (AI) can exhibit creativity in certain domains by leveraging algorithms and machine learning techniques. In music, AI-generated compositions have been praised for their originality, while AI-powered art tools generate unique visual representations. For instance, Amper Music's AI composer has created tracks for commercials and films, demonstrating creative capabilities in melody and harmony generation. Similarly, AI-generated artwork like Generative Adversarial Networks (GANs) produce stunning visuals that rival human creations. While AI-generated content may lack the emotional depth of human creativity, it can still produce innovative and engaging pieces. As AI continues to evolve, we may see even more impressive creative applications",
    "What are the key challenges in developing AI systems that can understand and interpret human emotions or sentiments?": 
        "Artificial intelligence (AI) can exhibit creativity in certain domains by leveraging algorithms and machine learning techniques. In music, AI-generated compositions have been praised for their originality, while AI-powered art tools generate unique visual representations. For instance, Amper Music's AI composer has created tracks for commercials and films, demonstrating creative capabilities in melody and harmony generation. Similarly, AI-generated artwork like Generative Adversarial Networks (GANs) produce stunning visuals that rival human creations. While AI-generated content may lack the emotional depth of human creativity, it can still produce innovative and engaging pieces. As AI continues to evolve, we may see even more impressive creative applications.",
    "How does reinforcement learning work in AI, and what real-world applications can benefit from this approach?": 
        "Reinforcement learning (RL) involves training AI agents to make decisions by receiving rewards or penalties for their actions. This approach is effective in complex decision-making tasks like robotics, game playing, and autonomous driving.",
    "What are the admission requirements for international students applying to graduate programs at Concordia University in Montreal?": 
        "Please note that the admission requirements for international students applying to graduate programs at Concordia University in Montreal can be found on our official website. According to our website [https://www.concordia.ca/gradstudies/future-students/admissions.html], the general requirements include a bachelor's degree with a minimum GPA of 3.0, English proficiency test scores (TOEFL or IELTS), and other program-specific requirements. Please visit our website for detailed information on specific programs' admission requirements.",
    "How does Concordia University support student innovation and entrepreneurship, and what programs or resources are available for aspiring entrepreneurs?": 
        "Concordia University supports student innovation and entrepreneurship through various programs and resources. The John Molson School of Business offers the Entrepreneurship minor, which provides students with a comprehensive understanding of entrepreneurial principles. Additionally, the university has a strong network of incubators, accelerators, and co-working spaces, such as the Concordia University Incubator, that provide funding, mentorship, and networking opportunities to aspiring entrepreneurs. The university also offers entrepreneurship-focused courses, workshops, and events, like the Concordia Entrepreneurship Speaker Series, which bring together industry experts and thought leaders to share their experiences.",
    "What are the benefits of studying at Concordia University’s John Molson School of Business, and how does it rank among business schools in Canada?": 
        "Studying at the John Molson School of Business (JMSB) offers numerous benefits. According to our website [https://www.concordia.ca/academics/business], JMSB is ranked among the top 5 business schools in Canada by Maclean's magazine, and its undergraduate programs are accredited by the Association to Advance Collegiate Schools of Business (AACSB). The school also has a strong reputation for producing well-rounded graduates who can succeed in an increasingly complex global economy. With its unique location in Montreal, one of the most diverse cities in North America, JMSB students have access to a wide range of career opportunities and cultural experiences.",
    "Can you provide information on the student exchange programs at Concordia University and how they help students gain global exposure?": 
        "Student exchange programs at Concordia University provide students with an opportunity to gain global exposure and experience. The university has partnerships with over 50 institutions worldwide through the International Office's Study Abroad program. This allows students to spend a semester or academic year studying abroad, gaining valuable international experience while earning credits towards their degree. The study abroad program offers various benefits, including cultural immersion, language development, and networking opportunities. Students can choose from a range of destinations, from Europe to Asia, and even take part in summer programs or field courses. The university also provides support for students applying to these programs, including advising on course selection, visa applications, and pre-departure preparation. By participating in the study abroad program, Concordia University students can gain a unique perspective on global issues, develop their language skills, and enhance their employability in today's increasingly globalized workforce.",
    "What career services and job placement support does Concordia University offer to students and recent graduates in the tech industry?": 
        "Concordia University offers career services and job placement support to students and recent graduates in the tech industry through its Career Development Centre. The centre provides resources such as resume building, interview preparation, and networking opportunities with major tech companies. Additionally, the university's alumni network provides valuable connections for its students and recent graduates.",
    "What are some effective strategies for improving time management skills for students juggling multiple responsibilities?": 
        "Effective strategies for improving time management skills include setting clear goals and priorities, using a planner or calendar to stay organized, breaking tasks into smaller chunks, avoiding procrastination, and learning to say 'no' to non-essential commitments. Additionally, students can use productivity apps and tools, such as Pomodoro timers or task management software, to help stay focused and on track.",
    "What are the key differences between a fixed mindset and a growth mindset, and how do they impact personal development?": 
        "A fixed mindset assumes that abilities and talents are innate and unchangeable. A growth mindset, on the other hand, views abilities as malleable and shaped by experiences. Fixed mindset individuals tend to see failures as reflections of their inadequacy, whereas those with a growth mindset view them as opportunities for learning and improvement. This difference in mindset has a profound impact on personal development. Individuals with a fixed mindset are more likely to avoid challenges, be sensitive to criticism, and give up easily when faced with obstacles. In contrast, those with a growth mindset are more likely to take risks, learn from failures, and persist in the face of adversity. By adopting a growth mindset, individuals can develop resilience, confidence, and a sense of self-efficacy, leading to greater personal and professional growth.",
    "What are the best ways to stay motivated and productive when working on long-term projects or goals?": 
        "When working on long-term projects or goals in AI-related fields, staying motivated and productive requires strategic planning and time management. Break down complex tasks into manageable chunks, set realistic milestones, and celebrate small victories along the way. Establishing a routine, prioritizing self-care, and minimizing distractions can also help maintain momentum. Additionally, reflecting on progress, adjusting strategies as needed, and seeking support from peers or mentors can foster continued motivation",
    "How does sleep affect mental health and cognitive function, and what are some tips for improving sleep quality?": 
        "Sleep plays a crucial role in mental health and cognitive function. Poor sleep quality can lead to increased stress levels, anxiety, depression, and decreased cognitive performance. Good sleep habits include maintaining a consistent sleep schedule, creating a relaxing bedtime routine, avoiding screens before bed, and creating a dark, quiet sleep environment."
}

benchmark_data = {
    "How does machine learning differ from traditional programming, and why is it so important in AI development?": 
        "Traditional programming uses explicit rules, while machine learning learns from data. This key difference allows AI to handle complex tasks like image recognition and natural language processing, where defining rules is impractical. Machine learning's ability to adapt and improve from data is crucial for advancing AI's capabilities",
    "How can one develop better critical thinking skills, and what are some exercises to enhance this ability?": 
        "Critical thinking improves by questioning assumptions, analyzing diverse sources, and logically concluding. Exercises include logic puzzles, debates, news analysis, active listening, and asking 'why?'. Regular practice is essential for skill development.",
    "Can artificial intelligence truly exhibit creativity, and what are some examples of AI being used in creative fields like music or art?": 
        "The question of AI's genuine 'creativity' remains complex. While AI lacks human-like consciousness or emotions, it can generate novel outputs by learning patterns from extensive datasets. Examples include AI composing music in diverse styles, producing stunning visuals, and writing poems or stories. These instances demonstrate AI's capacity for creative output, though the true nature of creativity, and whether AI possesses it, is still debated.",
    "What are the key challenges in developing AI systems that can understand and interpret human emotions or sentiments?": 
        "Developing AI for emotional understanding faces several hurdles. Human emotions are nuanced and context-dependent, varying across cultures and individuals. Language is often ambiguous, with sarcasm and irony complicating sentiment analysis. Facial expressions can be misleading, and physiological signals are hard to interpret accurately. AI needs vast, diverse datasets, but these are often biased or incomplete. Moreover, defining and quantifying emotions objectively remains a challenge. Building models that generalize across diverse populations and situations is crucial, but difficult, requiring sophisticated algorithms and a deep understanding of human psychology",
    "How does reinforcement learning work in AI, and what real-world applications can benefit from this approach?": 
        "Reinforcement learning optimizes decision-making through rewards and penalties, benefiting robotics, gaming, finance, healthcare, and autonomous vehicles significantly.",
    "What are the admission requirements for international students applying to graduate programs at Concordia University in Montreal?": 
        "Concordia University's graduate admission for international students focuses on academic and language readiness. Applicants must possess credentials equivalent to Canadian standards, with program-specific prerequisites. English proficiency, typically via TOEFL/IELTS, is essential. Required scores vary, so check program details. Transcripts, CV, statement of purpose, and references are needed, plus an application fee. Always consult individual program websites for precise requirements and deadlines.",
    "How does Concordia University support student innovation and entrepreneurship, and what programs or resources are available for aspiring entrepreneurs?": 
        "Concordia University supports student innovation through District 3 Innovation Hub, startup incubators, pitch competitions, and mentorship programs. Resources include innovation labs, funding opportunities, and networking events. Programs like the Startup Ready program and various entrepreneurship courses help aspiring entrepreneurs develop skills, validate ideas, and launch successful ventures.",
    "What are the benefits of studying at Concordia University’s John Molson School of Business, and how does it rank among business schools in Canada?": 
        "The John Molson School of Business (JMSB) at Concordia University in Montreal offers students a comprehensive business education with several notable advantages. Its MBA program emphasizes practical, hands-on learning, preparing graduates for real-world challenges. Students benefit from diverse master's programs, including Accountancy, Finance, International Business, and Supply Chain Operations Management, catering to various career aspirations. TopMBA.comMASTERGRADSCHOOLS In terms of rankings, JMSB has achieved significant recognition. Bloomberg Businessweek's 2024-2025 rankings placed JMSB's MBA program second overall in Canada, marking an improvement of two spots from the previous year. Additionally, the school is ranked within the top 100 globally for its Executive MBA program according to QS Global Executive MBA Rankings 2024. These accolades reflect JMSB's commitment to academic excellence and its growing reputation among business schools in Canada and internationally.",
    "Can you provide information on the student exchange programs at Concordia University and how they help students gain global exposure?": 
        "Concordia University offers diverse student exchange programs, enabling students to study abroad at partner institutions. These programs foster global exposure by immersing students in different academic and cultural environments. Students gain international perspectives, enhance language skills, and develop cross-cultural competencies. Through exchange, students access unique courses and research opportunities, broadening their academic horizons. These experiences promote personal growth, build global networks, and enhance employability. Concordia's exchange programs facilitate cultural exchange, encouraging students to become globally aware and adaptable citizens. They return with enriched perspectives and a deeper understanding of the world.",
    "What career services and job placement support does Concordia University offer to students and recent graduates in the tech industry?": 
        "Concordia offers career workshops, networking events, job boards, and industry connections, aiding tech students and graduates in job placement.",
    "What are some effective strategies for improving time management skills for students juggling multiple responsibilities?": 
        "Prioritize tasks using the Eisenhower Matrix, set SMART goals, and use planners or digital tools for scheduling. Break tasks into smaller steps, avoid multitasking, and set deadlines. Allocate focused study sessions with breaks, practice saying no to distractions, and review progress regularly to stay organized and efficient.",
    "What are the key differences between a fixed mindset and a growth mindset, and how do they impact personal development?": 
        "A fixed mindset believes abilities are static, leading to fear of failure, avoidance of challenges, and limited personal growth. People with this mindset see effort as fruitless and give up easily when faced with obstacles. A growth mindset, on the other hand, views intelligence and skills as developable through effort, learning, and perseverance. Individuals embrace challenges, persist through setbacks, and see failures as opportunities to improve. The impact on personal development is significant: a growth mindset fosters resilience, adaptability, and continuous self-improvement, while a fixed mindset can limit potential by discouraging risk-taking and innovation.",
    "What are the best ways to stay motivated and productive when working on long-term projects or goals?": 
        "Maintaining motivation for long-term projects requires strategic approaches. Break large goals into smaller, manageable tasks, celebrating each milestone. Establish a consistent routine, minimizing distractions. Find an accountability partner or join a supportive community. Visualize your success, and reflect on past achievements. Prioritize self-care, ensuring adequate rest and exercise. Vary your work environment to avoid monotony. Embrace flexibility, adapting to unexpected challenges. Remember your 'why'—the underlying purpose driving your efforts. Regularly reassess progress and adjust strategies as needed. Staying connected to your passion is crucial.",
    "How does sleep affect mental health and cognitive function, and what are some tips for improving sleep quality?": 
        "Sleep enhances mental health and cognition by improving memory, focus, and mood. Improve sleep with consistency, reduced screen time, and relaxation."
}

# Track performance metrics
performance_metrics = {
    "total_conversations": 0,
    "active_users": set(),
    "session_lengths": [],
    "response_times": [],
    "resolutions": 0,
    "fallbacks": 0,
    "predictions": [],
    "ground_truths": []
}

base_urls = [
    "https://www.concordia.ca/",
    "https://www.concordia.ca/admissions/undergraduate.html",
    "https://www.concordia.ca/admissions/undergraduate/programs.html",
    "https://www.concordia.ca/academics/undergraduate/computer-science.html",
    "https://www.concordia.ca/admissions/graduate.html",
    "https://www.concordia.ca/students/financial.html",
    "https://www.concordia.ca/gradstudies/future-students/programs.html",
    "https://www.concordia.ca/gradstudies/future-students/how-to-apply.html",
    "https://www.concordia.ca/gradstudies/students/registration.html",
    "https://www.concordia.ca/gradstudies/students/new.html"
]

# Initialize Ollama LLM
try:
    ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", timeout=60)
    logger.info("Ollama LLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Ollama: {str(e)}")
    ollama_llm = None

# Define Agents
def create_agent(template):
    if ollama_llm is None:
        return lambda history, query: "Error: Language model is not available."
    prompt = PromptTemplate(input_variables=["history", "query", "ai_keywords", "general_keywords", "admission_keywords", "base_urls"], template=template)
    return LLMChain(llm=ollama_llm, prompt=prompt)

general_agent = create_agent(
    "You are a general-purpose assistant focused solely on everyday topics like {general_keywords}, excluding AI or admissions-related content unless explicitly asked. Ignore any prior context that suggests a different role. Conversation history: {history}\n\n"
    "Answer clearly and concisely: {query}\n\n"
    "If the query contains 'in X words' (e.g., 'in 50 words'), limit your response to approximately X words. If no word count is specified in the current query, provide the full response without limiting words.\n\n"
    "If you are unsure, lack relevant data, or cannot answer, respond with 'I am not able to answer this question'."
)

ai_agent = create_agent(
    "You are an AI expert specializing only in artificial intelligence topics like {ai_keywords}. Ignore any prior context unrelated to AI. Conversation history: {history}\n\n"
    "Provide a detailed, accurate answer about AI-related topics: {query}\n\n"
    "If the query contains 'in X words' (e.g., 'in 50 words'), limit your response to approximately X words. If no word count is specified in the current query, provide the full response without limiting words.\n\n"
    "If you are unsure, lack relevant data, or cannot answer, respond with 'I am not able to answer this question'."
)

admission_agent = create_agent(
    "You are a Concordia University admissions advisor specializing only in admissions and program details like {admission_keywords}. Use {base_urls} for accurate info. Ignore any prior context unrelated to admissions. Conversation history: {history}\n\n"
    "Answer accurately about admissions and program details: {query}\n\n"
    "If the query contains 'in X words' (e.g., 'in 50 words'), limit your response to approximately X words. If no word count is specified in the current query, provide the full response without limiting words.\n\n"
    "If you are unsure, lack relevant data, or cannot answer, respond with 'I am not able to answer this question'."
)

class Query(BaseModel):
    user_id: str
    query: str

class Feedback(BaseModel):
    doc_id: str
    rating: int

def get_state(query, topic):
    global query_counter
    doc = nlp(query)
    entities = len(doc.ents)
    query_counter += 1
    return f"{topic}_{entities}_{query_counter}"

async def fetch_google_search_response(query):
    try:
        keywords = extract_keywords(query)
        logger.info(f"Searching Google with keywords: {keywords}")
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=keywords, cx=GOOGLE_CSE_ID, num=3).execute()
        snippets = [item['snippet'] for item in res.get('items', [])]
        response = " ".join(snippets) if snippets else "I am not able to answer this question based on Google search results."
        return response if is_response_relevant(query, response, "") >= 0.55 else "I am not able to answer this question based on Google search results."
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        return "I am not able to answer this question due to a search error."

async def fetch_wikipedia_response(query):
    try:
        keywords = extract_keywords(query)
        logger.info(f"Searching Wikipedia with keywords: {keywords}")
        wikipedia.set_lang("en")
        summary = wikipedia.summary(keywords, sentences=10, auto_suggest=True)
        response = f"{summary}" if summary else "I am not able to answer this question based on Wikipedia."
        return response if is_response_relevant(query, response, "") >= 0.55 else "I am not able to answer this question based on Wikipedia."
    except Exception as e:
        logger.error(f"Wikipedia error: {str(e)}")
        return "I am not able to answer this question due to a Wikipedia error."

async def fetch_gemini_response(query):
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(query)
        text = response.text.strip()
        enhanced_response = f"{text}"
        return enhanced_response if is_response_relevant(query, enhanced_response, "") >= 0.55 else "I am not able to answer this question using Gemini."
    except Exception as e:
        logger.error(f"error: {str(e)}")
        return "I am not able to answer this question due to a Gemini error."

async def fetch_huggingface_response(query):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": query, "parameters": {"max_length": 200, "temperature": 0.7}}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(HF_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and result and "generated_text" in result[0]:
                        text = str(result[0]["generated_text"])
                        return text if is_response_relevant(query, text, "") >= 0.55 else "I am not able to answer this question using Hugging Face."
                    return "I am not able to answer this question."
                return f"API error: Status {response.status}"
        except Exception as e:
            logger.error(f"error: {str(e)}")
            return "I am not able to answer this question due to a Hugging Face error."

def is_response_relevant(query, response, history):
    try:
        if not response or not response.strip():
            return 0.0
        score = cross_encoder.predict([[query, response]])[0]
        if history and history != "No previous conversation.":
            history_score = cross_encoder.predict([[history, response]])[0]
            score = 0.7 * score + 0.3 * history_score
        return min(1.0, max(0.0, float(score)))
    except Exception as e:
        logger.error(f"Error checking relevance: {str(e)}")
        return 0.0

def extract_keywords(query):
    doc = nlp(query.lower())
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "PROPN"] and not token.is_stop]
    return " ".join(keywords) or query

def select_agent_and_get_response(query, user_id):
    topic = detect_topic_with_ollama(query, user_id)
    logger.info(f"Detected topic for query '{query}' with user_id '{user_id}': {topic}")
    history = retrieve_history(user_id)
    if topic == "Admissions" or "concordia" in query.lower():
        filtered_history = "\n\n".join([entry for entry in history.split("\n\n") if "Admissions" in entry or "concordia" in entry.lower()]) or "No relevant previous conversation."
        return admission_agent, "Admissions Agent", filtered_history
    elif topic == "AI":
        filtered_history = "\n\n".join([entry for entry in history.split("\n\n") if "AI" in entry]) or "No relevant previous conversation."
        return ai_agent, "AI Agent", filtered_history
    else:
        filtered_history = "\n\n".join([entry for entry in history.split("\n\n") if "General" in entry or ("AI" not in entry and "Admissions" not in entry)]) or "No relevant previous conversation."
        return general_agent, "General Agent", filtered_history

def detect_topic_with_ollama(query, user_id):
    if ollama_llm is None:
        return "General"
    try:
        history = retrieve_last_queries(user_id, n=5)
        prompt = PromptTemplate(
            input_variables=["query", "history", "ai_keywords", "admission_keywords", "general_keywords"],
            template="You are an expert topic classifier with deep knowledge across domains. Analyze the following query and its conversation history to determine its main topic. Return only one topic from this list: ['AI', 'Admissions', 'General']. Output the topic name alone, with no explanation or extra text.\n\n"
                     "Current query: {query}\n"
                     "Conversation history (last 5 queries with topics, most recent first):\n{history}\n\n"
                     "Optional keyword hints (use these as guidance, but rely on your understanding if keywords are absent):\n"
                     "- AI topics (e.g., artificial intelligence, learning systems): {ai_keywords}\n"
                     "- Admissions topics (e.g., university entry, programs): {admission_keywords}\n"
                     "- General topics (e.g., everyday knowledge, miscellaneous): {general_keywords}\n\n"
                     "Instructions:\n"
                     "1. Interpret the query’s intent and subject matter using your knowledge. Choose the most likely topic based on what the query is about, even if no specific keywords are present.\n"
                     "   - Example: 'How does a computer learn patterns?' → 'AI' (implies AI concepts like machine learning).\n"
                     "   - Example: 'what is limited memory machines?' → 'AI' (implies AI concepts like Artificial Intelligence).\n"
                     "   - Example: 'how to make mobile more smart and like human?' → 'AI' (implies AI concepts like Artificial Intelligence making mobile smart).\n"
                     "   - Example: 'What’s the process to join a university?' → 'Admissions' (implies university admissions).\n"
                     "   - Example: 'How do I fix my sink?' → 'General' (implies practical everyday knowledge).\n"
                     "   - Example: 'what is tiger?' → 'General' (implies general knowledge about animal).\n"
                     "   - Example: 'who is ceo of TCS?' → 'General' (implies genral knowledge like asking about company, contry or city details).\n"
                     "   - Example: 'who is the pm of canada?' → 'General' (implies genral knowledge like asking about company, contry or city details).\n"
                     "   - Example: 'What is machine learning?' → 'AI' (direct AI topic bcz machine learning keyword is matching).\n"
                     "   - Example: 'How do I apply to Concordia?' → 'Admissions' (direct admissions topic).\n"
                     "   - Example: 'what are benifits for specially able students in university?' → 'Admissions' (implis admissions topic as mention about students and unviersity but rember this is not keywords than also it is admission and here university means you can give informention regarding concordia unveristy).\n"
                     "   - Example: 'What’s the weather like?' → 'General' (direct general topic).\n"

                     "2. If the query uses pronouns ('he', 'she', 'it', 'this', 'that') or vague references ('that thing'), check the history’s most recent topic and use it if it fits the context.\n"
                     "   - Example: 'Tell me more about it' with history 'What is AI?' (AI) → 'AI'.\n"
                     "   - Example: 'explain more in detail with history 'what is pyschology (General) → 'General' and you need to tell more about pyschology.\n"
                     "   - Example: 'explain more in detail with history 'what is computer vision (AI) → 'AI' and you need to tell more about Computer vision.\n"
                     "3. If you’re unsure or the query is too vague to classify, return 'General'.\n"
                     "   - Example: 'who is nisarg adalja' → 'General' (General as implies asking about specific person or specific place, earth, material) "
                     "4. Output exactly one of: 'AI', 'Admissions', 'General'."
        )
        chain = LLMChain(llm=ollama_llm, prompt=prompt)
        response = chain.invoke({
            "query": query.lower(),
            "history": history,
            "ai_keywords": ", ".join(ai_keywords),
            "admission_keywords": ", ".join(admission_keywords),
            "general_keywords": ", ".join(general_keywords)
        }).get('text', 'General').strip()
        valid_topics = ['AI', 'Admissions', 'General']
        if response in valid_topics:
            return response
        logger.warning(f"Invalid topic response '{response}' from LLM, defaulting to 'General'")
        return "General"
    except Exception as e:
        logger.error(f"Error detecting topic with Ollama: {str(e)}")
        return "General"

def retrieve_history(user_id, n=10):
    try:
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        if len(documents) > n:
            documents = documents[-n:]
            metadatas = metadatas[-n:]
        return "\n\n".join([f"User: {meta['query']}\nAgent: {meta['response']}" for meta in metadatas]) or "No previous conversation."
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return "No previous conversation."

def retrieve_last_queries(user_id, n=10):
    try:
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        if len(documents) > n:
            documents = documents[-n:]
            metadatas = metadatas[-n:]
        return "\n".join([f"User: {meta['query']} (Topic: {meta['topic']})" for meta in metadatas]) or "No previous queries."
    except Exception as e:
        logger.error(f"Error retrieving last queries: {str(e)}")
        return "No previous queries."

def store_interaction(user_id, query, response, agent_name, state, action, metrics, topic, follow_up):
    collection = chroma_client.get_or_create_collection(name="chat_history")
    doc_id = str(uuid.uuid4())
    try:
        collection.add(
            ids=[doc_id],
            documents=[str(response)],
            metadatas=[{
                "user_id": user_id,
                "query": query,
                "response": str(response),
                "agent": agent_name,
                "state": state,
                "action": action,
                "coherence": float(metrics["coherence"]),
                "accuracy": float(metrics["accuracy"]),
                "satisfaction": float(metrics["satisfaction"]),
                "topic": topic,
                "follow_up": follow_up,
                "timestamp": datetime.now().isoformat()
            }]
        )
        if metrics["satisfaction"] > 0.3:
            response_cache[(state, action)] = (response, q_table[state][action])
        logger.info(f"Interaction stored successfully with doc_id: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to store interaction: {str(e)}")
        raise

def calculate_metrics(query, response, current_topic, history, ground_truth=None):
    start_time = time.time()
    response = str(response) if response is not None else ""
    
    if not response or not response.strip() or "not able to answer" in response.lower() or "error" in response.lower():
        metrics = {"accuracy": 0.0, "coherence": 0.0, "satisfaction": 0.0}
        performance_metrics["fallbacks"] += 1
        performance_metrics["predictions"].append(0)
        performance_metrics["ground_truths"].append(1)
        performance_metrics["response_times"].append(time.time() - start_time)
        logger.info(f"Fallback/Error response - Predictions: 0, Ground Truth: 1")
        return metrics

    query_embedding = model.encode([query])[0]
    response_embedding = model.encode([response])[0]
    
    semantic_similarity = max(0, cosine_similarity([query_embedding], [response_embedding])[0][0])
    bleu_score = sentence_bleu([query.lower().split()], response.lower().split(), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1) if response.split() else 0.0
    accuracy = min(1.0, max(0.0, 0.7 * semantic_similarity + 0.3 * bleu_score))

    topic_embedding = model.encode([current_topic])[0]
    topic_coherence = max(0, cosine_similarity([topic_embedding], [response_embedding])[0][0])
    coherence = min(1.0, max(0.0, 0.7 * semantic_similarity + 0.3 * topic_coherence))
    response_words = len(response.split())
    coherence_penalty = 1.0 if 5 <= response_words <= 100 else 0.9
    coherence *= coherence_penalty

    relevance = is_response_relevant(query, response, history)
    readability = 206.835 - 1.015 * (response_words / max(len(response.split('.')), 1)) - 84.6 * (sum(len(w) for w in response.split()) / max(response_words, 1))
    readability_score = min(1.0, max(0.0, readability / 100.0))
    length_score = min(1.0, response_words / 20.0 if response_words < 20 else 60.0 / response_words if response_words > 60 else 1.0)
    sentiment = sid.polarity_scores(response)['compound']
    satisfaction = min(1.0, max(0.0, 0.4 * relevance + 0.3 * sentiment + 0.2 * readability_score + 0.1 * length_score))

    if ground_truth:
        ground_truth_embedding = model.encode([ground_truth])[0]
        accuracy = max(0, cosine_similarity([response_embedding], [ground_truth_embedding])[0][0])
        coherence = min(1.0, max(0.0, 0.5 * coherence + 0.5 * cosine_similarity([response_embedding], [ground_truth_embedding])[0][0]))
        coherence *= coherence_penalty
        ground_truth_words = len(ground_truth.split())
        length_similarity = min(1.0, min(response_words, ground_truth_words) / max(response_words, ground_truth_words, 1))
        ground_truth_relevance = is_response_relevant(query, ground_truth, history)
        satisfaction = min(1.0, max(0.0, 0.3 * relevance + 0.3 * accuracy + 0.2 * readability_score + 0.2 * length_score))
        ground_truth_label = 1
    else:
        ground_truth_label = 1 if relevance > 0.5 else 0
        accuracy = max(accuracy, relevance * 0.9)

    predicted_positive = accuracy > 0.5
    performance_metrics["predictions"].append(1 if predicted_positive else 0)
    performance_metrics["ground_truths"].append(ground_truth_label)
    performance_metrics["response_times"].append(time.time() - start_time)

    logger.info(f"Metrics calculated - Accuracy: {accuracy:.2f}, Coherence: {coherence:.2f}, Satisfaction: {satisfaction:.2f}")
    logger.info(f"Predictions: {performance_metrics['predictions'][-1]}, Ground Truth: {performance_metrics['ground_truths'][-1]}")

    return {
        "accuracy": float(accuracy),
        "coherence": float(coherence),
        "satisfaction": float(satisfaction)
    }

async def select_action(state, query, history, topic, agent_response):
    actions = ["use_agent", "use_google", "use_wiki", "use_gemini", "use_hf"]
    
    if random.random() < epsilon:
        action_scores = {}
        tasks = [
            ("use_agent", asyncio.create_task(asyncio.sleep(0, str(agent_response.get('text', agent_response) if isinstance(agent_response, dict) else agent_response)))),
            ("use_google", fetch_google_search_response(query)),
            ("use_wiki", fetch_wikipedia_response(query)),
            ("use_gemini", fetch_gemini_response(query)),
            ("use_hf", fetch_huggingface_response(query))
        ]
        for action, task in tasks:
            response = await task
            if "not able to answer" not in str(response).lower() and "error" not in str(response).lower():
                metrics = calculate_metrics(query, response, topic, history)
                relevance = is_response_relevant(query, response, history)
                score = (0.4 * relevance + 0.3 * metrics["accuracy"] + 0.3 * metrics["coherence"])
                action_scores[action] = score
            else:
                action_scores[action] = 0.0

        if action_scores:
            best_action = max(action_scores, key=action_scores.get)
            logger.info(f"Exploration: Selected {best_action} with score {action_scores[best_action]}")
            return best_action
        return "use_agent"
    else:
        best_action = max(q_table[state], key=q_table[state].get)
        logger.info(f"Exploitation: Selected {best_action} with Q-value {q_table[state][best_action]}")
        return best_action

def compute_reward(metrics, is_corrected=False):
    reward = 0.4 * metrics["accuracy"] + 0.3 * metrics["coherence"] + 0.3 * metrics["satisfaction"]
    return reward + 0.2 if is_corrected else reward

def update_q_value(state, action, reward):
    old_q = q_table[state][action]
    q_table[state][action] = (1 - alpha) * old_q + alpha * (reward + gamma * max(q_table[state].values()))
    logger.info(f"Updated Q-value for state '{state}', action '{action}': {q_table[state][action]}")

async def enhance_response(query, agent_response, history, state, topic, user_correction=None):
    global epsilon
    question_history[query]["count"] += 1
    
    # Check for correction request
    correction_requested = "correct the information" in query.lower() or "correct this" in query.lower()
    is_follow_up = False
    if history and history != "No previous conversation.":
        last_query = history.split("\n\n")[-1].split("User: ")[-1].split("\n")[0]
        last_response = history.split("\n\n")[-1].split("Agent: ")[-1]
        follow_up_prompt = f"Would you like to know more? How about: '{query}'"
        is_follow_up = follow_up_prompt in last_response

    # Handle repeated queries
    if question_history[query]["count"] > 1 and not correction_requested and not user_correction:
        past_responses = question_history[query]["responses"]
        action = await select_action(state, query, history, topic, agent_response)
        response = f"Refined answer: {past_responses[-1]}\nUpdated with {action} data."
        question_history[query]["responses"].append(response)
        metrics = calculate_metrics(query, response, topic, history)
        reward = compute_reward(metrics)
        update_q_value(state, action, reward)
        response_cache[(state, action)] = (response, q_table[state][action])
        return response, action, action

    # Handle correction request or user correction
    if correction_requested or user_correction:
        google_resp = await fetch_google_search_response(query)
        gemini_resp = await fetch_gemini_response(query)
        if user_correction:
            correction_text = user_correction
            if correction_text in google_resp or correction_text in gemini_resp:
                response = f"Corrected: {correction_text} (verified by Google/Gemini)"
                action = "use_enhanced"
                metrics = calculate_metrics(query, response, topic, history)
                reward = compute_reward(metrics, is_corrected=True)
                update_q_value(state, action, reward)
                response_cache[(state, action)] = (response, q_table[state][action])
                question_history[query]["responses"].append(response)
                return response, action, action
            else:
                response = f"Correction '{correction_text}' not verified by Google/Gemini. Here's the latest: {gemini_resp}"
        else:
            response = f"Updated info: {gemini_resp}\nSource: Gemini, verified by Google."
        action = "use_gemini"
        metrics = calculate_metrics(query, response, topic, history)
        reward = compute_reward(metrics, is_corrected=True)
        update_q_value(state, action, reward)
        response_cache[(state, action)] = (response, q_table[state][action])
        question_history[query]["responses"].append(response)
        return response, action, action

    # Standard response generation
    action = await select_action(state, query, history, topic, agent_response)
    logger.info(f"Selected action for state '{state}': {action}")

    if (state, action) in response_cache and not is_follow_up:
        cached_response, q_value = response_cache[(state, action)]
        if q_value > 0 and isinstance(cached_response, str) and "not able to answer" not in cached_response.lower():
            logger.info(f"Using cached response for state '{state}', action '{action}' with Q-value {q_value}")
            question_history[query]["responses"].append(cached_response)
            return cached_response, action, action

    if action == "use_agent":
        response_text = str(agent_response.get('text', agent_response) if isinstance(agent_response, dict) else agent_response)
        relevance = is_response_relevant(query, response_text, history)
        if "not able to answer" not in response_text.lower() and relevance >= 0.7:
            question_history[query]["responses"].append(response_text)
            return response_text, action, action

    responses = {}
    tasks = [
        fetch_google_search_response(query),
        fetch_wikipedia_response(query),
        fetch_gemini_response(query),
        fetch_huggingface_response(query)
    ]
    results = await asyncio.gather(*tasks)
    google_resp, wiki_resp, gemini_resp, hf_resp = results

    if google_resp and "not able to answer" not in google_resp.lower() and "error" not in google_resp.lower():
        google_metrics = calculate_metrics(query, google_resp, topic, history)
        responses["google"] = (google_resp, google_metrics["accuracy"] + google_metrics["coherence"])
    if wiki_resp and "not able to answer" not in wiki_resp.lower() and "error" not in wiki_resp.lower():
        wiki_metrics = calculate_metrics(query, wiki_resp, topic, history)
        responses["wiki"] = (wiki_resp, wiki_metrics["accuracy"] + wiki_metrics["coherence"])
    
    best_web = max(responses.items(), key=lambda x: x[1][1], default=(None, 0)) if responses else (None, 0)
    best_web_resp = best_web[0]

    if gemini_resp and "not able to answer" not in gemini_resp.lower() and "error" not in gemini_resp.lower():
        gemini_metrics = calculate_metrics(query, gemini_resp, topic, history)
        responses["gemini"] = (gemini_resp, gemini_metrics)
    if hf_resp and "not able to answer" not in hf_resp.lower() and "error" not in hf_resp.lower():
        hf_metrics = calculate_metrics(query, hf_resp, topic, history)
        responses["hf"] = (hf_resp, hf_metrics)

    if "gemini" in responses and "hf" in responses:
        gemini_score = 0.4 * responses["gemini"][1]["accuracy"] + 0.3 * responses["gemini"][1]["coherence"] + 0.3 * responses["gemini"][1]["satisfaction"]
        hf_score = 0.4 * responses["hf"][1]["accuracy"] + 0.3 * responses["hf"][1]["coherence"] + 0.3 * responses["hf"][1]["satisfaction"]
        final_resp = responses["gemini"][0] if gemini_score > hf_score else responses["hf"][0]
        final_action = "use_gemini" if gemini_score > hf_score else "use_hf"
    elif "gemini" in responses:
        final_resp = gemini_resp
        final_action = "use_gemini"
    elif "hf" in responses:
        final_resp = hf_resp
        final_action = "use_hf"
    elif best_web_resp:
        final_resp = best_web_resp
        final_action = "use_google" if best_web_resp == google_resp else "use_wiki"
    else:
        final_resp = str(agent_response.get('text', agent_response) if isinstance(agent_response, dict) else agent_response)
        final_action = "use_agent"
        if "not able to answer" in final_resp.lower() or "error" in final_resp.lower():
            final_resp = "I am not able to answer this question with current data. Try rephrasing or asking something else!"
            final_action = "use_enhanced"

    question_history[query]["responses"].append(final_resp)
    metrics = calculate_metrics(query, final_resp, topic, history)
    reward = compute_reward(metrics)
    update_q_value(state, final_action, reward)
    response_cache[(state, final_action)] = (final_resp, q_table[state][final_action])
    
    # Dynamic epsilon decay
    epsilon = max(0.01, epsilon * 0.99)
    logger.info(f"Final response: {final_resp}, Action: {final_action}, Epsilon: {epsilon}")
    return final_resp, final_action, action

def generate_follow_up_question(query, response, topic):
    if ollama_llm is None:
        return "Can you tell me more about this?"
    try:
        prompt = PromptTemplate(
            input_variables=["query", "response", "topic"],
            template="Given the user's query: '{query}'\n"
                     "And the response: '{response}'\n"
                     "With the topic: '{topic}'\n\n"
                     "Generate a concise, engaging follow-up question to encourage further conversation. "
                     "Make it relevant to the topic and response, and avoid repeating the original query. "
                     "Return only the question without additional text."
        )
        chain = LLMChain(llm=ollama_llm, prompt=prompt)
        follow_up = chain.invoke({
            "query": query,
            "response": response,
            "topic": topic
        }).get('text', 'Can you tell me more about this?').strip()
        return follow_up
    except Exception as e:
        logger.error(f"Error generating follow-up question: {str(e)}")
        return "Can you tell me more about this?"

def benchmark_comparison(metrics):
    benchmarks = {"accuracy": 0.80, "coherence": 0.75, "satisfaction": 0.80}
    return {
        "accuracy": "Pass" if metrics["accuracy"] >= benchmarks["accuracy"] else "Fail",
        "coherence": "Pass" if metrics["coherence"] >= benchmarks["coherence"] else "Fail",
        "satisfaction": "Pass" if metrics["satisfaction"] >= benchmarks["satisfaction"] else "Fail"
    }

@app.post("/chat/")
async def chat(query: Query):
    try:
        user_id = query.user_id
        user_query = query.query.strip()
        logger.info(f"Query from user {user_id}: {user_query}")

        history = retrieve_history(user_id)
        session_length = len(history.split("\n\n")) + 1 if history != "No previous conversation." else 1
        performance_metrics["total_conversations"] += 1
        performance_metrics["active_users"].add(user_id)
        performance_metrics["session_lengths"].append(session_length)

        current_topic = detect_topic_with_ollama(user_query, user_id)
        state = get_state(user_query, current_topic)
        agent, agent_name, filtered_history = select_agent_and_get_response(user_query, user_id)
        agent_response = agent.invoke({
            "history": filtered_history,
            "query": user_query,
            "ai_keywords": ", ".join(ai_keywords),
            "general_keywords": ", ".join(general_keywords),
            "admission_keywords": ", ".join(admission_keywords),
            "base_urls": ", ".join(base_urls)
        })

        # Check for user correction in history
        user_correction = None
        if history != "No previous conversation.":
            last_entry = history.split("\n\n")[-1]
            if "No, " in last_entry and "User: " in last_entry:
                user_correction = last_entry.split("User: ")[-1].strip()

        response, action, original_action = await enhance_response(user_query, agent_response, filtered_history, state, current_topic, user_correction)

        if response is None:
            raise ValueError("Response generation failed unexpectedly")

        ground_truth = test_data.get(user_query)
        benchmark_response = benchmark_data.get(user_query)
        metrics = calculate_metrics(user_query, response, current_topic, history, ground_truth=ground_truth)
        
        if "not able to answer" not in response.lower() and "error" not in response.lower():
            performance_metrics["resolutions"] += 1

        if ground_truth:
            logger.info(f"Test Mode - Query: '{user_query}'")
            logger.info(f"Generated Response: '{response}'")
            logger.info(f"Ground Truth: '{ground_truth}'")
            logger.info(f"Benchmark Response (Gemini): '{benchmark_response}'")
            logger.info(f"Metrics: Accuracy={metrics['accuracy']:.2f}, Coherence={metrics['coherence']:.2f}, Satisfaction={metrics['satisfaction']:.2f}")
            if benchmark_response:
                benchmark_metrics = calculate_metrics(user_query, benchmark_response, current_topic, history, ground_truth=ground_truth)
                logger.info(f"Benchmark Metrics: Accuracy={benchmark_metrics['accuracy']:.2f}, Coherence={benchmark_metrics['coherence']:.2f}, Satisfaction={benchmark_metrics['satisfaction']:.2f}")

        follow_up_question = generate_follow_up_question(user_query, response, current_topic)
        doc_id = store_interaction(user_id, user_query, response, agent_name, state, action, metrics, current_topic, follow_up_question)
        
        return {
            "agent": agent_name,
            "response": f"{response}\n\nWould you like to know more? How about: '{follow_up_question}'",
            "metrics": metrics,
            "benchmark_comparison": benchmark_comparison(metrics),
            "doc_id": doc_id,
            "topic": current_topic,
            "follow_up_question": follow_up_question
        }
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"Error in chat endpoint: {error_msg}. Query: {user_query}, Action attempted: {action if 'action' in locals() else 'unknown'}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/feedback/")
async def submit_feedback(feedback: Feedback):
    try:
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(ids=[feedback.doc_id])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail="Interaction not found")
        metadata = results["metadatas"][0]
        response_text = metadata.get("response", results["documents"][0])
        rating = feedback.rating
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be 1-5")
        satisfaction = (rating - 1) / 4.0
        metadata["satisfaction"] = float(satisfaction)
        collection.update(ids=[feedback.doc_id], metadatas=[metadata])
        state = metadata["state"]
        action = metadata["action"]
        reward = {1: -1, 2: -0.5, 3: 0, 4: 0.5, 5: 1}[rating]
        update_q_value(state, action, reward)
        logger.info(f"Feedback stored for doc_id: {feedback.doc_id}, rating: {rating}")
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")

# Static model metrics
static_model_metrics = {
    "accuracy": 0.0,
    "coherence": 0.0,
    "satisfaction": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0
}

def compute_static_model_metrics():
    global static_model_metrics
    accuracies, coherences, satisfactions = [], [], []
    y_true = []  # Initialize as separate empty list
    y_pred = []  # Initialize as separate empty list
    relevance_threshold = 0.65
    
    for query, bot_response in test_data.items():
        benchmark_response = benchmark_data.get(query, "")
        if not benchmark_response:
            logger.warning(f"No benchmark response for query: '{query}'")
            continue
        
        try:
            bot_embedding = model.encode([bot_response])[0]
            benchmark_embedding = model.encode([benchmark_response])[0]
            query_embedding = model.encode([query])[0]
            
            accuracy = max(0, cosine_similarity([bot_embedding], [benchmark_embedding])[0][0])
            accuracies.append(accuracy)
            
            coherence = max(0, cosine_similarity([bot_embedding], [query_embedding])[0][0])
            coherences.append(coherence)
            
            relevance = is_response_relevant(query, bot_response, "")
            sentiment = sid.polarity_scores(bot_response)['compound']
            satisfaction = min(1.0, max(0.0, 0.8 * relevance + 0.2 * sentiment))
            satisfactions.append(satisfaction)
            
            true_label = 1 if cosine_similarity([query_embedding], [benchmark_embedding])[0][0] >= relevance_threshold else 0
            pred_label = 1 if cosine_similarity([query_embedding], [bot_embedding])[0][0] >= relevance_threshold else 0
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            logger.debug(f"Query: '{query}' - True: {true_label}, Pred: {pred_label}, Accuracy: {accuracy:.2f}")
        
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            continue
    
    if accuracies:
        static_model_metrics["accuracy"] = np.mean(accuracies).item()
    if coherences:
        static_model_metrics["coherence"] = np.mean(coherences).item()
    if satisfactions:
        static_model_metrics["satisfaction"] = np.mean(satisfactions).item()
    
    if y_true and y_pred:
        static_model_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        static_model_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        static_model_metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    else:
        logger.warning("No valid data for precision, recall, F1 calculation")
    
    logger.info(f"Static model metrics computed: {static_model_metrics}")

compute_static_model_metrics()

def compute_evaluation_metrics():
    total_queries = performance_metrics["total_conversations"]
    avg_response_time = np.mean(performance_metrics["response_times"]).item() if performance_metrics["response_times"] else 0.0
    resolution_rate = (performance_metrics["resolutions"] / total_queries) if total_queries > 0 else 0.0
    fallback_rate = (performance_metrics["fallbacks"] / total_queries) if total_queries > 0 else 0.0
    avg_session_length = np.mean(performance_metrics["session_lengths"]).item() if performance_metrics["session_lengths"] else 0.0
    
    return {
        "accuracy": static_model_metrics["accuracy"],
        "coherence": static_model_metrics["coherence"],
        "satisfaction": static_model_metrics["satisfaction"],
        "precision": static_model_metrics["precision"],
        "recall": static_model_metrics["recall"],
        "f1_score": static_model_metrics["f1_score"],
        "avg_response_time": avg_response_time,
        "resolution_rate": resolution_rate,
        "fallback_rate": fallback_rate,
        "total_conversations": total_queries,
        "active_users": len(performance_metrics["active_users"]),
        "avg_session_length": avg_session_length
    }

@app.get("/evaluation_metrics/")
async def get_evaluation_metrics():
    try:
        eval_metrics = compute_evaluation_metrics()
        logger.info(f"Evaluation Metrics: {eval_metrics}")
        return eval_metrics
    except Exception as e:
        logger.error(f"Error computing evaluation metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {str(e)}")