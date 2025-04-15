
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
import json
import os

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
epsilon = 0.1  # Retained for potential future use
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
q_table = defaultdict(lambda: {"use_agent": 0.0, "use_gemini": 0.0, "use_hf": 0.0})
question_history = defaultdict(lambda: {"count": 0, "responses": []})
query_counter = 0

# Reward memory file
REWARD_FILE = "reward_memory.json"
if os.path.exists(REWARD_FILE):
    with open(REWARD_FILE, "r") as f:
        reward_memory = json.load(f)
else:
    reward_memory = {}

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
    "https://www.concordia.ca/gradstudies/students/new.html",
    "https://www.concordia.ca/students/exams/schedule.html",
    "https://www.concordia.ca/students/exams/important-dates.html"
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
    """You are a general-purpose assistant focused solely on general domain, excluding AI or admissions-related content unless explicitly asked. Ignore any prior history that suggests a different topic than current query. Conversation history: {history} \n
    Answer clearly and concisely the given query : {query}
    If you are unsure, lack relevant data, or cannot answer or don't have answer respond with 'I am not able to answer this question'."""
)

ai_agent = create_agent(
    """You are an AI expert specializing only in artificial intelligence topics. Ignore any prior history unrelated to AI. Conversation history: {history} \n
    Provide a detailed, accurate answer about AI-related topics: {query}.
    If you are unsure, lack relevant data, or cannot answer or don't have answer respond with 'I am not able to answer this question'."""
)

admission_agent = create_agent(
    """You are a Concordia University admissions advisor specializing only in admissions and program details like {admission_keywords}. Use {base_urls} for accurate info. Ignore any prior context unrelated to admissions. Conversation history: {history} \n
    Answer accurately about admissions and program details: {query}
    If you are unsure, lack relevant data, or cannot answer or don't have answer respond with 'I am not able to answer this question'."""
)


# ai_agent = create_agent(
# """As an AI specialist, please help with the following query.

#     Core AI topics for reference:
#     {ai_keywords}

#     Recent chat history:
#     {history}

#     User query: {query}

#     Please provide a clear, technical, and helpful response."""
# )

# general_agent = create_agent(
# """As a knowledgeable General assistant, please help with the following query:
            
#     Previous conversation:
#     {history}

#     User query: {query}

#     Please provide a helpful and informative response.
# """
# )

# admission_agent = create_agent(
# """As a Concordia University Computer Science admission specialist, please help with the following query.

#     Use the following context to provide accurate and relevant answers:

#     Recent chat history:
#     {history}

#     refer the urls for more information:
#     {base_urls}

#     User query: {query}

#     Please guide the prospective student based on the admission requirements.
#     """
# )



class Query(BaseModel):
    user_id: str
    query: str

class Feedback(BaseModel):
    doc_id: str
    rating: int  # 1-5 scale

def get_state(query, topic):
    global query_counter
    doc = nlp(query)
    entities = len(doc.ents)
    query_counter += 1
    return f"{topic}_{entities}_{query_counter}"

async def fetch_gemini_response(query, user_id):
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        history=retrieve_history(user_id, 5)
        new_query="give accurate response of the query based on your knowledge : \n"+query +"\n use below Recent chat history only for refrence : \n"+history+"\n"
        response = gemini_model.generate_content(new_query)
        text = response.text.strip()
        enhanced_response = f"{text}"
        logger.info(enhanced_response)
        return enhanced_response if is_response_relevant(query, enhanced_response, "") >= 0.55 else "I am not able to answer this question using Gemini."
    except Exception as e:
        logger.error(f"error: {str(e)}")
        return "I am not able to answer this question due to a Gemini error."

async def fetch_google_search_response(query):
    """
    Fetches a response from Google Search using the Custom Search JSON API.
    Returns the snippet of the first relevant result or a fallback message.
    """
    api_key = "AIzaSyBsEjEpJwJXUYcmJcEmgRwLvzfnMYg2n5A"  # Replace with your Google API key
    cse_id = "b418152f819ee4c59"  # Replace with your Custom Search Engine ID
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&num=1"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "items" in data and len(data["items"]) > 0:
                        snippet = data["items"][0]["snippet"]
                        # Clean up snippet (remove dates, extra whitespace, etc.)
                        cleaned_response = " ".join(snippet.split())
                        logger.info(f"Google Search response for '{query}': {cleaned_response}")
                        return cleaned_response
                    else:
                        logger.warning(f"No Google Search results for query: {query}")
                        return "Could not find a response from Google Search."
                else:
                    logger.error(f"Google Search API error: Status {resp.status}")
                    return "Could not find a response from Google Search."
    except Exception as e:
        logger.error(f"Error fetching Google Search response for '{query}': {str(e)}")
        return "Could not find a response from Google Search."

# Example usage (for testing)
async def test_google_search():
    response = await fetch_google_search_response("What is artificial intelligence?")
    print(response)

async def fetch_wikipedia_response(query):
    """
    Fetches a summary from Wikipedia for the given query.
    Returns a relevant summary or a fallback message if no page is found.
    """
    try:
        wikipedia.set_lang("en")
        
        # Modify query to target Concordia University’s main page for broader context
        base_query = "Concordia University"
        loop = asyncio.get_event_loop()
        
        # Search for relevant pages
        search_results = await loop.run_in_executor(None, lambda: wikipedia.search(query))
        if not search_results:
            logger.warning(f"No Wikipedia search results for query: {query}")
            return "Could not find a response from Wikipedia."
        
        # Use the original query if it’s specific, else default to "Concordia University"
        wiki_query = query if "concordia" in query.lower() and "schedule" in query.lower() else base_query
        summary = await loop.run_in_executor(None, lambda: wikipedia.summary(wiki_query, sentences=5))
        
        if summary:
            # Check if the summary is relevant to exams or schedules
            if "exam" in summary.lower() or "schedule" in summary.lower():
                logger.info(f"Wikipedia response for '{query}': {summary}")
                return summary
            else:
                logger.info(f"Wikipedia summary for '{query}' not exam-specific: {summary}")
                return f"Wikipedia info: {summary} (Note: Specific exam schedules may not be available here; check Concordia’s official site.)"
        else:
            logger.warning(f"No Wikipedia summary for query: {query}")
            return "Could not find a response from Wikipedia."
    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(f"Wikipedia disambiguation error for '{query}': {str(e)}")
        return "Multiple Wikipedia entries found, please be more specific."
    except wikipedia.exceptions.PageError:
        logger.warning(f"Wikipedia page not found for query: {query}")
        return "Could not find a response from Wikipedia."
    except Exception as e:
        logger.error(f"Error fetching Wikipedia response for '{query}': {str(e)}")
        return "Could not find a response from Wikipedia."

# Example usage (for testing)
async def test_wikipedia():
    response = await fetch_wikipedia_response("Artificial intelligence")
    print(response)
    
def is_response_relevant(query, response, history=""):
    try:
        if not response or not response.strip():
            return 0.0
        score = cross_encoder.predict([[query, response]])[0]
        if history and history != "No previous conversation.":
            history_score = cross_encoder.predict([[history, response]])[0]
            score = 0.9 * score + 0.1 * history_score
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
    
    # Retrieve raw history from ChromaDB
    try:
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        history=retrieve_history(user_id)
        # Build conversation turns with topic filtering
        conversation_turns = []
        for doc, meta in zip(documents[-10:], metadatas[-10:]):  # Limit to last 10 turns, consistent with retrieve_history
            turn = [
                f"User: {doc}",
                f"{meta['agent']}: {meta['response']}",
                f"Topic: {meta['topic']}"
            ]
            conversation_turns.append("\n".join(turn))
        
        # Filter history based on detected topic
        if topic == "Admissions" or "concordia" in query.lower():  # Note: Updated "Admission" to "Admissions" for consistency
            filtered_turns = [turn for turn in conversation_turns if "Topic: Admissions" in turn]
            filtered_history = "\n\n".join(filtered_turns) or "No relevant previous conversation."
            return admission_agent, "Admissions Agent", history
        elif topic == "AI":
            filtered_turns = [turn for turn in conversation_turns if "Topic: AI" in turn]
            filtered_history = "\n\n".join(filtered_turns) or "No relevant previous conversation."
            return ai_agent, "AI Agent", history
        else:  # General
            filtered_turns = [turn for turn in conversation_turns if "Topic: General" in turn]
            filtered_history = "\n\n".join(filtered_turns) or "No relevant previous conversation."
            return general_agent, "General Agent", history
    
    except Exception as e:
        logger.error(f"Error retrieving or filtering history: {str(e)}")
        return general_agent, "General Agent", "No previous conversation."  # Default to general agent on error

def detect_topic_with_ollama(query, user_id):
    if ollama_llm is None:
        return "General"
    try:
        history = retrieve_last_queries(user_id, n=5)
        prompt = PromptTemplate(
            input_variables=["query", "history", "ai_keywords", "admission_keywords", "general_keywords"],
            # template=""" You are the expert in classifing the user queries into AI, Concordia/Admission related or General domain related
            # Understand full user query and classify the user query into one of these categories more accurately. IF you dont understand the query or the query use keywords like it, this, that, above or previous. Than refer the Recent Chat History and return the last history topic: 

            # Recent Chat History: 
            # {history} 

            # Query: 
            # {query}
            
            # Instructions:
            # 1. AI related - give 'AI'
            # 2. Concordia University related or Admission related or concordia Student related - give 'Admission'
            # 3. General domain except AI-related or Concordia university, student or admission related then- give 'General' 

            # Provide your answer in one word format only: 'AI' or 'Admission' or 'General' """
           template=""" You are the expert in classifing the user queries into AI, Concordia/Admission related or General domain related
            Understand the following user query and classify the user query into one of these categories more accurately:

            Query: 
            {query}

            
            Instructions:
            1. AI related - give 'AI'
            2. Concordia University related or Admission related or concordia Student related - give 'Admission'
            3. If it is in continous to the previous topic like query used the keywords in queries like 'it', 'this', 'that' , 'above' or 'previous'. Than only refer the below recent chat history to understand context and return the Topic of the last chat history {history}. Other retur the Topic based on the Query itself.
            4. General domain except AI-related or Concordia university, student or admission related then- give 'General' 
           
            


            Provide your answer in one word format only: 'AI' or 'Admission' or 'General' """
         )
        chain = LLMChain(llm=ollama_llm, prompt=prompt)
        response = chain.invoke({
            "query": query.lower(),
            "history": history,
            "ai_keywords": ", ".join(ai_keywords),
            "admission_keywords": ", ".join(admission_keywords),
            "general_keywords": ", ".join(general_keywords)
        }).get('text', 'General').strip()
        valid_topics = ['AI', 'Admission', 'General']
        if "AI" in response:
            return "AI"
        elif "Admission" in response:
            return "Admission"
        elif "General" in response:
            return "General"
        else:
            logger.warning(f"Invalid topic response '{response}' from LLM, defaulting to 'General'")
            return "General"
    except Exception as e:
        logger.error(f"Error detecting topic with Ollama: {str(e)}")
        return "General"

def retrieve_history(user_id, n=20):
    try:
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        conversation_turns = []
        for doc, meta in zip(documents[-n:], metadatas[-n:]):
            conversation_turns.append(f"User: {doc}")
            conversation_turns.append(f"{meta['agent']}: {meta['response']}")
            conversation_turns.append(f"Topic: {meta['topic']}")
        context = "\n".join(conversation_turns)
        return context
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return "No previous conversation."

def retrieve_last_queries(user_id, n=20):
    try:
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        conversation_turns = []
        for doc, meta in zip(documents[-n:], metadatas[-n:]):
            conversation_turns.append(f"User: {doc}")
            conversation_turns.append(f"{meta['agent']}: {meta['response']}")
            conversation_turns.append(f"Topic: {meta['topic']}")
        context = "\n".join(conversation_turns)
        return context
    except Exception as e:
        logger.error(f"Error retrieving last queries: {str(e)}")
        return "No previous queries."

def store_interaction(user_id, query, response, agent_name, state, action, metrics, topic, follow_up, reward=0.0):
    collection = chroma_client.get_or_create_collection(name="chat_history")
    doc_id = str(uuid.uuid4())
    try:
        collection.add(
            ids=[doc_id],
            documents=[str(query)],
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
                "reward": float(reward),  # Store reward (default 0 until feedback)
                "topic": topic,
                "follow_up": follow_up,
                "timestamp": datetime.now().isoformat()
            }]
        )
        logger.info(f"Interaction stored successfully with doc_id: {doc_id}")

        # Log interaction to evaluation_log.txt with truncation handling
        log_path = "evaluation_log.txt"
        if not os.path.exists(log_path) or os.stat(log_path).st_size == 0:
            with open(log_path, "w") as f:
                f.write(
                    "=== Concordia Multi-Agent Chatbot Evaluation Log ===\n"
                    "Each entry includes the timestamp, user input, selected agent, response, and evaluation metrics.\n\n"
                    "Metric Descriptions:\n"
                    "- Accuracy: Measures how well the response includes expected keywords relevant to the topic.\n"
                    "- Coherence: Measures how well the response connects with the user’s recent conversation history.\n"
                    "- Reward: Calculated from user feedback (1-5 scale mapped to -1 to 1) or default 0 if no feedback.\n"
                    "------------------------------------------------------------\n\n"
                )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_response = str(response).replace("\n", " ").strip()
        max_log_length = 200  # Max characters for response in log
        if len(clean_response) > max_log_length:
            clean_response = clean_response[:max_log_length] + " [TRUNCATED]"
        log_entry = (
            f"Timestamp   : {timestamp}\n"
            f"User ID     : {user_id}\n"
            f"Agent       : {agent_name}\n"
            f"User Query  : {query}\n"
            f"Response    : {clean_response}\n"
            f"Accuracy    : {metrics['accuracy']}\n"
            f"Coherence   : {metrics['coherence']}\n"
            f"Reward      : {reward}\n"
            "------------------------------------------------------------\n\n"
        )
        with open(log_path, "a") as f:
            f.write(log_entry)

        return doc_id
    except Exception as e:
        logger.error(f"Failed to store interaction: {str(e)}")
        raise

def check_prior_feedback(query):
    query_key = query.strip().lower()
    if query_key in reward_memory:
        ratings = reward_memory[query_key]
        avg_rating = sum(ratings) / len(ratings)
        return avg_rating
    return None

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

def compute_reward(rating):
    # Map 1-5 rating to reward between -1 and 1
    reward_map = {1: -1, 2: -0.5, 3: 0, 4: 0.5, 5: 1}
    return reward_map.get(rating, 0)  # Default to 0 if invalid rating

def update_q_value(state, action, reward):
    old_q = q_table[state][action]
    q_table[state][action] = (1 - alpha) * old_q + alpha * (reward + gamma * max(q_table[state].values()))
    logger.info(f"Updated Q-value for state '{state}', action '{action}': {q_table[state][action]}")

async def enhance_response(query, agent, agent_name, agent_response, history, state, topic, user_id):
    # Define negative keywords for response evaluation
    negative_keywords = [
        "not able to answer"
    ]
    
    # Check prior feedback from reward_memory.json
    prior_rating = check_prior_feedback(query)
    action = "use_agent"
    agent_response_text = str(agent_response.get('text', agent_response) if isinstance(agent_response, dict) else agent_response).lower()
    has_negative_keywords = any(keyword in agent_response_text for keyword in negative_keywords)

    # Check for cached similar responses using embeddings
    collection = chroma_client.get_or_create_collection(name="chat_history")
    query_embedding = model.encode([query])[0]
    results = collection.get(where={"user_id": user_id})
    if results["ids"]:
        for idx, doc in enumerate(results["documents"]):
            doc_embedding = model.encode([doc])[0]
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            if similarity > 0.9:  # High similarity threshold
                metadata = results["metadatas"][idx]
                if metadata["reward"] >= 0.5:  # Use if prior feedback is positive
                    logger.info(f"Reusing cached response for similar query: {doc}")
                    return metadata["response"], action, action, results["ids"][idx]

    # If prior rating >= 4, use cached response
    if prior_rating and prior_rating >= 4:
        results = collection.get(where={"query": query, "user_id": user_id})
        if results["ids"]:
            metadata = results["metadatas"][-1]  # Most recent match
            return metadata["response"], action, action, results["ids"][-1]

    # Check Q-table for action preference (used later if needed)
    q_values = q_table[state]
    use_fallback = prior_rating is not None and prior_rating <= 2  # Only regenerate if rating exists and is low

    if use_fallback or has_negative_keywords:  # Fallback if low rating or negative response
        if use_fallback:
            logger.info(f"Initial response low-rated ({prior_rating}): {agent_response_text}. Attempting regeneration.")
        else:
            logger.info(f"Initial response negative: {agent_response_text}. Attempting fallback.")
        
        # Retry with Ollama agent using a modified prompt (only for low rating)
        if use_fallback:
            retry_prompt = f"Please provide a more detailed or different explanation for: {query}."
            agent_response_retry = agent.invoke({
                "history": history,
                "query": retry_prompt,
                "ai_keywords": ", ".join(ai_keywords),
                "general_keywords": ", ".join(general_keywords),
                "admission_keywords": ", ".join(admission_keywords),
                "base_urls": ", ".join(base_urls)
            })
            retry_text = str(agent_response_retry.get('text', agent_response_retry) if isinstance(agent_response_retry, dict) else agent_response_retry).lower()
            
            if is_response_relevant(query,retry_text)>0.5:
                response = retry_text
                action = "use_agent"
                logger.info(f"Regenerated Ollama response successful: {response}")
                # Calculate metrics and store the interaction
            else:
                logger.info(f"Regenerated Ollama response still negative: {retry_text}. Trying Gemini.")
        else:
            retry_text = None  # Skip Ollama retry if only negative keywords triggered
        
        if has_negative_keywords or retry_text==None:
            # Always try Gemini first
            gemini_resp = await fetch_gemini_response(query, user_id)
            gemini_resp_lower = gemini_resp.lower()
            if is_response_relevant(query,gemini_resp_lower)>0.55:
                response = gemini_resp
                action = "use_gemini"
                logger.info(f"Using Gemini response: {response}")
            else:
                # Try Google Search if Gemini fails
                google_resp = await fetch_google_search_response(query)
                google_resp_lower = google_resp.lower()
                if is_response_relevant(query,google_resp_lower)>0.55:
                    response = google_resp
                    action = "use_google"
                    logger.info(f"Using Google Search response: {response}")
                else:
                    # Try Wikipedia if Google fails
                    wiki_resp = await fetch_wikipedia_response(query)
                    wiki_resp_lower = wiki_resp.lower()
                    if is_response_relevant(query,wiki_resp_lower)>0.55:
                        response = wiki_resp
                        action = "use_wikipedia"
                        logger.info(f"Using Wikipedia response: {response}")
                    else:
                        response = "I am not able to provide a satisfactory answer to this question. Please try rephrasing or asking something else!"
                        action = "use_agent"
                        logger.info(f"All attempts failed (Gemini: {gemini_resp_lower}, Google: {google_resp_lower}, Wiki: {wiki_resp_lower}). Final response: {response}")
        
    else:
        # Use initial agent response if no rating exists or rating > 2, and no negative keywords
        response = agent_response_text
        logger.info(f"Using initial agent response: {response}")

    # Calculate metrics and store the interaction
    question_history[query]["responses"].append(response)
    metrics = calculate_metrics(query, response, topic, history)
    follow_up = ""  # Placeholder, updated later
    doc_id = store_interaction(user_id, query, response, agent_name, state, action, metrics, topic, follow_up, reward=0.0)

    logger.info(f"Final response: {response}, Action: {action}")
    return response, action, action, doc_id

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

        response, action, original_action, doc_id = await enhance_response(user_query, agent, agent_name, agent_response, filtered_history, state, current_topic, user_id)

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
        
        # Update interaction with user_id, agent_name, and follow_up
        collection = chroma_client.get_or_create_collection(name="chat_history")
        results = collection.get(ids=[doc_id])
        if results["ids"]:
            metadata = results["metadatas"][0]
            metadata["user_id"] = user_id
            metadata["agent"] = agent_name
            metadata["follow_up"] = follow_up_question
            collection.update(ids=[doc_id], metadatas=[metadata])

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
        query = metadata.get("query", "Unknown")
        response_text = metadata.get("response", results["documents"][0])
        rating = feedback.rating
        
        if not 1 <= rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Calculate reward from rating (1-5 scale mapped to -1 to 1)
        reward = compute_reward(rating)
        
        # Update ChromaDB with new reward
        metadata["reward"] = float(reward)
        collection.update(ids=[feedback.doc_id], metadatas=[metadata])
        
        # Update reward_memory.json
        query_key = query.strip().lower()
        if query_key not in reward_memory:
            reward_memory[query_key] = []
        reward_memory[query_key].append(rating)
        with open(REWARD_FILE, "w") as f:
            json.dump(reward_memory, f, indent=2)
        
        # Update Q-value
        state = metadata["state"]
        action = metadata["action"]
        update_q_value(state, action, reward)
        
        # Log feedback to feedback_log.txt
        feedback_log = "feedback_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(feedback_log, "a") as f:
            f.write(
                f"Timestamp: {timestamp}\n"
                f"Doc ID: {feedback.doc_id}\n"
                f"User ID: {metadata['user_id']}\n"
                f"Query: {query}\n"
                f"Rating (1-5): {rating}\n"
                f"Reward: {reward}\n"
                "------------------------------------------------------------\n\n"
            )
        
        logger.info(f"Feedback stored for doc_id: {feedback.doc_id}, rating: {rating}, reward: {reward}")
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
    y_true = []
    y_pred = []
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
