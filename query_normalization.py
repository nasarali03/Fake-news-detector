import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def query_groq_llm(query: str, conversation_type: str = "news") -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    # Initialize the Groq LLM via LangChain
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.7,
        streaming=True,
    )
    
    if conversation_type == "casual":
        # Casual mode: introduce itself as VariNews
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are VariNews 📰, an AI assistant designed to detect fake news.  
            When users greet you or have casual conversation, respond politely.  
            Always introduce yourself as 'VariNews' in your first response.  
            
            - Be friendly, empathetic, and concise (under 100 words).  
            - Gently guide the user to provide a news headline or article for analysis.  
            - Example: 'Hi there! I'm VariNews 📰. I'm here to help you check if a news article is real or fake. 
              Could you please share a news headline or some text?'"""),
            ("user", "{query}")
        ])
    else:
        # News mode: reformulate user query into more descriptive article-like text
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are VariNews 📰, an AI assistant that helps detect fake news.  
            Take the user’s input (which may be a short headline or snippet)  
            and expand it into a short, news-style paragraph (around 100 words).  
            Keep the meaning intact, avoid adding opinions, and ensure it looks like a real news article.  
            """),
            ("user", "{query}")
        ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})
    return response

def query_groq_llm_casual(query: str) -> str:
    """Wrapper function specifically for casual conversations"""
    return query_groq_llm(query, conversation_type="casual")
