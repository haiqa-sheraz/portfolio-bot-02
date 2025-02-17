from fastapi import FastAPI, Request
import requests
import json
import os
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is taken from the environment variable
API_KEY = os.getenv("HF_API_KEY")  # Reads from environment variable

# Hugging Face API URL
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# Hugging Face API Headers
if API_KEY is None:
    raise ValueError("API key for Hugging Face is missing. Please set the HF_API_KEY environment variable.")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"  # Using the API key from environment
}

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://haiqasherazai.vercel.app"# Add the URL of your frontend (e.g., Next.js app)
      # Alternatively, add your deployed frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows localhost & Vercel
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS)
    allow_headers=["*"],  # Allows all headers
)

@app.options("/chat")
async def preflight():
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")

    CONTEXT = """  
    - Name: Haiqa Sheraz Alhassan  
    - Age: 23 years (as of February 2025)  
    - Location: Jeddah, Saudi Arabia  
    
    Education:  
    - Bachelors in Artificial Intelligence – FAST-NUCES (Completed: June 2024)  
    - A Levels – Pakistan International School, Jeddah (2018-2020)  
    - Employment Status: Unemployed, seeking a remote job opportunity in AI or related field  

    Technical Skills:  
    - AI & Machine Learning: RAG, Machine Learning, Deep Learning, NLP, Computer Vision  
    - Programming Languages: Python, C, C++, C#, HTML, CSS, PHP, SQL  
    - Frameworks & Libraries: Langchain, LlamaIndex, TensorFlow, PyTorch  
    - Data Analysis & Visualization: Tableau, Power BI  
    - Software & Tools: Microsoft Bot Framework, Postman, Microsoft Cognitive Services, Blob storage, Docker, Azure AI, AzureOpenAI
    - Other Skills: OOP, Data Structures, Model Deployment, Front-End Development  

    Work Experience:  
    - AI Solutions Engineer – Imperium Dynamics (July 2024 - December 2024)  
    - Developed a Teams meeting transcript processing bot using Microsoft Bot Framework. It generates summary and action items for a meeting and sends them via email to all participants. 
    - Created a product search system integrating Azure Blob Storage  
    - Built a chat streaming application using GPT-4 with real-time text-to-speech conversion  

    - Associate Consultant (Data & AI) – TheCoded (Internship: October 2023 - December 2023)  
    - Researched Microsoft Copilot  
    - Developed a machine learning chatbot for medical applications  

    Projects:  
    - Skintelligent: AI Skincare Regime Generation System: Utilizes LlamaIndex, RAG techniques, and Vision Transformer in PyTorch to assess acne and wrinkles, feeding data into Mistral-7B LLM to generate personalized skincare routines.  
    - Product Visual Search System: Built a recommendation system for product and image suggestions, integrating 50,000 image embeddings into Pinecone using the CLIP model for retrieval. 
    - PDF Summarizer: Utilizing summarization pipeline of Lamini-Flan T5 with 248M parameters, used to generate precise summaries of any input PDF  
    - Medical Chatbot: Provides personalized medical advice, treatments, and precautions using the Nouse-Hermes Llama2 model fine-tuned on a medical responses dataset from HuggingFace.  
    - Sentiment Analysis on Movie Reviews: Categorizes reviews as positive, negative, or neutral using TF-IDF for representation, achieving 82.32% accuracy with Naive Bayes and 98.35% with LSTM in five epochs.  
    - NFT Marketplace Website: Created an interactive user interface with HTML and CSS, with MySQL database structure and PHP to securely connect the frontend and backend - buying, selling, and exploring NFTs  
    - Global Terrorism Report: Used visualization techniques including heatmaps, word-clouds, line graphs, and bar charts on Tableau  

    Achievements & Certifications:  
    - Dean’s List – GPA 3.81  
    - Runner-up in AI-NEXUS FAST Computer Vision Competition  
    - TensorFlow Developer Professional Certificate – DeepLearning.ai  
    - Data Science & Machine Learning Bootcamp – Udemy  

    Leadership & Extracurricular Activities:  
    - Media & Promotions Head – FAST Data Science Society  
    - Media Content Co-Head – Procom-FAST  

    """

    INSTRUCTIONS = """
    You are a bot developed to answer questions for me (Haiqa Sheraz Alhassan). I have designed you to answer questions about my professional life, education and career. Follow the guidelines mentioned below:
    - Answer questions only using the information provided in the CONTEXT.  
    - Keep responses concise and to the point.  
    - Strictly answer only the question asked. Do not ask or answer any additional questions beyond what is explicitly requested. Do not include extra details, follow-up questions, or unrelated information.
    - If any personal question is asked or asked about any person then say "I answer questions only on professional basis, not on personal life"
    - If asked about any language or concept or technology other than that which is mentioned in CONTEXT, then say "She does not know it but she is a fast-learner"
    - When asked about any project, then first give the information about it which is provided in CONTEXT, then say "For further information, kindly contact Haiqa via email"
    - Only when asked to state something special about me say: She is a fast learner. Before joining Imperium Dynamics, she had no experience working with cloud-based solutions, but she quickly learned to use Azure AI tools and deployment, demonstrating her eagerness to learn. Therefore, she is well-suited for new opportunities and job roles that require acquiring technical skills beyond her current knowledge.
    """

    payload = {
        "inputs": f"Context: {CONTEXT}\n\nInstructions: {INSTRUCTIONS}\n[INST] {question} [/INST]",
        "parameters": {
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        generated_text = response.json()[0]["generated_text"]
        
        # Format the text to make it more readable
        formatted_text = " ".join(generated_text.split("\n"))  # Remove line breaks
        
        return {"answer": formatted_text.strip()}
    else:
        return {"error": f"Error {response.status_code}: {response.text}"}