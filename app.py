from fastapi import FastAPI, Request
import requests
import json
import os

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

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")

    CONTEXT = """
    Haiqa Sheraz Alhassan is a highly skilled AI developer with expertise in Retrieval Augmented Generation (RAG), Python programming, Machine Learning, Deep Learning, and Chatbot development, with proficiency in frameworks such as Langchain, LlamaIndex, TensorFlow, and PyTorch, along with data analysis skills using Tableau and Power BI. She is currently completing a BS in Artificial Intelligence from FAST-NUCES (2020-2024) and previously completed A Levels at Pakistan International School, Jeddah (2018-2020). She currently lives in Jeddah, Saudi Arabia. Her technical skills encompass OOP, Data Structures, Natural Language Processing, Computer Vision, model deployment, front-end development, and various programming languages including Python, C, C++, C#, HTML, CSS, PHP, and SQL. She has hands-on experience with tools like Microsoft Bot Framework, Postman, and Microsoft Cognitive Services. As an AI Solutions Engineer at Imperium Dynamics (July 2024 - December 2024), she developed a bot using Microsoft Bot Framework for Teams meeting transcript processing, worked on a product search system integrating Azure Blob Storage, and created a chat streaming application using GPT-4 with real-time text-to-speech conversion. During her internship at TheCoded (October 2023 - December 2023) as an Associate Consultant in Data & AI, she conducted research on Microsoft Copilot and developed an ML chatbot for medical applications. Her projects include "Skintelligent," an AI-powered skincare regime generator using LlamaIndex and Mistral-7B, a product visual search system integrating 50,000 image embeddings into Pinecone using CLIP, a PDF summarizer leveraging Lamini-Flan T5, a medical chatbot fine-tuned on the Nouse-Hermes Llama2 model, sentiment analysis on movie reviews achieving 98.35% accuracy with LSTM, an NFT marketplace website with MySQL integration, and a global terrorism report with Tableau visualizations. She has received accolades such as making the Dean’s List (GPA 3.81), being a runner-up in the AI-NEXUS FAST Computer Vision competition, earning a TensorFlow Developer Professional Certificate from DeepLearning.ai, and completing a Data Science and Machine Learning Bootcamp on Udemy. She has held leadership roles as Media and Promotions Head for the FAST Data Science Society and Media Content Co-Head for Procom-FAST. Give answer using the context provided and answer in 3 points. Limit your answer up to 50 words and answer only the question asked. No further info is required. Do not ask or answer any additional questions. If answer to a question is not provided in the context just say I do not know the answer to this question.
    """

    payload = {
        "inputs": f"Context: {CONTEXT}\n\n[INST] {question} [/INST]",
        "parameters": {"return_full_text": False}
    }

    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))

    if response.status_code == 200:
        generated_text = response.json()[0]["generated_text"]
        
        # Format the text to make it more readable
        formatted_text = " ".join(generated_text.split("\n"))  # Remove line breaks
        
        return {"answer": formatted_text.strip()}
    else:
        return {"error": f"Error {response.status_code}: {response.text}"}