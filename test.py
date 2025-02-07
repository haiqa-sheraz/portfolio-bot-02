import requests
import json

url = "https://portfolio-bot-6kls.onrender.com/chat"  # Replace with your actual URL

# Define the payload with the question
data = {
    "question": "Tell me about her project Skintelligent"
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
