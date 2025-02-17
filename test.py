import requests
import json

url = "http://127.0.0.1:8000/chat"  # Replace with your actual URL

# Define the payload with the question
data = {
    "question": "tell me about the project on MS Teams"
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")
