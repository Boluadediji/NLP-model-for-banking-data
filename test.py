# import pickle

# vectorizer, model = pickle.load(open('bank_model.pkl', 'rb'))
# while True:
#     text = input('Enter text: ')

#     X = vectorizer.transform([text])
#     pred = model.predict(X)
#     if hasattr(model, 'predict_proba'):
#         probs = model.predict_proba(X).tolist()[0]
#         print(f'prediction:{pred}, \nprobs:{[f"{prob:.2f}" for prob in probs]}')


import requests

import os

from dotenv import load_dotenv

load_dotenv()

endpoint = "https://bank-live-a43757.southafricanorth.inference.ml.azure.com/score"
api_key = os.getenv("REQUEST_KEY")


def get_prediction(text):
    response = requests.post(endpoint, data='{"text": "' + text + '"}', headers={
        'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'})
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"
    

if __name__=="__main__":
    while True:
        user_input = input("Enter your banking question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        prediction = get_prediction(user_input)
        print(f"Prediction: {prediction}")
    


