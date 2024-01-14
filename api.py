from fastapi import FastAPI, HTTPException
import pickle
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

app = FastAPI()

# Loading the Tfidf and the model 
tfidf = pickle.load(open('tf_idf.pkt', 'rb'))
lgb_model = pickle.load(open('toxicity_model.pkt', 'rb'))

@app.post('/predict')
async def predict(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    # Ensure that tfidf is a TfidfVectorizer object
    if not isinstance(tfidf, TfidfVectorizer):
        raise HTTPException(status_code=500, detail="Invalid TF-IDF vectorizer object")
    
    # Transform the input text tp Tfidf vectors
    text_tfidf = tfidf.transform([text]).toarray()

    # Predict the class of the input text
    prediction = lgb_model.predict(text_tfidf)

    # Map the predicted class to a string
    class_name = "Toxic" if prediction == 0 else "Non-Toxic"

    # Return the prediction in a JSON response
    return {
        "text": text, 
        "class": class_name
    }