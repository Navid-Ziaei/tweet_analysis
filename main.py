import uvicorn
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import pandas as pd
from io import StringIO
from pydantic import BaseModel

from transformers import AutoTokenizer

from src.model.spicy_model.model import predict_label_spacy
from src.model.supervised_models.transformer_base_models import load_model_and_predict
from src.model.supervised_models.xgboost_test_classifier import XGBoostTextClassifier
from src.model.transformer_base_tagging.model_pos_tag import predict_label_bert

app = FastAPI()


def evaluate(texts, model, model_path):
    if model == 'xgboost':
        text_classifier = XGBoostTextClassifier(data=texts, labels=None, mode='weight')
        output, predicted_labels = text_classifier.load_and_predict(model_path, texts)
    elif model == 'spacy_ner':
        output, predicted_labels = predict_label_spacy(texts, model_name="en_core_web_trf")
    elif model == 'bert_ner':
        output, predicted_labels = predict_label_bert(texts, model_name="dslim/bert-base-NER-uncased")
    elif model == 'transformer':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        output, predicted_labels = load_model_and_predict(model_path, tokenizer, texts)
    return output, predicted_labels

class PredictionRequest(BaseModel):
    text: str
    type_name: str
    model_path: str


@app.get("/")  # This decorator is used to define the path operation for the root URL
async def read_root():
    return {"Message": "Welcome to the Tweet Analyzer API!"}


# Endpoint for text input with model specification
@app.post("/predict/")
async def create_prediction(request: PredictionRequest):
    try:
        # Assuming you have a function `evaluate` that uses these parameters
        result, _ = evaluate([request.text], request.type_name, request.model_path)
        return result[0]  # Assuming `evaluate` returns a list where each item is a dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




