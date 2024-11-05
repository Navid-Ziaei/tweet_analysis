from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import pandas as pd
from io import StringIO
import uvicorn
from transformers import AutoTokenizer

from src.model.spicy_model.model import predict_label_spacy
from src.model.supervised_models.transformer_base_models import load_model_and_predict
from src.model.supervised_models.xgboost_test_classifier import XGBoostTextClassifier
from src.model.transformer_base_tagging.model_pos_tag import predict_label_bert

app = FastAPI()


# Endpoint for text input with model specification
@app.get("/")  # This decorator is used to define the path operation
async def read_root():
    return {"Message": "Tweet Analyzer!"}


@app.post("/predict/")
async def create_prediction(text: str, type_name: str = Form(...), model_path: str = Form(...)):
    try:
        result, _ = evaluate([text], type_name, model_path)
        return result[0]  # Return the first and only element as this is a single text prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for CSV file input with model specification
@app.post("/predict_csv/")
async def create_prediction_from_csv(file: UploadFile = File(...), type_name: str = Form(...),
                                     model_path: str = Form(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    try:
        # Read the content of the file into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        df['prediction'] = df['text'].apply(lambda x: evaluate([x], type_name, model_path)[0]['label'])

        # Convert predictions to DataFrame for JSON response
        response_df = pd.DataFrame(df)
        response = response_df.to_json(orient="records")
        return {"predictions": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def evaluate(texts, model, model_path):
    # This function will handle the logic to predict using your models
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
