from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from io import StringIO
import uvicorn

app = FastAPI()


# Example model function
def predict(text: str) -> dict:
    # This is a placeholder for your actual model prediction logic
    return {"text": text, "prediction": "positive"}


# Endpoint for text input
@app.post("/predict/")
async def create_prediction(text: str):
    try:
        result = predict(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for CSV file input
@app.post("/predict_csv/")
async def create_prediction_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    try:
        # Read the content of the file into a pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        predictions = df['text'].apply(predict)

        # You can now save the predictions to a CSV or just return them
        response = predictions.to_json(orient="records")
        return {"predictions": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
