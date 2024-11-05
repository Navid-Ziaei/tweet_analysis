# Tweet Analysis

Welcome to the GitHub repository for our Tweet Analyzer (TA) models! 
The primary objective of this project is to distinguish tweets related to the "Across Protocol" from those using the word "across" in a general context. This involves exploratory data analysis (EDA), training and validating predictive models, and performing error analysis to evaluate model performance.

<br/>

## Table of Contents
* [General Information](#general-information)
* [Getting Started](#getting-started)
* [Settings Configuration](#settings-configuration)
* [Prepare Data](#prepare-data)
* [API Usage](#api-usage)
* [Making Predictions](#making-prediction)
* [Repository Structure](#repository-structure)
* [Contributing](#contributing)
* [License](#license)
<br/>

## General Information
This project leverages machine learning to analyze tweet data, identifying specific references to the "Across Protocol" blockchain technology among general uses of the word "across". By automating this differentiation, the project aims to enhance data analytics and provide insights relevant for market analysis and sentiment tracking.

## Getting Started

1. Clone this repository to your local machine.

2. Install the required dependencies. `pip install -r requirements.txt`

3. Prepare your dataset

4. Create the `./configs/settings.yaml` according to `./cinfigs/settings_sample.yaml`

5. Create the `./configs/device_path.yaml` according to `./cinfigs/device_path_sample.yaml`

6. Run the `train.py` script to train the model.

7. Run the `evaluate.py` script to evaluate the trained models.


## Prepare Data
- You need a csv file with two columns `text` and `is_related_to_Across_protocol` located in `raw_dataset_path` (Specified in `./cinfigs/device_path.yaml`)

## API Usage
### Run the API
Run the FastAPI application to interact with the models via HTTP requests:
```bash
python -m uvicorn main:app
```
## Making Predictions

### Text Prediction

To make a prediction for a single piece of text, use the following CURL command in your terminal:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
        "text": "Example tweet content here",
        "type_name": "transformer",
        "model_path": "./saved_models/"
      }'
```

- **URL:** `/predict/`
- **Method:** `POST`
- **Description:** Returns a prediction based on the input text, model type, and path to the model.
- **Body:**
  - `text` (string): The text to analyze.
  - `type_name` (string): The type of model to use (`xgboost`, `spacy_ner`, `bert_ner`, or `transformer`).
  - `model_path` (string): The file path to the pre-trained model.
- **Required Headers:**
  - `Content-Type`: `application/json`
- **Body Example:**

```json
{
  "text": "Sample tweet text goes here",
  "type_name": "transformer",
  "model_path": "/path/to/your/model"
}
```


## Settings Configuration

The `settings.yml` file is a crucial configuration file used to control various aspects of the data preprocessing, model training, and evaluation processes. It enables users to customize the behavior of the code based on their requirements. Below is a detailed explanation of the settings provided in the `settings.yml`:


- model:
  - `spicy NER`
  - `stanford NER`
  - `Bert NER`
  - `XGBoost`
  - `Transformer`
-  This section specifies the machine learning models to be used. even one of them can be used
- `Prrproceesing`: 
- `sampling_method`: Specify the balancing method from:
  - `RandomOverSampler`
  - `RandomUnderSampler`
  - `None`
- `batch_size`: The number of samples per batch during training. Set to 100, meaning each training iteration will process 100 samples.
- `num_epochs`: The number of epochs (iterations over the entire dataset) for training. Set to 100, which means the model will train for 100 epochs.
= `debug_mode`: if it is false, it save results in debug folder, otherwise it each time it creates a folder with timestamp and save results to it.


## Repository Structure
This repository is organized as follows:

- `/train.py`: The main script to train the model.

- `/evaluate.py`: The main script to evaluate the trained models.

- `/results/`: This folder is generated when you first run the model. The results are saved with time stamp if the `debug_mode: true` is in `settings.yaml`

- `/src/`: All codes are located in this folder

- `/src/data`: Contains scripts for data loading (`data_loader.py`, `data_preprocessor.py`) to load the data, analyze it and preprocess it.

- `/src/experiments/`: Contains the `explainatory_data_analysis` script for checking the EDA and play with data! This folder contains scripts for different experiments

- `/src/model/`: Contains the functions required to build, train and evaluate models.

- `/src/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `/src/utils`: Contains utility script `utils.py` for some helper function that are used in code.

- `/visualization`: Contains the `vizualize_utils.py` script for data and result visualization.
<br/>



## Contributing

We encourage you to this project! 

## License

This project is licensed under the terms of the MIT license.