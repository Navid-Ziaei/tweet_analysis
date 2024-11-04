# Tweet Analysis

Welcome to the GitHub repository for our Ride Conversion Predictor (RCP) models! 
The primary objective of this project is to develop a predictive model that forecasts the 
likelihood of a ride request being converted into a completed ride. This project involves conducting a 
comprehensive exploratory data analysis (EDA) to understand the dataset, training and validating the 
predictive model and performing an error analysis to evaluate its performance. 
<br/>

## Table of Contents
* [General Information](#general-information)
* [Getting Started](#getting-started)
* [Settings Configuration](#settings-configuration)
* [Prepare Data](#prepare-data)
* [Repository Structure](#repository-structure)
* [Contributing](#contributing)
* [License](#license)
<br/>

## General Information
In the dynamic and fast-paced environment of Snapp-Cab, efficiently managing and analyzing vast amounts of data is critical. With over four million rides conducted daily, the data generated contains numerous patterns and relationships that, if uncovered, can significantly enhance business value and improve user experience for passengers and drivers. The ability to predict whether a ride request will convert into an actual ride is one such challenge that can provide valuable insights to optimize operations and service delivery.

## Getting Started

1. Clone this repository to your local machine.

2. Install the required dependencies. `pip install -r requirements.txt`

3. Prepare your dataset

4. Create the `./configs/settings.yaml` according to `./cinfigs/settings_sample.yaml`

5. Create the `./configs/device_path.yaml` according to `./cinfigs/device_path_sample.yaml`

6. Run the `main.py` script to execute the model.



## Prepare Data
1. To load data, you need two folder:
   - preprocessed data folder : the model save preprocessed datato this folder
   - data folder : this folder should contain `train.csv` and `test.csv`
2. Set the path to this folder in `device_path.yaml` as `raw_dataset_path` and `preprocessed_dataset_path`

## Settings Configuration

The `settings.yml` file is a crucial configuration file used to control various aspects of the data preprocessing, model training, and evaluation processes. It enables users to customize the behavior of the code based on their requirements. Below is a detailed explanation of the settings provided in the `settings.yml`:


- model:
  - `xgboost`
  - `logistic_regression`
  - `tabnet`
  - `ft_transformer`
  - `category_embedding`
  - `gandalf`
  This section specifies the machine learning models to be used. even one of them can be used
- `impute_strategy`: The strategy for handling missing values. `mean` indicates that missing values will be filled with the mean of the respective feature.
- `missing_threshold`: The proportion of missing values in a feature above which the feature will be removed. A value of 0.7 means features with more than 70% missing values will be discarded.
- `remove_out_layers`: Indicates whether to remove outliers from the dataset. Set to False, so outliers will not be removed.
- `scaling_mode`: Defines the method for feature scaling. `normalization` means features will be scaled to a standard range, typically [0, 1].
- `sampling_method`: Specify the balancing method from:
  - `SMOTE`
  - `RandomOverSampler`
  - `RandomUnderSampler`
  - `None`
- `load_trained_model`: Specifies whether to load a previously trained model. False indicates that a new model should be trained from scratch.
- `batch_size`: The number of samples per batch during training. Set to 100, meaning each training iteration will process 100 samples.
- `num_epochs`: The number of epochs (iterations over the entire dataset) for training. Set to 100, which means the model will train for 100 epochs.
- `test_size`: The proportion of the dataset to be used for testing. A value of 0.2 means 20% of the data will be allocated to the test set.
= `debug_mode`: if it is false, it save results in debug folder, otherwise it each time it creates a folder with timestamp and save results to it.

## Run using Docker
To simplify the setup and execution of the Ride Conversion Predictor project, you can use Docker. Docker provides a consistent environment across different machines, ensuring that all dependencies and configurations are correctly set up. Follow the steps below to run the project using Docker:

1. **Install Docker**: 
   Ensure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Clone the Repository**:
   Clone the repository to your local machine if you haven’t already:
   ```bash
   git clone https://github.com/Navid-Ziaei/Ride-Conversion-Predictor.git
   cd Ride-Conversion-Predictor
   ```
3. **Build the Docker Image**:
   Navigate to the project directory and build the Docker image using the provided Dockerfile. This process installs all necessary dependencies and sets up the environment.
   ```bash
   docker build -t ride-conversion-predictor .
   ```
4. Run the Docker Container:
Once the image is built, you can run the Docker container. This command will start the container and execute the main.py script inside it.
   ```bash
   docker run --gpus all -it 
   -v D:/path_to_dataset/data:/ride_conversion_predictor/data 
   -v D:/path_to_result_folder/results:/ride_conversion_predictor/results ride_conversion_predictor

   ```
5. Check Results:
After running the container, results and logs will be saved to the results/ directory in your project folder. Check this folder for output files and evaluation metrics.

## Repository Structure
This repository is organized as follows:

- `/main.py`: The main script to run.

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