# MLB Performance Predictor

This project is a personal endeavor to predict MLB player performance using machine learning techniques. Below is a detailed description of the structure of the project, including the important files, notebooks, and folders.

## Project Structure

### Files
- **`predict_2025.py`**  
    This script contains the main logic for predicting MLB player performance for the 2025 season. It uses trained machine learning models and processes input data to generate predictions.

- **`data_preprocessing.py`**  
    A utility script for cleaning and preparing raw data for analysis. It handles missing values, feature engineering, and data normalization.

- **`model_training.py`**  
    This script is responsible for training the machine learning models. It includes hyperparameter tuning and model evaluation.

- **`requirements.txt`**  
    A list of all Python dependencies required to run the project. Use `pip install -r requirements.txt` to install them.

### Notebooks
- **`exploratory_analysis.ipynb`**  
    This Jupyter Notebook contains exploratory data analysis (EDA) to understand trends, correlations, and distributions in the dataset.

- **`model_evaluation.ipynb`**  
    A notebook dedicated to evaluating the performance of different machine learning models and comparing their results.

- **`future_predictions.ipynb`**  
    This notebook demonstrates how predictions for future seasons (e.g., 2025) are generated using the trained models.

### Folders
- **`data/`**  
    Contains all the datasets used in the project. This includes raw data, processed data, and any additional datasets for testing.

- **`models/`**  
    Stores the trained machine learning models in serialized formats (e.g., `.pkl` or `.h5`).

- **`notebooks/`**  
    A folder to organize all Jupyter Notebooks used in the project.

- **`scripts/`**  
    Contains Python scripts for various tasks such as data preprocessing, model training, and prediction generation.

- **`results/`**  
    Includes output files such as prediction results, evaluation metrics, and visualizations.

## How to Use
1. Install dependencies:  
     ```bash
     pip install -r requirements.txt
     ```
2. Run the `predict_2025.py` script to generate predictions for the 2025 season:  
     ```bash
     python predict_2025.py
     ```
3. Explore the notebooks in the `notebooks/` folder for detailed analysis and insights.

## Acknowledgments
This project was created as a personal learning experience. Feedback and suggestions are welcome!  