# Titanic Death Predictor

A machine learning project that predicts whether a passenger survived the Titanic disaster based on various passenger attributes.

## Project Overview

This project uses the famous Titanic dataset to build a classification model that predicts passenger survival outcomes. The project includes both a machine learning pipeline (built with scikit-learn) and an interactive web application (built with Streamlit) for making predictions.

## Dataset

The project uses the Titanic dataset containing information about passengers aboard the RMS Titanic. The dataset includes:

- **Passenger Class**: Travel class (1st, 2nd, or 3rd)
- **Gender**: Male or Female
- **Age**: Passenger's age
- **Siblings/Spouse Count**: Number of siblings or spouses traveling with the passenger
- **Children/Parents Count**: Number of children or parents traveling with the passenger
- **Fare**: Ticket fare price
- **Embark Port**: Port of embarkation (S, C, or Q)
- **Survived**: Target variable (0 = Did not survive, 1 = Survived)

## Project Structure

```
├── main.py              # Streamlit web application for interactive predictions
├── pipeline.ipynb       # Machine learning pipeline and model training notebook
├── train.csv            # Titanic training dataset
└── README.md            # Project documentation
```

## Machine Learning Pipeline

The model uses a scikit-learn pipeline with the following steps:

1. **Data Preprocessing**:
   - Missing value imputation (mean for numeric, most frequent for categorical)
   - One-hot encoding for categorical features
   - MinMax scaling for numeric features

2. **Model**: Decision Tree Classifier

3. **Evaluation**: Cross-validation and accuracy scoring

## Installation & Setup

### Prerequisites

- Python 3.7+
- pip or conda

### Required Packages

```bash
pip install pandas numpy scikit-learn streamlit
```

### Training the Model

1. Open `pipeline.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load and preprocess the data
   - Split into train/test sets
   - Train the machine learning model
   - Save the trained pipeline as `pipe.pkl`

```bash
jupyter notebook pipeline.ipynb
```

## Running the Web Application

Once the model is trained (after running the notebook), start the Streamlit application:

```bash
streamlit run main.py
```

The app will open in your default web browser at `http://localhost:8501`

## Using the Predictor

1. **Input Passenger Information**:
   - Select passenger class (1-3)
   - Select gender (Male/Female)
   - Enter age (0-120)
   - Enter number of siblings/spouses traveling
   - Enter number of children/parents traveling
   - Enter ticket fare
   - Select embark port (S, C, or Q)

2. **Click "Predict"**: The model will predict whether the passenger survived or not

## Model Performance

The model is evaluated using:
- Train/test split (80/20)
- Cross-validation scores
- Accuracy metrics

## Features Used

The final model uses the following features:
- Passenger Class
- Gender
- Age
- Number of Siblings/Spouses
- Number of Children/Parents
- Fare
- Embark Port

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Testing additional classifier models
- Feature engineering for better predictive power
- Adding model performance metrics dashboard
- Containerizing the application with Docker

## License

This project uses the publicly available Titanic dataset for educational purposes.

## Author

Created as a machine learning project to predict Titanic passenger survival outcomes.
