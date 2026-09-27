# Titanic Survival Prediction

A complete Machine Learning and Deep Learning project that predicts Titanic passenger survival and provides a Streamlit dashboard.

## Project structure

```text
Titanic-Survival-Prediction/
├── data/
│   └── Titanic-Dataset.csv
├── models/
├── images/
├── notebooks/
├── results/
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Features

- Titanic dataset preprocessing
- Missing-value handling
- One-hot encoding and feature scaling
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Artificial Neural Network (ANN)
- 1D CNN
- Accuracy, Precision, Recall, F1 Score and ROC-AUC comparison
- Confusion matrix
- Saved deployable scikit-learn pipeline
- Streamlit prediction dashboard

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script saves the deployable model to `models/titanic_best_model.pkl` and evaluation files to `results/`.

## Streamlit deployment

Upload the complete repository to GitHub. On Streamlit Community Cloud, choose `app.py` as the main file and deploy from the repository.

## Important

The dataset used by the training script must remain at `data/Titanic-Dataset.csv`.
