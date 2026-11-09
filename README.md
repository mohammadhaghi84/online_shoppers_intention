🛍️ Online Shoppers Purchasing Intention Prediction

Python 3.10+ Scikit-Learn Streamlit Pandas License: MIT


An end-to-end Machine Learning and web application project that predicts whether an online e-commerce visitor will complete a purchase transaction (Revenue = True / False) during their session.


The project encompasses exploratory data analysis (EDA), automated feature preprocessing pipelines, comparative benchmark of multiple classification algorithms under class imbalance, hyperparameter tuning with 5-fold cross-validation, and deployment via an interactive, production-grade Streamlit web app.



📌 Table of Contents

Executive Summary
Dataset Overview
Key EDA Insights
Machine Learning Pipeline
Model Benchmarking & Evaluation
Interactive Web Application
Project Architecture
Getting Started
How to Run
Authors & Acknowledgments


🚀 Executive Summary

E-commerce conversion rates often hover around 2-3%. Identifying high-intent shoppers in real time allows businesses to deploy targeted promotions, optimize ad spends, and prevent cart abandonment.


Using session and behavioral web analytics, this project develops a robust predictive pipeline capable of detecting purchasing intent despite strong class imbalance (~84.5% non-purchasers vs. ~15.5% purchasers). The final tuned Random Forest Classifier achieved an F1-score of 0.617 on the minority purchase class (with an overall test accuracy of ~89.8%), outperforming baseline models, Logistic Regression, KNN, SVM, Decision Trees, and ensemble voting classifiers.



📊 Dataset Overview

The dataset consists of 12,330 online shopping sessions with 18 attributes (10 numerical, 7 categorical/indicator, and 1 binary target):


Feature Category	Features	Description
Page Activity	Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration	Counts and total duration (seconds) spent across different web page categories.
Google Analytics Metrics	BounceRates, ExitRates, PageValues	Standard web metrics measuring visitor drop-off and transactional page value prior to checkout.
Contextual Factors	SpecialDay, Month, Weekend	Seasonal event proximity (e.g., Valentine's Day), calendar month, and weekend indicators.
Technical & Traffic	OperatingSystems, Browser, Region, TrafficType, VisitorType	Client platform specifications, geolocation segment, visitor cohort, and marketing source.
Target Variable	Revenue	Boolean (True / False) indicating whether the session resulted in a purchase.


🔍 Key EDA Insights

Impact of Page Values (PageValues):
Sessions resulting in purchase have a mean PageValues score of 27.26 (median: 16.76), compared to 1.98 (median: 0.0) for non-purchasing visits. It serves as the single strongest predictive feature across all models.
Seasonality & Purchasing Rates:
November accounts for both high traffic (2,998 sessions) and the highest conversion rate (25.35%), largely driven by seasonal Black Friday/Cyber Monday campaigns.
May attracted the largest session volume (3,364 sessions) but converted at only 10.85%.
Visitor Cohort Dynamics:
New Visitors convert at 24.91% (1,694 sessions), significantly higher than Returning Visitors at 13.93% (10,551 sessions), indicating strong intent among first-time arrivals.
Class Imbalance:
Non-purchasing sessions (Revenue = False): 10,422 (84.53%)
Purchasing sessions (Revenue = True): 1,908 (15.47%)
Strategic approach: Accuracy alone is misleading; F1-Score, Precision, and Recall on the positive class serve as primary evaluation metrics.


⚙️ Machine Learning Pipeline

The pipeline uses scikit-learn's Pipeline and ColumnTransformer to completely eliminate data leakage during training, cross-validation, and production inference:


[Raw Features Input]
        │
        ├── Numerical Features (14) ───> [SimpleImputer (median)] ───> [StandardScaler] ──────┐
        │                                                                                     ▼
        └── Categorical Features (3) ──> [SimpleImputer (most_freq)] ─> [OneHotEncoder] ─> [Feature Matrix (29)]
                                                                                              │
                                                                                              ▼
                                                                                    [Tuned Random Forest]
                                                                                              │
                                                                                              ▼
                                                                                    [Prediction & Probability]

Stratified Split: 80% train (9,864 sessions) / 20% test (2,466 sessions), maintaining the ~15.5% target proportion.
Handling Imbalance: Integrated class_weight='balanced' within tree classifiers.
Hyperparameter Optimization: Conducted 5-Fold Cross-Validation using GridSearchCV scoring on f1:
n_estimators: [50, 100, 200]
max_depth: [10, 20, None]
min_samples_split: [2, 5]
Best Parameters: {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5} (Best CV F1: 0.6864).


📈 Model Benchmarking & Evaluation

Performance on held-out test data (2,466 sessions):


Model	Test Accuracy	Precision (True)	Recall (True)	F1-Score (True)	Notes
Dummy Classifier (Baseline)	84.51%	0.00%	0.00%	0.000	Always predicts majority class (False)
Logistic Regression	88.12%	74.32%	35.60%	0.481	Linear decision boundary; misses non-linear interactions
K-Nearest Neighbors (KNN)	87.31%	64.56%	40.05%	0.494	Distance-based clustering on normalized features
Decision Tree	85.64%	53.65%	53.93%	0.538	Higher recall, but prone to high variance
SVM (RBF Kernel)	88.69%	70.68%	46.07%	0.558	Strong separation; computationally heavier
Voting Ensemble (Soft)	89.17%	76.02%	43.98%	0.557	Weighted probability consensus
Tuned Random Forest (Selected)	89.82%	73.99%	52.88%	0.617	Optimal balance of precision, recall, and stability

Confusion Matrix (Selected Model)

True Negatives (False correctly predicted): 2,019
False Positives (False predicted as True): 65
False Negatives (True missed): 180
True Positives (True correctly caught): 202


🖥️ Interactive Web Application

An end-user web dashboard was built using Streamlit to demonstrate the trained model in production:


Clean UI & Responsive Layout: Modern custom CSS, styled cards, and intuitive section groupings.
Real-Time Probability Estimation: Displays both binary prediction (Revenue = True/False) and a confidence percentage meter (predict_proba).
Input Controls: Categorized into Page Engagement, Session Quality (Bounce/Exit rates, PageValues), and Visitor Demographics.
Standalone Architecture: Directly loads the serialized .joblib model pipeline without requiring external scoring scripts.


📁 Project Architecture

├── online-shoppers-purchasing-intention.ipynb   # Complete Jupyter Notebook (EDA, Modeling, CV)
├── shopping_intention_model.joblib             # Serialized best Random Forest pipeline (preprocessor + model)
├── score.py                                    # Standalone Python scoring & inference module
├── app.py                                      # Streamlit web deployment script
├── requirements.txt                            # Project dependencies
└── README.md                                   # Project documentation


🛠️ Getting Started

Prerequisites

Python 3.9 or higher
pip package manager

Installation

Clone the repository:

git clone https://github.com/your-username/online-shoppers-intention.git
cd online-shoppers-intention
Create and activate a virtual environment:

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Or install manually:

pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit


💻 How to Run

1. Model Training & Export

To retrain the model and regenerate the serialized pipeline binary: Open and run all cells in online-shoppers-purchasing-intention.ipynb. The notebook will automatically output shopping_intention_model.joblib.


2. Standalone Script Inference

You can test programmatic scoring using score.py:


python score.py

3. Launching the Web App

Run the Streamlit application from the terminal:


streamlit run app.py

The application will launch automatically in your browser at http://localhost:8501.



💡 Example Prediction Payload

import joblib
import pandas as pd

# Load serialized model pipeline
model = joblib.load("shopping_intention_model.joblib")

# Sample visitor session
session_data = pd.DataFrame([{
    "Administrative": 1,
    "Administrative_Duration": 25.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 18,
    "ProductRelated_Duration": 650.0,
    "BounceRates": 0.015,
    "ExitRates": 0.035,
    "PageValues": 16.5,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": False
}])

# Inference
prediction = model.predict(session_data)[0]
probability = model.predict_proba(session_data)[0][1]

print(f"Purchase Intent: {prediction} | Conversion Probability: {probability:.2%}")
# Output: Purchase Intent: True | Conversion Probability: 78.40%