# ==========================================================
# AI DEVOPS INCIDENT CLASSIFIER
# MODEL TRAINING SCRIPT
#
# Purpose:
#   Train a machine learning model that classifies
#   DevOps incidents into predefined categories.
#
# Workflow:
#   1. Load training dataset
#   2. Split features and labels
#   3. Create ML pipeline
#   4. Train the model
#   5. Save trained model
# ==========================================================

# ----------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

print("Loading training dataset...")

df = pd.read_csv("data/incidents.csv")

# ----------------------------------------------------------
# PREPARE FEATURES AND LABELS
# ----------------------------------------------------------

X = df["text"]
y = df["label"]

# ----------------------------------------------------------
# BUILD MACHINE LEARNING PIPELINE
# ----------------------------------------------------------

pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer()
    ),

    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )

])

# ----------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------

print("Training AI model...")

pipeline.fit(X, y)

# ----------------------------------------------------------
# SAVE TRAINED MODEL
# ----------------------------------------------------------

joblib.dump(
    pipeline,
    "models/incident_classifier.joblib"
)

# ----------------------------------------------------------
# TRAINING COMPLETE
# ----------------------------------------------------------

print("Model trained successfully.")
