# Toxic Comment Classification Project

## Overview

This project aims to classify tweets as toxic or non-toxic using a dataset of 24,000 tweets. The primary focus is on extensive data cleaning and employing various machine learning models for classification. Additionally, a FastAPI API and a Streamlit app have been developed for interactive usage.

## Dataset

- The dataset consists of 24,000 tweets.

## Data Cleaning

Extensive data cleaning was performed, including:

- Removal of punctuation, numbers, and stop words.
- Lemmatization using the NLTK library.
- Tokenization.

## Models Used

The following machine learning models were utilized:

1. Naive Bayes
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. LightGBM (Gradient Boosting)

## Hyperparameter Tuning

Hyperparameter tuning was carried out to optimize the model performance. The parameters included:

- Number of estimators (n_estimators)
- Learning rate
- Maximum depth of trees (max_depth)
- Number of leaves (num_leaves)
- Subsample
- Colsample by tree
- Regularization alpha (reg_alpha)
- Regularization lambda (reg_lambda)
- Boosting type

## FastAPI API

A FastAPI API was developed for seamless integration with other applications. The API serves as a backend for making predictions based on the trained models.

## Streamlit App

A Streamlit app was created to provide an interactive user interface for exploring the model's predictions and visualizing the results.
