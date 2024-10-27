# 文件：src/main.py

import pandas as pd
import numpy as np
import nltk
import re
import string
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('all')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATA_PATH = 'data/'

def load_data():
    print("Loading data...")
   
    train_df_full = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    test_ids = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))['Id']
    
    train_df = train_df_full[train_df_full['Score'].notnull()].reset_index(drop=True)
    
    test_df = train_df_full[train_df_full['Id'].isin(test_ids)].reset_index(drop=True)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True) #all 100% train data will be used
    
    return train_df, test_df, test_ids

def preprocess_text(text):
    if not isinstance(text, str):
        text = ''
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

def apply_preprocessing(df):
    print("Applying preprocessing...")
    if 'Summary' in df.columns:
        summary = df['Summary'].fillna('')
    else:
        summary = ''
    if 'Text' in df.columns:
        text = df['Text'].fillna('')
    else:
        text = ''
    df['combined_text'] = summary + ' ' + text
    df['cleaned_text'] = df['combined_text'].apply(preprocess_text)
    return df

def feature_engineering(train_df, test_df):
    print("Feature engineering...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_text'])
    X_test_tfidf = vectorizer.transform(test_df['cleaned_text'])
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_model(X, y):
    print("Training model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    return best_model

def create_submission(model, X_test, test_df, test_ids):
    print("Creating submission...")
    test_predictions = model.predict(X_test)
    
    submission_df = pd.DataFrame({
        'Id': test_df['Id'],
        'Score': test_predictions
    })
    
    submission_df = submission_df.set_index('Id').loc[test_ids].reset_index()
    
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

def main():
    train_df, test_df, test_ids = load_data()
    train_df = apply_preprocessing(train_df)
    test_df = apply_preprocessing(test_df)
    y = train_df['Score'].astype(int)
    X_train_tfidf, X_test_tfidf, vectorizer = feature_engineering(train_df, test_df)
    best_model = train_model(X_train_tfidf, y)
    create_submission(best_model, X_test_tfidf, test_df, test_ids)
    print("All done!")

if __name__ == '__main__':
    main()
