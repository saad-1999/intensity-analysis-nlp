import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os
import zipfile

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Data Collection
def load_data(base_path):
    files = ['angriness.csv', 'happiness.csv', 'sadness.csv']
    dataframes = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(base_path, file))
            dataframes.append(df)
        except FileNotFoundError:
            print(f'Error: {file} not found.')
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df.sample(frac=1).reset_index(drop=True)

# Data Preprocessing
def preprocess_text(text, lemmatizer, stop_words):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(tokens)

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load and preprocess data
base_path = r'C:\Users\firoc\Desktop\code\Upgrad'
combined_df = load_data(base_path)
combined_df['processed_content'] = combined_df['content'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

# Feature Engineering
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(combined_df['processed_content']).toarray()
y = combined_df['intensity']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=500), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
print(f'Best Parameters: {grid.best_params_}')

# Model with best parameters
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f'Accuracy with Best Parameters: {accuracy_score(y_test, y_pred_best)}')
print(classification_report(y_test, y_pred_best))

# Save the best model and vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

