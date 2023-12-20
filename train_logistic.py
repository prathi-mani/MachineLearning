import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# Load the datasets
train_path = 'train.txt'
dev_path = 'dev.txt'
train_df = pd.read_csv(train_path, delimiter='\t', header=None, names=['label', 'text'])
dev_df = pd.read_csv(dev_path, delimiter='\t', header=None, names=['label', 'text'])

# Preprocessing: Convert labels from '+' to 1 and '-' to 0
train_df['label'] = train_df['label'].apply(lambda x: 1 if x == '+' else 0)
dev_df['label'] = dev_df['label'].apply(lambda x: 1 if x == '+' else 0)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(train_df['text'])
X_dev_tfidf = vectorizer.transform(dev_df['text'])

# Labels
y_train = train_df['label']
y_dev = dev_df['label']

# Simulate "epochs" by incrementally increasing the training data size
for epoch in range(1, 11):
    # Calculate the number of samples to include in this "epoch"
    subset_size = int(X_train_tfidf.shape[0] * epoch / 10)
    #print(subset_size)
    X_train_subset = X_train_tfidf[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Train Logistic Regression model on the subset
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_subset, y_train_subset)
    
    # Predict on the development set
    y_dev_pred = logistic_model.predict(X_dev_tfidf)
    
    # Calculate and print the development error
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    dev_error = 1 - dev_accuracy

     # Calculate positive ratio (recall) manually
# Calculate predicted positive ratio
    predicted_positives = (y_dev_pred == 1).sum()
    predicted_positive_ratio = predicted_positives / len(y_dev_pred) if len(y_dev_pred) > 0 else 0

    print(f"Epoch {epoch}, Development Error: {dev_error:.2%}")
   # print(f"Predicted Positive Ratio: {predicted_positive_ratio:.2%}")

    