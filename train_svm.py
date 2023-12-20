from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the datasets
train_path = 'train.txt'
dev_path = 'dev.txt'
train_df = pd.read_csv(train_path, delimiter='\t', header=None)
dev_df = pd.read_csv(dev_path, delimiter='\t', header=None)

# Convert labels from '+' to 1 and '-' to 0
train_df[0] = train_df[0].apply(lambda x: 1 if x == '+' else 0)
dev_df[0] = dev_df[0].apply(lambda x: 1 if x == '+' else 0)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(train_df[1])
X_dev_tfidf = vectorizer.transform(dev_df[1])

# Labels
y_train = train_df[0].values
y_dev = dev_df[0].values

# Loop through each increment of the training data
for epoch in range(1, 11):
    # Calculate the number of samples to include in this "epoch"
    subset_size = int((X_train_tfidf.shape[0] * epoch) // 10)
    X_train_subset = X_train_tfidf[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Train SVM model on the subset
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_subset, y_train_subset)
    
    # Predict on the development set
    y_dev_pred = svm_model.predict(X_dev_tfidf)
    
    # Calculate and print the development error
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    dev_error = 1 - dev_accuracy
    print(f"Epoch {epoch}, Development Error: {dev_error:.2%}")
