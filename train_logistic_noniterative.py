import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to convert labels from '+' to 1 and '-' to 0
def convert_label(label):
    return 1 if label == '+' else 0

# Function to predict and write the output to a file
def predict_and_write_output(vectorizer, model, test_df, outputfile):
    # Transform the test data using the loaded vectorizer
    X_test_tfidf = vectorizer.transform(test_df['text'])
    
    # Predict using the logistic regression model
    y_test_pred = model.predict(X_test_tfidf)
    
    # Write the predictions to the output file
    with open(outputfile, 'w') as fout:
        for text, label in zip(test_df['text'], y_test_pred):
            label_char = '+' if label == 1 else '-'
            fout.write(f"{label_char}\t{text}\n")

# Load the datasets
train_path = 'train.txt'
dev_path = 'test.txt'
train_df = pd.read_csv(train_path, delimiter='\t', header=None, names=['label', 'text'])
dev_df = pd.read_csv(dev_path, delimiter='\t', header=None, names=['label', 'text'])

# Preprocessing: Convert labels from '+' to 1 and '-' to 0
train_df['label'] = train_df['label'].map(convert_label)
dev_df['label'] = dev_df['label'].map(convert_label)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(train_df['text'])
X_dev_tfidf = vectorizer.transform(dev_df['text'])

# Labels
y_train = train_df['label']
y_dev = dev_df['label']

# Train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_tfidf, y_train)

# Predict on the development set
y_dev_pred = logistic_model.predict(X_dev_tfidf)

# Calculate and print the accuracy and error on the development set
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
dev_error = 1 - dev_accuracy

# Calculate predicted positive ratio
predicted_positives = (y_dev_pred == 1).sum()
predicted_positive_ratio = predicted_positives / len(y_dev_pred) if len(y_dev_pred) > 0 else 0

print(f"Development Accuracy: {dev_accuracy:.2%}")
print(f"Development Error: {dev_error:.2%}")
print(f"Predicted Positive Ratio: {predicted_positive_ratio:.2%}")

# Predict and write output for a test set (here we use the dev set as an example)
outputfile = 'test.logisticregression.txt.prediction'
predict_and_write_output(vectorizer, logistic_model, dev_df, outputfile)
print(f"Predictions written to {outputfile}")
