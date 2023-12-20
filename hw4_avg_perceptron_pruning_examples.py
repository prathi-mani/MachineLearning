import numpy as np
import os
import time
from gensim.models import KeyedVectors
from collections import Counter

class SVector(dict):
    def dot(self, other):
        return sum(self.get(k, 0) * v for k, v in other.items())

def word_count(file_path):
    word_counter = Counter()
    with open(file_path, 'r') as file:
        for line in file:
            _, sentence = line.strip().split(maxsplit=1)
            words = sentence.split()
            word_counter.update(words)
    return word_counter

def compute_sentence_embedding(sentence, model, word_counter):
    words = sentence.split()
    valid_words = [word for word in words if word in model.key_to_index and word_counter[word] > 1]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model[word] for word in valid_words], axis=0)

def one_hot_encode(sentence, vocab):
    encoding = np.zeros(len(vocab))
    for word in sentence.split():
        if word in vocab:
            encoding[vocab[word]] = 1
    return encoding

def load_data(file_path, model, word_counter, vocab):
    sentences, labels, one_hot_sentences = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            label, sentence = line.strip().split(maxsplit=1)
            labels.append(1 if label == '+' else -1)
            sentences.append(compute_sentence_embedding(sentence, model, word_counter))
            one_hot_sentences.append(one_hot_encode(sentence, vocab))
    return np.array(sentences), np.array(labels), np.array(one_hot_sentences)

def train(train_embeddings, train_labels, dev_embeddings, dev_labels, epochs=5):
    t = time.time()
    best_err = 1.
    model = SVector()
    model_avg = SVector()
    c = 1
    for it in range(1, epochs + 1):
        updates = 0
        for i in range(len(train_embeddings)):
            label = train_labels[i]
            sent_embedding = dict(enumerate(train_embeddings[i]))
            if label * model.dot(sent_embedding) <= 0:
                updates += 1
                update = {k: label * v for k, v in sent_embedding.items()}
                model.update((k, model.get(k, 0) + v) for k, v in update.items())
                model_avg.update((k, model_avg.get(k, 0) + c * v) for k, v in update.items())
            c += 1

        averaged_model = {k: model.get(k, 0) - (model_avg.get(k, 0) / c) for k in set(model).union(model_avg)}

        dev_err = test(dev_embeddings, dev_labels, averaged_model)
        best_err = min(best_err, dev_err)
        print(f"epoch {it}, updates {updates / len(train_embeddings) * 100:.1f}%, dev error {dev_err * 100:.1f}%")
    print(f"best dev error {best_err * 100:.1f}%, |w|={len(averaged_model)}, time: {time.time() - t:.1f} secs")


def get_vector(model, key, size):
    if key in model.key_to_index:
        return model[key]
    else:
        return np.zeros(size)

def test(dev_embeddings, dev_labels, model):
    correct = 0
    for i in range(len(dev_embeddings)):
        sent_embedding = dict(enumerate(dev_embeddings[i]))
        prediction = 1 if sum(model.get(k, 0) * v for k, v in sent_embedding.items()) > 0 else -1
        if prediction == dev_labels[i]:
            correct += 1
    return 1 - correct / len(dev_embeddings)

# Load the custom pre-trained Word2Vec model
model_path = 'embs_train.kv'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} was not found.")
model = KeyedVectors.load(model_path)

# Count word occurrences in the training data
word_counter = word_count('train.txt')

# Creating a vocabulary for one-hot encoding
vocab = {word: i for i, word in enumerate(word_counter.keys())}

# Load the training and development data
train_embeddings, train_labels, _ = load_data('train.txt', model, word_counter, vocab)
dev_embeddings, dev_labels, dev_one_hot = load_data('dev.txt', model, word_counter, vocab)

# Train the model with Word2Vec embeddings
train(train_embeddings, train_labels, dev_embeddings, dev_labels, epochs=10)

# Test with one-hot representations
dev_err_one_hot = test(dev_one_hot, dev_labels, model)
print(f"One-hot representation dev error: {dev_err_one_hot * 100:.1f}%")

# Compare and find examples
for i, (emb, one_hot, label) in enumerate(zip(dev_embeddings, dev_one_hot, dev_labels)):
    prediction_emb = 1 if sum(model.get(k, 0) * v for k, v in dict(enumerate(emb)).items()) > 0 else -1
    prediction_one_hot = 1 if sum(model.get(k, 0) * v for k, v in dict(enumerate(one_hot)).items()) > 0 else -1
    if prediction_emb == label and prediction_one_hot != label:
        print(f"Example {i}: Correct with Word2Vec, incorrect with one-hot")
