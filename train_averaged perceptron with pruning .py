#!/usr/bin/env python3

from __future__ import division
import sys
import time
from svector import svector
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS





def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words, word_counts):
    v = svector()
    for word in words:
        if  word in word_counts and word_counts[word] > 1:
            v[word] += 1
    v['bias'] = 1
    return v
    
def test(devfile, model, word_counts):
    false_positives = []
    false_negatives = []
    positive= 0
    label_count = sum(1 for label, _ in read_from(devfile) if label == 1) 
    for i, (label, words) in enumerate(read_from(devfile), 1):
        features = make_vector(words, word_counts)
        prediction = model.dot(features)
        confidence = abs(prediction)
        
        if label == -1 and prediction > 0:
            false_positives.append((words, confidence))
        elif label == 1 and prediction <= 0:
            false_negatives.append((words, confidence))

    false_positives.sort(key=lambda x: x[1], reverse=True)
    false_negatives.sort(key=lambda x: x[1], reverse=True)
    
    recall = len(false_negatives) / (len(false_negatives) + sum(1 for label, _ in read_from(devfile) if label == 1))
    error_rate = (len(false_negatives) + len(false_positives)) / i
    
   
    return error_rate, recall, false_positives[:5], false_negatives[:5],positive/label_count

def compute_word_counts(trainfile):
    word_counts = {}
    for _, words in read_from(trainfile):
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def train(trainfile, devfile, epochs=5):
    word_counts = compute_word_counts(trainfile)
    print("Model size before neglecting one-count words:", len(word_counts))
    t = time.time()
    best_err = 1.
    model = svector()
    model_avg = svector()
    c = 0
    total_updates = 0 
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1):
            sent = make_vector(words, word_counts)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_avg += label * sent * c
            c += 1
        total_updates += updates
        model_output = c * model - model_avg
        
        dev_err, dev_recall, strong_false_positives, strong_false_negatives, positive = test(devfile, model_output, word_counts)
        best_err = min(best_err, dev_err)
        
        print("epoch %d, update %.1f%%, dev %.1f%%  " % (it, updates / i * 100, dev_err * 100))
        #print(f"Epoch {it}, Recall: {dev_recall:.2%}, Development Error: {dev_err:.2%}")
        

    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model_output), time.time() - t))
  #  print("Modified: Average update percentage: %.1f%%" % (total_updates / (epochs * i) * 100))

    # Strongly incorrect predictions
    #print("\nStrongly incorrect false positives:")
    #for words, confidence in strong_false_positives:
       # print(' '.join(words), "Confidence:", confidence)

    #print("\nStrongly incorrect false negatives:")
    #for words, confidence in strong_false_negatives:
       # print(' '.join(words), "Confidence:", confidence)


 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 text_classifier.py TRAINFILE DEVFILE")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2], 10)
