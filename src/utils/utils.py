from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

def save_metrics(metrics, file_name):
    df_metrics = pd.DataFrame(metrics, columns=['Model', 'Precision', 'Recall', 'F1-score'])
    df_metrics.to_csv(file_name, index=False)
def evaluate_model(predicted_labels, true_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    prf = precision_recall_fscore_support(true_labels, predicted_labels, pos_label=1, average='binary')
    return cm, prf[0], prf[1], prf[2]

def update_dataframe(data_frame, predicted_labels, model_tag):
    data_frame[f'predicted_{model_tag}'] = predicted_labels
    return data_frame


def plot_confusion_matrix(cm, model_name, save_path):
    # Set larger font sizes
    sns.set_context('talk')  # sets the context to "talk", which has larger fonts by default
    plt.figure(figsize=(8, 8))  # increase figure size for better readability

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                     annot_kws={'size': 22})  # increase annotation font size

    # Set labels with larger font sizes
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Actual', fontsize=20)
    plt.title(f'{model_name}', fontsize=28)

    # Adjust layout to make room for labels if necessary
    plt.tight_layout()

    plt.savefig(save_path + f"{model_name}_confusion_matrix.png")
    plt.show()
def count_keywords(tweets, additional_stopwords=set()):
    """
    Count words in tweets, excluding common stopwords and any specified additional stopwords.
    Returns the word counts and the total number of tweets for normalization.
    """
    stop_words = set(stopwords.words('english')) | additional_stopwords
    words = " ".join(tweets).split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(20)
    words, counts = zip(*most_common_words)
    return words, counts, len(tweets)