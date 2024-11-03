import pandas as pd
import re
from nltk.tokenize import word_tokenize, TweetTokenizer
from src.utils import *
from src.visualization import *

import matplotlib.pyplot as plt
from collections import Counter

class DatasetHandler():
    def __init__(self, paths):
        self.paths = paths
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.paths.raw_dataset_path)
        self.data['text_length'] = self.data['text'].apply(len)
        self.data['word_count'] = self.data['text'].apply(lambda x: len(x.split()))

        return self.data

    def preprocess_data(self):
        self.data['clean_text'] = self.data['text'].apply(self.preprocess)
        self.data.to_csv(self.paths.path_result + 'clean_data.csv')
        return self.data

    def preprocess(self, text):
        # Replace @usernames with <username>
        text = re.sub(r'\@[\w]+', '<username>', text)

        # Replace URLs with <url>
        text = re.sub(r'http\S+|www\S+|https\S+', '<url>', text, flags=re.MULTILINE)

        # Replace email addresses with <email>
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<email>', text)

        # Convert '%' to ' percent' without removing it during the cleaning process
        text = re.sub(r'%', ' percent', text)

        # Modify regex to keep only alphanumeric characters, specific punctuation, and spaces
        # Adding '.' to keep decimal points, and '+', '-' for changes in values
        text = re.sub(r'[^A-Za-z0-9&$().\'+\-\s]+', ' ', text)

        # Tokenization - this will ensure that tokens are cleanly separated
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(text)

        # Convert all tokens into lowercase to maintain consistency and join them back into a string
        text = ' '.join(tokens).lower()

        return text


    def explanatory_data_analysis(self):
        across_tweets = self.data[self.data['is_related_to_Across_protocol'] == 1]['clean_text']
        non_across_tweets = self.data[self.data['is_related_to_Across_protocol'] == 0]['clean_text']

        plot_label_hist(self.data, self.paths.path_result)
        plot_tweer_length_hist(self.data, self.paths.path_result)

        # Count words for Across Protocol tweets
        across_words, across_counts, num_across_tweets = count_keywords(across_tweets)
        # Count words for non-Across Protocol tweets
        non_across_words, non_across_counts, num_non_across_tweets = count_keywords(non_across_tweets)

        plot_word_frequencies_subplots(across_words, across_counts, non_across_words, non_across_counts,
                                       num_across_tweets,
                                       num_non_across_tweets, self.paths.path_result)

        plot_grouped_word_frequencies(across_words, across_counts, non_across_words, non_across_counts,
                                      num_across_tweets=num_across_tweets,
                                      num_non_across_tweets=num_non_across_tweets,
                                      save_path=self.paths.path_result)

