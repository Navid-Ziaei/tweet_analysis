from collections import Counter
from nltk.corpus import stopwords

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