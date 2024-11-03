import matplotlib.pyplot as plt
import numpy as np


def plot_word_frequencies_subplots(across_words, across_counts, non_across_words, non_across_counts, num_across_tweets, num_non_across_tweets, save_path):
    """
    Plot two bar charts of word frequencies as subplots in a single figure.

    Args:
    across_words (List[str]): Words from Across Protocol tweets.
    across_counts (List[int]): Counts of words from Across Protocol tweets.
    non_across_words (List[str]): Words from non-Across Protocol tweets.
    non_across_counts (List[int]): Counts of words from non-Across Protocol tweets.
    num_across_tweets (int): Total number of Across Protocol tweets.
    num_non_across_tweets (int): Total number of non-Across Protocol tweets.
    save_path (str): Path to save the figure.
    """
    # Normalize counts by the number of tweets
    across_norm_counts = np.array(across_counts) / num_across_tweets
    non_across_norm_counts = np.array(non_across_counts) / num_non_across_tweets

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # two rows, one column

    # Plot for Across Protocol tweets
    ax1.bar(across_words, across_norm_counts, color='blue')
    ax1.set_title('Most Common Words in Across Protocol Tweets')
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Normalized Frequency (per tweet)')
    ax1.set_xticklabels(across_words, rotation=45)

    # Plot for Non-Across Protocol tweets
    ax2.bar(non_across_words, non_across_norm_counts, color='green')
    ax2.set_title('Most Common Words in Non-Across Protocol Tweets')
    ax2.set_xlabel('Words')
    ax2.set_ylabel('Normalized Frequency (per tweet)')
    ax2.set_xticklabels(non_across_words, rotation=45)

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.3, top=0.95)  # Adjust the layout to make room for tick labels

    # Save the figure
    plt.savefig(save_path + "word_frequency_comparison.png")
    plt.show()
def plot_grouped_word_frequencies(across_words, across_counts, non_across_words, non_across_counts, num_across_tweets, num_non_across_tweets, save_path):
    """
    Plot a grouped bar chart of normalized word frequencies for Across Protocol related and non-related tweets.
    """
    # Normalize the counts by the number of tweets in each category
    across_norm_counts = [count / num_across_tweets for count in across_counts]
    non_across_norm_counts = [count / num_non_across_tweets for count in non_across_counts]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(14, 6))  # Increased figure size

    # Calculate bar positions
    total_width = 0.8
    ind = np.arange(len(across_words))  # the x locations for the groups
    width = total_width / 2  # the width of the bars

    # Plot bars for Across Protocol words
    rects1 = ax.bar(ind - width/2, across_norm_counts, width, label='Across Protocol', color='blue')

    # Align words from both categories
    non_across_aligned_counts = [non_across_norm_counts[non_across_words.index(word)] if word in non_across_words else 0 for word in across_words]

    # Plot bars for non-Across Protocol words
    rects2 = ax.bar(ind + width/2, non_across_aligned_counts, width, label='Non-Across Protocol', color='green')

    # Add some text for labels, title, and axes ticks
    ax.set_xlabel('Words')
    ax.set_ylabel('Normalized Frequency (per tweet)')
    ax.set_title('Comparison of Normalized Word Frequencies')
    ax.set_xticks(ind)
    ax.set_xticklabels(across_words, rotation=45, ha='right')  # Adjusted rotation and alignment

    ax.legend()

    plt.tight_layout()  # Adjust layout
    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin to prevent cutting off x-tick labels
    plt.savefig(save_path + "word_frequency_diff.png")
    plt.show()

def plot_word_frequencies(words, counts, title, color, save_path):
    """
    Plot a bar chart of word frequencies.

    Args:
    words (List[str]): A list of words.
    counts (List[int]): A list of counts corresponding to the words.
    title (str): The title for the plot.
    color (str): Color of the bars in the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color=color)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.tight_layout()  # Adjust layout
    plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin to prevent cutting off x-tick labels
    plt.savefig(save_path + f"word_frequency_{title}.png")
    plt.show()

def plot_label_hist(data, save_path):
    # Count the frequency of each label
    label_counts = data['is_related_to_Across_protocol'].value_counts()

    # Plotting
    plt.figure(figsize=(8, 5))
    label_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Tweet Labels')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=['Label 0: General Usage', 'Label 1: Across Protocol'], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path + "label_hist.png")
    plt.show()

def plot_tweer_length_hist(data, save_path):
    if 'text_length' not in data.columns:
        data['text_length'] = data['text'].apply(len)
        data['word_count'] = data['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data['text_length'], bins=30, color='skyblue')
    plt.title('Distribution of Tweet Lengths (Characters)')
    plt.xlabel('Length of tweets')
    plt.ylabel('Number of tweets')

    plt.subplot(1, 2, 2)
    plt.hist(data['word_count'], bins=30, color='skyblue')
    plt.title('Distribution of Tweet Lengths (Words)')
    plt.xlabel('Number of words')
    plt.ylabel('Number of tweets')

    plt.tight_layout()
    plt.savefig(save_path + "word_count.png")
    plt.show()
