import torch
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe
from collections import Counter
import pandas as pd

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# pre trained model for tokenizing
nltk.download('punkt')


def get_glove_embeddings(df_column, dim=100):
    """
    Inputs:
    df_column: a single column of a data frame of textual data
    dim: choose dim of embedding vectors

    Outputs:
    returns tensor object of embeddings
    """
    glove = GloVe(name='6B', dim=dim)

    # set of unique words in the dataframe column, tried this to speed it up, not sure if it worked
    # tokenize to clean formatting
    unique_words = set(word for sentence in df_column for word in nltk.word_tokenize(sentence))
    
    # initialize dictionary
    word_to_vec = {word: glove[word].numpy() for word in unique_words if word in glove.stoi}

    # maps each word to embedding
    def map_to_embedding(sentence):
        return [word_to_vec.get(word, glove.get_vecs_by_tokens('<unk>').numpy()) 
                for word in nltk.word_tokenize(sentence)]

    # Use a new list to store embeddings instead of modifying the dataframe
    # supposedly faster, idk
    embeddings_list = df_column.apply(map_to_embedding).tolist()
    #convert each list to tensor
    embeddings = [torch.tensor(sentence) for sentence in embeddings_list]
    # padding
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

    return embeddings



def get_bow_embeddings(df_column = df_column, min_freq=200):

    """
    Inputs:
    df_column: a single column of a data frame of textual data
    min_freq: minimum frequency a word needs to appear to be considered

    Outputs:
    returns tensor object of BoW representations
    """
    # Flatten the dataframe column into a single list of words and count the word frequencies
    words = [word for sentence in df_column for word in nltk.word_tokenize(sentence)]
    word_freqs = Counter(words)

    # Filter out words that appear less than the minimum frequency
    filtered_words = {word: count for word, count in word_freqs.items() if count >= min_freq}

    # Sort the words by frequency and keep only the top vocab_size words
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    reduced_vocab = {word for word, _ in sorted_words}

    # Initialize a CountVectorizer object with the reduced vocabulary
    vectorizer = CountVectorizer(vocabulary=reduced_vocab)

    # Fit the vectorizer on the text data and transform the data
    bow = vectorizer.fit_transform(df_column)

    # Convert the result to a dense matrix and then to a DataFrame
    df_bow = pd.DataFrame(bow.todense(), columns=vectorizer.get_feature_names_out())

    # Convert the DataFrame to a PyTorch tensor
    tensor_bow = torch.tensor(df_bow.values)

    return tensor_bow


