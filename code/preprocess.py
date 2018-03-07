"""
Code for load pretrained embeddings and convert the text to embeddings

embedding used: https://github.com/facebookresearch/fastText/blob/master/docs/english-vectors.md
"""
import numpy as np
import tqdm
import os, re, csv, math, codecs
import nltk
import argparse
import sys
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from autocorrect import spell

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def read_embedding_list(file_path):
    """
    This is used for load fasttext embeddings
    :param file_path:
    :return:
    """
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path) as f:
        for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:-1]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train


def tokenize_sentences(sentences, words_dict):
    # old tokenize sentense
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def save(path, X_train, X_test, y_train, embeddings):
    embedding_path = os.path.join(path, 'embeddings.npz')
    np.savez(embedding_path, embeddings)

    xtrain_path = os.path.join(path, 'train.npz')
    np.savez(xtrain_path, X_train)

    xtest_path = os.path.join(path, 'test.npz')
    np.savez(xtest_path, X_test)

    label_path = os.path.join(path, 'label.npz')
    np.savez(label_path, y_train)


def clean_text(sentences, correct = False):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    result = []
    for senstence in tqdm.tqdm(sentences):
        senstence = senstence.replace("'","")
        tokens = tokenizer.tokenize(senstence)
        if correct:
            filtered = [spell(word) for word in tokens if word not in stop_words]
        else:
            filtered = [word for word in tokens if word not in stop_words]
        result.append(" ".join(filtered))

    return result

def get_tokenizer(text, num_words, char_level = False):
    tokenizer = Tokenizer(num_words=num_words, lower=True, char_level=char_level)
    tokenizer.fit_on_texts(text)
    return tokenizer

def load_embedding(path):
    embeddings_index = {}
    f = codecs.open(path, encoding='utf-8')
    for line in tqdm.tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def prepare_embedding(embeddings_index, word_index, max_word, embed_dim):
    words_not_found = []
    nb_words = min(max_word, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)

    return embedding_matrix

def main():
    parser = argparse.ArgumentParser(description="convert text to emebeddings")
    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("save_path")
    parser.add_argument("--sentences-length", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=200000)
    parser.add_argument("--correct", action='store_true', default=False)

    try:
        args = parser.parse_args()

    except:
        parser.print_help()
        sys.exit(1)

    print("Loading data...")
    train_data = pd.read_csv(args.train_file_path)
    test_data = pd.read_csv(args.test_file_path)

    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    y_train = train_data[CLASSES].values

    cleaned_train = clean_text(list_sentences_train, args.correct)
    cleaned_test = clean_text(list_sentences_test, args.correct)
    tokenizer = get_tokenizer(cleaned_train + cleaned_test, args.max_words)

    print("Tokenizing sentences in train set...")
    tokenized_sentences_train = tokenizer.texts_to_sequences(cleaned_train)
    """
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
    """


    print("Tokenizing sentences in test set...")
    tokenized_sentences_test = tokenizer.texts_to_sequences(cleaned_test)
    """
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)
    """
    words_dict = tokenizer.word_index
    words_dict[UNKNOWN_WORD] = len(words_dict)

    print("Loading embeddings...")
    #embedding_list, embedding_word_dict = read_embedding_list(args.embedding_path)
    embedding_word_dict = load_embedding(args.embedding_path)
    embedding_size = len(list(embedding_word_dict.values())[0])

    print("Preparing data...")
    """
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)
    """
    embedding_matrix = prepare_embedding(embedding_word_dict, words_dict, args.max_words, embedding_size)

    """
    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    """

    train_list_of_token_ids = sequence.pad_sequences(tokenized_sentences_train, maxlen=args.sentences_length)
    test_list_of_token_ids = sequence.pad_sequences(tokenized_sentences_test, maxlen=args.sentences_length)

    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)

    save(args.save_path, X_train, X_test, y_train, embedding_matrix)


if __name__ == '__main__':
    main()
