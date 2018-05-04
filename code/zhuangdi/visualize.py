#!/usr/bin/env python
"""
Minimal Example
===============

Generating a square wordcloud from the US constitution using default arguments.
"""

from os import path
from wordcloud import WordCloud
import pandas as pd

d = path.dirname(__file__)
d = '../../data/'

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def get_text():
    ## Read the whole text.
    #text = open(path.join(d, 'constitution.txt')).read()
    #text = open(path.join(d, 'train.csv')).read()
    text_f = pd.read_csv(d + 'train.csv')
    text_f.fillna(' ', inplace=True)
    text_content = text_f["comment_text"]
    text = text_content.values.tolist()

    text_labels = text_f[list_classes].values
    neg_text = [text[i] for i, label in enumerate(text_labels) if any(label)]
    pos_text = [text[i] for i, label in enumerate(text_labels) if any(label) == False]

    neg_text = " ".join(neg_text)
    pos_text = " ".join(pos_text)
    return neg_text, pos_text

def main():
    neg_text, pos_text = get_text()

    # Generate a word cloud image
    wordcloud = WordCloud().generate(neg_text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(neg_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    # The pil way (if you don't have matplotlib)
    # image = wordcloud.to_image()
    # image.show()

main()
