import string
import nltk
import pandas
import argparse
import os.path

from sklearn.feature_extraction.text import TfidfVectorizer
from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def prepare_command_line():
    global filepath, keyword_count

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='The path to the file containing the text in the XML format')
    parser.add_argument('keywords_count', help='The number of keywords to be extracted from each text')

    args = parser.parse_args()
    filepath = args.filepath
    if not os.path.isfile(filepath):
        print('Invalid Path')
        exit()

    try:
        keyword_count = int(args.keywords_count)
        keyword_count = max(1, keyword_count)
    except ValueError:
        print('Invalid number of keywords')
        exit()

    print(f'Searching for {keyword_count} key terms at {filepath}')


def get_tree():
    return etree.parse(filepath)


# Find the articles in the XML document
def get_article():
    root = get_tree().getroot()

    for element in root.iterfind('news'):
        print(element.tag)

    for element in root.iter():
        if element.tag == 'news':
            yield element[0].text, element[1].text.lower()


def main():
    lemmatizer = WordNetLemmatizer()
    words_to_remove = list(list(stopwords.words('english')) + list(string.punctuation))
    headers = []
    texts_words = []

    for article_header, article_text in get_article():
        # Tokenize the text
        tokens = word_tokenize(article_text)

        words = []
        for token in tokens:
            # Lemmatize each token
            words.append(lemmatizer.lemmatize(token))

        texts_words.append(' '.join(words))
        headers.append(article_header)

    # Count the TF-IDF metric for each word in all stories
    vectorizer = TfidfVectorizer(stop_words=words_to_remove)
    tfidf_matrix = vectorizer.fit_transform(texts_words)
    vocabulary = vectorizer.get_feature_names()

    for text_num in range(len(headers)):
        header = headers[text_num]
        text_vocabulary = tfidf_matrix[text_num]
        data = pandas.DataFrame(text_vocabulary.T.todense(), columns=["TF-IDF"])
        data = data.sort_values('TF-IDF', ascending=False)

        print(header + ':')

        # Create tuples with the word and its value
        words = map(lambda vocabulary_index, tf_idf: (vocabulary[vocabulary_index], tf_idf[0]), data.index, data.values)

        # Filter only nouns (words with the tag NN)
        filtered_words = filter(lambda item: nltk.pos_tag([item[0]])[0][1] == 'NN', words)

        # Sort the items first by the tf-idf value than by their names in descending order
        sorted_words = sorted(filtered_words, key=lambda item: (item[1], item[0]), reverse=True)

        # Print the first words
        for x in range(keyword_count):
            word, value = sorted_words[x]
            print(word, end=' ')

        print('\n')


if __name__ == '__main__':
    filepath = ''
    keyword_count = 0

    prepare_command_line()
    main()
