# Key-Terms-Extractor
 A program that extracts key terms from news articles using NLTK and sklearn to apply tokenization, lemmatization, part-of-speech tagging and tf-idf vectorization.

## Features
* Command line arguments
* XML parsing with lxml
* Word tokenization with nltk
* Word lemmatization using Word Net Lemmatizer 
* Stopwords removal
* TF-IDF (term frequency-inverse document frequency) vectorization using sklearn

## Usage
 This program reads the texts from an XML file. This file must contain 'news' tags that have 2
tags: the first one is the header, and the second one is the text. Example:
```xml
<news>
    <value>Brain Disconnects During Sleep</value>
    <value>Scientists may have ... in Pasadena.</value>
</news>
 ```

 You should specify the path to the file, and the number of keywords to extract in the command line. Example:
 ```commandline
 python key_terms.py example.xml 5
 ```

## Output
```
Searching for 5 key terms at example.xml
Brain Disconnects During Sleep:
sleep cortex consciousness tononi tm 

New Portuguese skull may be an early relative of Neandertals:
skull fossil europe trait genus
```

