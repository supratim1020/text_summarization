import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest


def summarizer(rawdocs):
    # listing stopwords
    stopwords = list(STOP_WORDS)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdocs)

    tokens = [token.text for token in doc]

    # dictionary of word frequency
    word_freq = {}
    for word in doc:
        # picking those words which aren't stopwords and punctuations
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    # Extracting the word which have highest frequency
    max_freq = max(word_freq.values())

    # Evaluating Normalized frequency (i.e., current_word_frequency / maximum_frequency)
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq

    # Storing sentence tokens in list
    sent_tokens = [sent for sent in doc.sents]

    # Addition of normalized fequency of each word of a sentence
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    # Length of summary = 30% of actual paragraph
    select_len = int(len(sent_tokens)*0.4)
    # takes those sentences(i.e., 'select_len' no of sentences) which have highest accuracy score
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    # print(summary)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    # print(summary)
    # print("Length of original text: ",len(text.split(' ')))
    # print("Length of summary text: ",len(summary.split(' ')))

    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))
