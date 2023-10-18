import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

text = """Samsung recently cancelled its in-person MWC 2021 event, instead, committing to an online-only format. The South Korean tech giant recently made it official, setting a time and date for the Samsung Galaxy MWC Virtual Event.

The event will be held on June 28 at 17:15 UTC (22:45 IST) and will be live-streamed on YouTube. In its release, Samsung says that it will introduce its “ever-expanding Galaxy device ecosystem”. Samsung also plans to present the latest technologies and innovation efforts in relation to the growing importance of smart device security.

Samsung will also showcase its vision for the future of smartwatches to provide new experiences for users and new opportunities for developers. Samsung also shared an image for the event with silhouettes of a smartwatch, a smartphone, a tablet and a laptop."""

# Stop words are the words in a stop list (or stoplist or negative dictionary) which are filtered out (i.e. stopped) before or after processing of natural language data (text) because they are insignificant.
# So we have to drop stop_words as well as punctuations
stopwords = list(STOP_WORDS)
# print(stopwords)
# Out[]: ['really', 'all', 'fifteen', '’ll', 'somewhere', 'nine', 'neither', ... 'even', 'just', 'towards', 'well', 'wherein', 'least']

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
# print(doc)
# Out[]: Samsung recently cancelled its in-person MWC 2021 event, ... silhouettes of a smartwatch, a smartphone, a tablet and a laptop.

# storing each word of 'doc' into list tokens(here we are considering each word as a token).
tokens = [token.text for token in doc]
# print(tokens)
# Out[]: ['Samsung', 'recently', 'cancelled', 'its', 'in', '-', 'person', ... ,',', 'a', 'tablet', 'and', 'a', 'laptop', '.']

# dictionary of word frequency
word_freq = {}
for word in doc:
    # picking those words which aren't stopwords and punctuations
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
        if word.text not in word_freq.keys():
            word_freq[word.text] = 1
        else:
            word_freq[word.text] += 1
# print(word_freq)
# Out[]: {'Samsung': 6, 'recently': 2, 'cancelled': 1, 'person': 1, 'MWC': 2, ... ,'smartwatch': 1, 'smartphone': 1, 'tablet': 1, 'laptop': 1}

# Extracting the word which have highest frequency
max_freq = max(word_freq.values())
# print(max_freq)
# Out[]: 6

# Evaluating Normalized frequency (i.e., current_word_frequency / maximum_frequency)
for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq
# print(word_freq)
# Out[]: {'Samsung': 1.0, 'recently': 0.3333333333333333, 'cancelled': 0.16666666666666666, ... , 'smartphone': 0.16666666666666666, 'tablet': 0.16666666666666666, 'laptop': 0.16666666666666666}

# Storing sentence tokens in list
sent_tokens = [sent for sent in doc.sents]
# print(len(sent_tokens))
# out[]: 7
# print(sent_tokens)
# [Samsung recently cancelled its in-person MWC 2021 event, instead, committing to an online-only format., ... ,Samsung also shared an image for the event with silhouettes of a smartwatch, a smartphone, a tablet and a laptop.]

# Addition of normalized fequency of each word of a sentence
sent_scores = {}
for sent in sent_tokens:
    for word in sent:
        if word.text in word_freq.keys():
            if sent not in sent_scores.keys():
                sent_scores[sent] = word_freq[word.text]
            else:
                sent_scores[sent] += word_freq[word.text]
# print(sent_scores)
# out[]: {Samsung recently cancelled its in-person MWC 2021 event, instead, committing to an online-only format.: 3.3333333333333326, ... , Samsung also shared an image for the event with silhouettes of a smartwatch, a smartphone, a tablet and a laptop.: 2.666666666666666}

# Length of summary = 30% of actual paragraph
select_len = int(len(sent_tokens)*0.3)
# print(select_len)
# takes those sentences(i.e., 'select_len' no of sentences) which have highest accuracy score
summary = nlargest(select_len, sent_scores, key=sent_scores.get)
# print(summary)

final_summary=[word.text for word in summary]
summary=' '.join(final_summary)
print(summary)
print("Length of original text: ",len(text.split(' ')))
print("Length of summary text: ",len(summary.split(' ')))

