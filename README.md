# TextSummarization_TextRank
textrank algorithm using for text summarization

Understanding the Problem Statement
Being a major tennis buff, I always try to keep myself updated with what’s happening in the sport by religiously going through as many online tennis updates as possible. However, this has proven to be a rather difficult job! There are way too many resources and time is a constraint.

Therefore, I decided to design a system that could prepare a bullet-point summary for me by scanning through multiple articles. How to go about doing this? That’s what I’ll show you in this tutorial. We will apply the TextRank algorithm on a dataset of scraped articles with the aim of creating a nice and concise summary.



 

Please note that this is essentially a single-domain-multiple-documents summarization task, i.e., we will take multiple articles as input and generate a single bullet-point summary. Multi-domain text summarization is not covered in this article, but feel free to try that out at your end.

You can download the dataset we’ll be using from here.
 

Implementation of the TextRank Algorithm
So, without any further ado, fire up your Jupyter Notebooks and let’s implement what we’ve learned so far.

Import Required Libraries
First, import the libraries we’ll be leveraging for this challenge.

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
 

Read the Data
Now let’s read our dataset. I have provided the link to download the data in the previous section (in case you missed it).

df = pd.read_csv("tennis_articles_v4.csv")
 

Inspect the Data
Let’s take a quick glance at the data.

df.head()

We have 3 columns in our dataset — ‘article_id’, ‘article_text’, and ‘source’. We are most interested in the ‘article_text’ column as it contains the text of the articles. Let’s print some of the values of the variable just to see what they look like.

df['article_text'][0]
Output:

"Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player 
has no problems in openly speaking about it and in a recent interview she said: 'I don't really 
hide any feelings too much. I think everyone knows this is my job here. When I'm on the courts 
or when I'm on the court playing, I'm a competitor and I want to beat every single person whether 
they're in the locker room or across the net...
df['article_text'][1]
BASEL, Switzerland (AP), Roger Federer advanced to the 14th Swiss Indoors final of his career by beating 
seventh-seeded Daniil Medvedev 6-1, 6-4 on Saturday. Seeking a ninth title at his hometown event, and a 99th 
overall, Federer will play 93th-ranked Marius Copil on Sunday. Federer dominated the 20th-ranked Medvedev and had 
his first match-point chance to break serve again at 5-1...
df['article_text'][2]
Roger Federer has revealed that organisers of the re-launched and condensed Davis Cup gave him three days to 
decide if he would commit to the controversial competition. Speaking at the Swiss Indoors tournament where he will 
play in Sundays final against Romanian qualifier Marius Copil, the world number three said that given the 
impossibly short time frame to make a decision, he opted out of any commitment...
Now we have 2 options – we can either summarize each article individually, or we can generate a single summary for all the articles. For our purpose, we will go ahead with the latter.

 

Split Text into Sentences
Now the next step is to break the text into individual sentences. We will use the sent_tokenize( ) function of the nltk library to do this.

from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list
Let’s print a few elements of the list sentences.

sentences[:5]
Output:

['Maria Sharapova has basically no friends as tennis players on the WTA Tour.',
"The Russian player has no problems in openly speaking about it and in a recent
interview she said: 'I don't really hide any feelings too much.",
'I think everyone knows this is my job here.',
"When I'm on the courts or when I'm on the court playing,
I'm a competitor and I want to beat every single person whether they're in the
locker room or across the net.So I'm not the one to strike up a conversation about
the weather and know that in the next few minutes I have to go and try to win a tennis match.",
"I'm a pretty competitive girl."]
 

Download GloVe Word Embeddings
GloVe word embeddings are vector representation of words. These word embeddings will be used to create vectors for our sentences. We could have also used the Bag-of-Words or TF-IDF approaches to create features for our sentences, but these methods ignore the order of the words (and the number of features is usually pretty large).

We will be using the pre-trained Wikipedia 2014 + Gigaword 5 GloVe vectors available here. Heads up – the size of these word embeddings is 822 MB.

!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
Let’s extract the words embeddings or word vectors.

# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
len(word_embeddings)
400000
We now have word vectors for 400,000 different terms stored in the dictionary – ‘word_embeddings’.

 

Text Preprocessing
It is always a good practice to make your textual data noise-free as much as possible. So, let’s do some basic text cleaning.

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]
Get rid of the stopwords (commonly used words of a language – is, am, the, of, in, etc.) present in the sentences. If you have not downloaded nltk-stopwords, then execute the following line of code:

nltk.download('stopwords')
Now we can import the stopwords.

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
Let’s define a function to remove these stopwords from our dataset.

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
We will use clean_sentences to create vectors for sentences in our data with the help of the GloVe word vectors.

 

Vector Representation of Sentences
# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
Now, let’s create vectors for our sentences. We will first fetch vectors (each of size 100 elements) for the constituent words in a sentence and then take mean/average of those vectors to arrive at a consolidated vector for the sentence.

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)
Note: For more text preprocessing best practices, you may check our video course, Natural Language Processing (NLP) using Python.

 

Similarity Matrix Preparation
The next step is to find similarities between the sentences, and we will use the cosine similarity approach for this challenge. Let’s create an empty similarity matrix for this task and populate it with cosine similarities of the sentences.

Let’s first define a zero matrix of dimensions (n * n).  We will initialize this matrix with cosine similarity scores of the sentences. Here, n is the number of sentences.

# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
We will use Cosine Similarity to compute the similarity between a pair of sentences.

from sklearn.metrics.pairwise import cosine_similarity
And initialize the matrix with cosine similarity scores.

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
 

Applying PageRank Algorithm
Before proceeding further, let’s convert the similarity matrix sim_mat into a graph. The nodes of this graph will represent the sentences and the edges will represent the similarity scores between the sentences. On this graph, we will apply the PageRank algorithm to arrive at the sentence rankings.

import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
 

Summary Extraction
Finally, it’s time to extract the top N sentences based on their rankings for summary generation.

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 10 sentences as the summary
for i in range(10):
  print(ranked_sentences[i][1])
When I'm on the courts or when I'm on the court playing, I'm a competitor and I want to beat every single person 
whether they're in the locker room or across the net.So I'm not the one to strike up a conversation about the 
weather and know that in the next few minutes I have to go and try to win a tennis match.

Major players feel that a big event in late November combined with one in January before the Australian Open will 
mean too much tennis and too little rest.

Speaking at the Swiss Indoors tournament where he will play in Sundays final against Romanian qualifier Marius 
Copil, the world number three said that given the impossibly short time frame to make a decision, he opted out of 
any commitment.

"I felt like the best weeks that I had to get to know players when I was playing were the Fed Cup weeks or the 
Olympic weeks, not necessarily during the tournaments.

Currently in ninth place, Nishikori with a win could move to within 125 points of the cut for the eight-man event 
in London next month.

He used his first break point to close out the first set before going up 3-0 in the second and wrapping up the 
win on his first match point.
The Spaniard broke Anderson twice in the second but didn't get another chance on the South African's serve in the 
final set.

"We also had the impression that at this stage it might be better to play matches than to train.

The competition is set to feature 18 countries in the November 18-24 finals in Madrid next year, and will replace 
the classic home-and-away ties played four times per year for decades.

Federer said earlier this month in Shanghai in that his chances of playing the Davis Cup were all but non-existent.
And there we go! An awesome, neat, concise, and useful summary for our articles.
