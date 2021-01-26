import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Some simple data visualizations. 
# Dataset scraped from Founta et al. (2018)

path = 'drive/My Drive/COS397 IW/tweets_preprocessed_NEW.csv'
df = pd.read_csv(path, lineterminator='\n')

labels = df.labels.values

ab_ct = 0
hs_ct = 0
sp_ct = 0
nm_ct = 0

for lab in labels:
  if lab == 'abusive':
    ab_ct = ab_ct + 1
  elif lab == 'spam':
    sp_ct = sp_ct + 1
  elif lab == 'normal':
    nm_ct = nm_ct + 1
  else:
     hs_ct = hs_ct + 1

print('Abusive language:', ab_ct)
print('Hate speech:', hs_ct)
print('Spam:', sp_ct)
print('Normal:', nm_ct)

print(len(labels))

sizes = [ab_ct, hs_ct, sp_ct, nm_ct]
chart_labels = ['Abusive Language (19900)', 'Hate Speech (4040)', 'Spam (12308)', 'Normal (50205)']
fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(0.3,0,1,1)

theme = plt.get_cmap('hsv')
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightblue']
ax.pie(sizes, colors = colors, labels = chart_labels, autopct='%1.1f%%', radius = 1800, shadow=True, startangle=140)
ax.axis('equal')

plt.show()

pd.set_option('display.max_colwidth', -1)
df.sample(10)

df.head()

from textblob import TextBlob
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud

labels = df.text.values
tweets = df.text.values

corpus = []

hs_corpus = []
ab_corpus = []
sp_corpus = []
nm_corpus = []

for i in range(0, 10000):  
    tweet = tweets[i]
    tweet = tweet.lower()
    tweet = re.sub(r'https://t.co/\S+', '', tweet) 
    tweet = re.sub('rt', '', tweet)  
    tweet = tweet.replace('amp', ' and ')
    tweet = re.sub('[^a-zA-Z0-9]', ' ', tweet)
    tweet = tweet.split()
    tweet = [word for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)

    label = labels[i]

    if label == 'abusive':
      ab_corpus.append(tweet)
    elif label == 'spam':
      sp_corpus.append(tweet)
    elif label == 'normal':
      nm_corpus.append(tweet)
    else:
      hs_corpus.append(tweet)

    corpus.append(tweet)

corpuses = [corpus, hs_corpus, ab_corpus, sp_corpus, nm_corpus]

# WordCloud visualization
all_words = ' '.join([text for text in nm_corpus])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color='white').generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Hate Speech Tweets')
plt.axis('off')
plt.show()

# Term Frequency - TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=10000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(tweets)

mentions = []
mention_pattern = re.compile(r"@[a-zA-Z_]+")
mention_matches = list(df['text'].apply(mention_pattern.findall))
mentions_dict = {}
for match in mention_matches:
    for singlematch in match:
        if singlematch not in mentions_dict.keys():
            mentions_dict[singlematch] = 1
        else:
            mentions_dict[singlematch] = mentions_dict[singlematch]+1

# Create an ordered list of tuples with the most mentioned users and 
# the number of times they have been mentioned
mentions_ordered_list =sorted(mentions_dict.items(), key=lambda x:x[1])
mentions_ordered_list = mentions_ordered_list[::-1]
# Pick the 20 top mentioned users to plot and separate the previous 
# list into two lists: one with the users and one with the values
mentions_ordered_values = []
mentions_ordered_keys = []
for item in mentions_ordered_list[0:20]:
    mentions_ordered_keys.append(item[0])
    mentions_ordered_values.append(item[1])

fig, ax = plt.subplots(figsize = (10,10))
y_pos = np.arange(len(mentions_ordered_values))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'g']
ax.barh(y_pos ,list(mentions_ordered_values)[::-1], align='center', edgecolor = 'black', color = colors, linewidth=1)
ax.set_yticks(y_pos )
ax.set_yticklabels(list(mentions_ordered_keys)[::-1])
ax.set_xlabel("Nº of mentions")
ax.set_title("Most mentioned accounts", fontsize = 20)

plt.show()

total_avg = sum( map(len, tweets) ) / len(tweets)
print('Average char. count in dataset: ', total_avg)
