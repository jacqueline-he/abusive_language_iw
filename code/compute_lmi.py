import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import pickle
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import nltk
from nltk.util import ngrams
import re
import unicodedata
import seaborn as sns

# For visualization 
from IPython.display import display, HTML, display_html
CSS = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(CSS))




# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

path_train = 'drive/My Drive/COS397 IW/data-train_NEW.csv'
df_train = pd.read_csv(path_train, lineterminator='\n', index_col=0)
df_train.sample(20)

corpus = df_train.ProcessedText.values

# Clean up text data - lemmatize, remove stopwords, convert to lowercase,
# tokenize
def basic_clean(text):
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

# Convert text to bigrams list
def bigrams(text):
  token = basic_clean(text)
  return list(ngrams(token, 2)) 


print(bigrams("hi, this is a test."))

def populate_label_bigrams():
  n_bigrams = defaultdict(int)
  a_bigrams = defaultdict(int)
  s_bigrams = defaultdict(int)
  h_bigrams = defaultdict(int)
  total_bigrams = defaultdict(int)

  num_bigrams_n = 0
  num_bigrams_a = 0
  num_bigrams_s = 0
  num_bigrams_h = 0
  for index, row in df_train.iterrows():
    b_list = bigrams(row.ProcessedText)
    if row.labels == 'normal':
      num_bigrams_n += len(b_list)
      for b in b_list:
        n_bigrams[b] += 1
        total_bigrams[b] += 1
    elif row.labels == 'abusive':
      num_bigrams_a += len(b_list)
      for b in b_list:
        a_bigrams[b] += 1
        total_bigrams[b] += 1
    elif row.labels == 'spam':
      num_bigrams_s += len(b_list)
      for b in b_list:
        s_bigrams[b] += 1
        total_bigrams[b] += 1
    else:
      num_bigrams_h += len(b_list)
      for b in b_list:
        h_bigrams[b] += 1
        total_bigrams[b] += 1
  return total_bigrams, n_bigrams, a_bigrams, s_bigrams, h_bigrams, num_bigrams_n, num_bigrams_a, num_bigrams_s, num_bigrams_h

total_bigrams, n_bigrams, a_bigrams, s_bigrams, h_bigrams, num_bigrams_n, num_bigrams_a, num_bigrams_s, num_bigrams_h = populate_label_bigrams

n_lmi = defaultdict(int)
a_lmi = defaultdict(int)
s_lmi = defaultdict(int)
h_lmi = defaultdict(int)

def fill_lmi(bigram_list, lmi_list, total_bigram_len):
  for key in bigram_list:
  # p(w,c)
    pwc = float(bigram_list.get(key) / total_bigram_len)
    pcw = float(bigram_list.get(key) / total_bigrams.get(key))
    pc = float(len(bigram_list) / total_bigram_len)
    res = 0
    if float(pcw/pc) > 0:
      res = pwc*math.log(pcw/pc)
    lmi_list[key] = res*10**6


D = len(total_bigrams)
fill_lmi(n_bigrams, n_lmi, D)
fill_lmi(a_bigrams, a_lmi, D)
fill_lmi(s_bigrams, s_lmi, D)
fill_lmi(h_bigrams, h_lmi, D)

def print_lmi_score(lmi_type, lmi_list):
  label = lmi_type + " BIGRAM"
  print ("{:<20} {:<80}".format(label, 'LMI SCORE')) 
  # print each data item. 
  for key, value in sorted(lmi_list.items(), key=lambda item: item[1], reverse=True)[0:15]: 
    (word1, word2) = key
    print ("{:<20} {:<80}".format(word1 + " " + word2, value))

print_lmi_score('NORMAL', n_lmi)
print_lmi_score('ABUSIVE', a_lmi)
print_lmi_score('SPAM', s_lmi) 
print_lmi_score('HATE SPEECH', h_lmi)

def visualize_lmi_score(lmi_type, lmi_list):
  label = lmi_type + " BIGRAM"
  sorted_list = sorted(lmi_list.items(), key=lambda item: item[1], reverse=True)[0:15]
  sorted_phrase = []
  sorted_score = []

  for k, v in sorted_list:
    (phrase, score) = k
    sorted_n_phrase.append(phrase + ' ' +  score)
    sorted_n_score.append(v)
  d = {label: sorted_n_phrase, 'LMI Score': sorted_n_score}
  df = pd.DataFrame(data=d)
  cm = sns.light_palette('green', as_cmap=True)
  return df.style.background_gradient(cmap=cm, low=0, high=1, axis=0).hide_index()

nv = visualize_lmi_score('NORMAL', n_lmi)
av = visualize_lmi_score('ABUSIVE', a_lmi)
sv = visualize_lmi_score('SPAM', s_lmi) 
hv = visualize_lmi_score('HATE SPEECH', h_lmi)

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.render()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

display_side_by_side(nv, av, sv, hv)
