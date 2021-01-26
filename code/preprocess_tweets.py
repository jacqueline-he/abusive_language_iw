"""preprocess_tweets.ipynb"""

import csv
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

path = 'drive/My Drive/COS397 IW/hatespeech_text_label_vote.csv'
df = pd.read_csv(path, sep='\t', header=None)

df.columns = ['text', 'labels', 'votes']

# See dimensions 
df.shape

!pip install ekphrasis
!pip install pytorch_pretrained_bert

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from pytorch_pretrained_bert import BertTokenizer

def preprocess(df):
  df['ProcessedText'] = None
  df['ProcessedText_length'] = 0
  df['ProcessedText_BERT'] = None
  df['ProcessedText_BERTbase_length'] = 0
  print(df.columns)

  text_processor = TextPreProcessor(  # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated", 'emphasis', 'censored'}, fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons])
  
        # Tweets pre-processing
  for index, row in df.iterrows():
        s = df.loc[index, 'text']
        # remove RT and USER
        s = "".join(re.sub('RT @[\w_]+: ', ' ', s))
        # df.loc[index, 'text'] = "".join(re.sub(r'&#  ;', ' ', df.loc[index, 'text']))
        # df.loc[index, 'text'] = "".join(re.sub(r' &# ;', ' ', df.loc[index, 'text']))
        # remove special characters
        s = "".join(re.sub(r'&#\d+;', ' ', s))
        # pre-processing
        s = " ".join(text_processor.pre_process_doc((s)))
        s = "".join(re.sub(r'\<[^>]*\>', ' ', s))
        # Remove non-ascii words or characters
        s = "".join([i if ord(i) < 128 else '' for i in s])
        s = s.replace(r'_[\S]?',r'')
        s = s.replace(r'[ ]{2, }',r' ')
        # Remove &, < and >
        s = s.replace(r'&amp;?', r'and')
        s = s.replace(r'&lt;', r'<')
        s = s.replace(r'&gt;', r'>')
        # Insert space between words and punctuation marks
        s = s.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
        s = s.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
        # Calculate text length for later use in LSTM
        s_length = len(s.split())
        # save ProcessedText and ProcessedText_length in final df
        df.loc[index, 'ProcessedText'] = s.strip()
        df.loc[index, 'ProcessedText_length'] = s_length

    # Drop texts with length <=2 and drop duplicates
  df = df[df['ProcessedText_length'] > 2]
  df = df.drop_duplicates(subset=['ProcessedText'])

    # BERT preprocess
  df['ProcessedText_BERT'] = '[CLS] ' + df.ProcessedText
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  df['ProcessedText_BERTbase_length'] = [len(tokenizer.tokenize(sent)) for sent in df.ProcessedText_BERT]
    # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # df['ProcessedText_BERTlarge_length'] = [len(tokenizer.tokenize(sent)) for sent in df.ProcessedText_BERT]

  label_dict = dict()
  for i, l in enumerate(list(df.labels.value_counts().keys())):
     label_dict.update({l: i})

  df['Mapped_label'] = [label_dict[label] for label in df.labels]
  return df

df_preprocessed = preprocess(df)
df_preprocessed.to_csv('tweets_preprocessed_NEW.csv', index=False)
