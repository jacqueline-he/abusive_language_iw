import os
import re
import csv
import pickle 

path = '/Users/jacquelinehe/Desktop/TwitterAAE-full-v1/twitteraae_all.tsv'
# path = '/Users/jacquelinehe/Desktop/TwitterAAE-full-v1/twitter_all.tsv'

def read_data(file_name, tweet_index, dem, tab_separated = False):
    result = []
    with open(path, 'r', errors='ignore') as data_file:
        reader = csv.reader(data_file) if not tab_separated else csv.reader(data_file, delimiter='\t')
        for row in reader:
            if row[dem] != 'null' and float(row[dem]) > 0.8:
                tweet = row[tweet_index]
                # Remove hashtags, mentions, and links in tweets
                tweet = re.sub(r"(?:\@|\#|https?\://)\S+", "", tweet)
                tweet = re.sub(r"(?:\@|\#|https?\://)\S+", "", tweet)
                tweet = tweet.replace(r'&amp;?', r'and')
                tweet = tweet.replace(r'&lt;', r'<')
                tweet = tweet.replace(r'&gt;', r'>')
                tweet = deEmojify(tweet)
                backslash = tweet.find("\\")
                if backslash != -1:
                  tweet = tweet[0:backslash]
                result.append(tweet)
    return result

data = read_data(path, 5, -1, tab_separated=True)
label = 'sae' #aave
with open(label + '.pkl', 'wb') as f:
    pickle.dump(sae_data, f)
