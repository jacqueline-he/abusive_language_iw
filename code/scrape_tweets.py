import wget
import os
import pandas as pd
import tweepy
import time
import datetime

url = 'https://raw.githubusercontent.com/ENCASEH2020/hatespeech-twitter/master/hatespeech_labels.csv'

# Download dataset if necessary 
if not os.path.exists('./hatespeech_labels.csv'):
  print('Downloading hatespeechtwitter.csv...')
  wget.download(url, './hatespeech_labels.csv')

# Load dataset into a pandas dataframe
df = pd.read_csv("./hatespeech_labels.csv", header=0, names=['tweet_id', 'maj_label'])

# Report number of tweets to scrape
print('Number of tweets: {:,}\n'.format(df.shape[0]))

ids = df.tweet_id.values
labels = df.maj_label.values

df.sample(10)

# Create OAuthHandler instance
consumer_key = 'RqmgrI62hjkRFSvg1qQoWnGbW'
consumer_secret = 'If6xVsxgjul18TLrA4sk5DQtnrO7QG7dJZAQzm3QRGb0iNE189'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

access_token = '1162886569889374208-ZAo6oM04JbdAOXzggDiEr8eOZSuyZD'
access_token_secret = 'gy39qiyrhcY70iNpEJFwN2QUQCmKZV2CQT9Ru4JrgIrOv'

auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

scraped_ids = []
scraped_text = []
scraped_labels = []

def get_tweets_list(batch_size, start_index):
  i = 0
  count = 0
  while i < batch_size:
    if (start_index + i) > 57670:
      return count
    
    tweet_id = ids[start_index + i]
    print(tweet_id)
    try:
      tweet = api.get_status(tweet_id)
      test = ""
      try:
        text = tweet.extended_tweet['full_text']
      except AttributeError as e:
        text = tweet.text
      scraped_ids.append(tweet_id)
      scraped_text.append(text)
      scraped_labels.append(labels[start_index + i])
      # print('%s,%s' % (tweet_id, tweet.text.encode('UTF-8')))
      count += 1
    except tweepy.TweepError as te:
      pass
      # traceback.print_exc(file=sys.stderr)
    i += 1

  # Write data to .csv file and download
  d = {'ids': scraped_ids, 'text': scraped_text, 'labels': scraped_labels}
  label_str = "scraped_tweets-" + str(start_index)+".csv"
  pd.DataFrame(data=d).to_csv(label_str, index=False, float_format='%.17f') 
  
  return count

def format_time(elapsed):
  elapsed_rounded = int(round((elapsed)))
  return str(datetime.timedelta(seconds=elapsed_rounded))

batch_size = 200

# Total time to scrape all tweets
total_t0 = time.time()
num_batch = 1

i = 0
count = 0
# Scrape tweets in batches
while i < 100:
  count += get_tweets_list(batch_size, i)
  elapsed = format_time(time.time() - total_t0)

  # Report progress.
  print("On batch ", num_batch, " with time elapsed: ", elapsed, " and count : ", count)

  num_batch += 1
  i += batch_size

# Write data to .csv file and download  
d = {'ids': scraped_ids, 'text': scraped_text, 'labels': scraped_labels}
pd.DataFrame(data=d).to_csv("scraped_tweets_total.csv", index=False)
