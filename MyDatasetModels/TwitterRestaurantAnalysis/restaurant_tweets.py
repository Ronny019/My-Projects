# Create the following files in the directory "C:\Twitter API" with appropriate information
# Access Token, Access Token Secret, API Key, API Key Secret , App Name

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''
with open('C:\Twitter API\API Key.txt','r') as key:
    for line in key:
        CONSUMER_KEY = line

with open('C:\Twitter API\API Key Secret.txt','r') as secret:
    for line in secret:
        CONSUMER_SECRET = line

with open('C:\Twitter API\Access Token.txt','r') as key:
    for line in key:
        ACCESS_TOKEN = line

with open('C:\Twitter API\Access Token Secret.txt','r') as secret:
    for line in secret:
        ACCESS_TOKEN_SECRET = line



from twython import TwythonStreamer
# appending data to a global variable is pretty poor form
# but it makes the example much simpler
tweets = []
class MyStreamer(TwythonStreamer):
    """our own subclass of TwythonStreamer that specifies
    how to interact with the stream"""
    def on_success(self, data):
        """what do we do when twitter sends us data?
        here data will be a Python dict representing a tweet"""
        # only want to collect English-language tweets
        if data['lang'] == 'en':
            tweets.append(data)
            print("received tweet #", len(tweets))
        # stop when we've collected enough
        if len(tweets) >= 10:
            self.disconnect()
    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()


restaurant = input('Enter restaurant name  ')
stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# starts consuming public statuses that contain the keyword (restaurant name)
stream.statuses.filter(track=restaurant)

import natural_language_processing as nlp
print("-----------------------------------------------------------")


from collections import Counter
top_hashtags = Counter(hashtag['text'].lower()
                        for tweet in tweets
                        for hashtag in tweet["entities"]["hashtags"])

print(top_hashtags.most_common(5))

tweet_texts = list(nlp.convert_into_proper_form(tweet["text"]) for tweet in tweets)

# Creating the Bag of Words model
X_tweets = nlp.cv.transform(tweet_texts).toarray()

predicted_ratings = nlp.classifier.predict(X_tweets)

for i,tweet in enumerate(tweets):
    print(str(tweet["text"]))
    print(predicted_ratings[i])
    print()
    print()