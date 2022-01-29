from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken="-"
asecret=""

class listener(StreamListener):

    def on_data(self, data):
        
        
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet,sentiment_value,confidence)
        time.sleep(1)
        print("NO")

        #if confidence*100 >= 80:
        output = open("Live Sentiment Analysis/twitter-out.txt","a")
        output.write(sentiment_value)
        output.write('\n')
        output.close()
        return True
        

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
