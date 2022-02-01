import nltk
import datetime
import json
import pickle

class Word:
    """Class for representing a word"""
    def __init__(self, word:str):
        """Constructor
        
        Args
        self - instance identifier
        word - word being represented"""
        self.word = word
        self.distribution = []
        for i in range(61):
            self.distribution.append([0]*24)

    def increment_ind(self, ind1 : int, ind2: int):
        self.distribution[ind1][ind2] = self.distribution[ind1][ind2] + 1

    def __eq__(self, value):
        return self.word == value


    def __repr__(self):
        count = 0
        for i in range(61):
            count += sum(self.distribution[i])
        return self.word + " : " + str(count)

def split_tweet(tweet:str)->[str]:
    """Splits a tweet into the words it contains and filters out links
    
    Args
    tweet - tweet to split
    
    Returns
    List of words the tweet contains"""
    tweet_words = tweet.split(" ")
    out_words = []
    for word in tweet_words:
        word = ''.join(e for e in word if e.isalnum()).lower()
        if word[0:4] == "http" or len(word) == 1 or len(word) == 0:
            pass
        else:
            out_words.append(word)
    return out_words

def load_words()->[Word]:
    unwanted = nltk.corpus.stopwords.words("italian")
    processed = []
    for i in range(1,31):
        print(i)
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
        for tweet in tweets:
            words = split_tweet(tweet["text"])
            tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
            for word in words:
                if word in unwanted:
                    pass
                elif word in processed:
                    ind = processed.index(word)
                    processed[ind].increment_ind(i-1,int(tweet_time.strftime("%H")))
                else:
                    formatted = Word(word)
                    print(formatted.distribution[i-1][int(tweet_time.strftime("%H"))])
                    formatted.increment_ind(i-1,int(tweet_time.strftime("%H")))
                    processed.append(formatted)
    for i in range(1,32):
        print(i)
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
        for tweet in tweets:
            words = split_tweet(tweet["text"])
            tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
            for word in words:
                if word in unwanted:
                    pass
                elif word in processed:
                    ind = processed.index(word)
                    processed[ind].increment_ind(i-1,int(tweet_time.strftime("%H")))
                else:
                    formatted = Word(word)
                    print(formatted.distribution[i-1][int(tweet_time.strftime("%H"))])
                    formatted.increment_ind(i-1,int(tweet_time.strftime("%H")))
                    processed.append(formatted)
    return processed

def save_words(words : [Word]):
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f, protocol = pickle.HIGHEST_PROTOCOL)

def open_words()->[Word]:
    with open("words.pkl", "rb") as f:
        word =  pickle.load(f)
    return word

def tweet_count():
    count = 0
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
        count += len(tweets)
        print(count)
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
        count += len(tweets)
        print(count)


if __name__ == "__main__":
    #nltk.download("stopwords")
    words = load_words()
    save_words(words)
    #tweet_count()


