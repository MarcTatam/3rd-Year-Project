import nltk
import datetime
import json
import pickle
import math

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
                if "http://" in word:
                    pass
                elif word in unwanted:
                    pass
                elif word in processed:
                    ind = processed.index(word)
                    processed[ind].increment_ind(i-1,int(tweet_time.strftime("%H")))
                else:
                    formatted = Word(word)
                    formatted.increment_ind(i-1,int(tweet_time.strftime("%H")))
                    processed.append(formatted)
    for i in range(1,32):
        print(i)
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
        for tweet in tweets:
            words = split_tweet(tweet["text"])
            tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
            for word in words:
                if "http://" in word:
                    pass
                elif word in unwanted:
                    pass
                elif word in processed:
                    ind = processed.index(word)
                    processed[ind].increment_ind(i-1,int(tweet_time.strftime("%H")))
                else:
                    formatted = Word(word)
                    formatted.increment_ind(i-1,int(tweet_time.strftime("%H")))
                    processed.append(formatted)
    return processed

def save_word(word: Word):
    with open(word.word+".pkl", "wb") as f:
        pickle.dump(word,f, protocol = pickle.HIGHEST_PROTOCOL)

def save_words(words : [Word], filename : str):
    with open(filename, "wb") as f:
        pickle.dump(words, f, protocol = pickle.HIGHEST_PROTOCOL)

def open_word(word:str)->[Word]:
    with open(word+".pkl", "rb") as f:
        word_obj =  pickle.load(f)
    return word_obj

def open_words(filename)->[Word]:
    with open(filename, "rb") as f:
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

def normalise(words:[Word])->[Word]:
    z = 0
    for word in words:
        z += 1
        print(z)
        total = 0
        for i in range(61):
            total += sum(word.distribution[i])
        if total > 1 :
            average = total/(61*24)
            sd_sum = 0
            for i in range(61):
                for j in range(24):
                    sd_sum = (word.distribution[i][j] - average)**2
            sd = math.sqrt(sd_sum/(61*24))
            for i in range(61):
                for j in range(24):
                    word.distribution[i][j] = (word.distribution[i][j] - average)/sd
    return words

def search_word(word:str)->Word:
    word_obj = Word(word)
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
                content = tweet["text"]
                word_obj.distribution[i-1][int(tweet_time.strftime("%H"))] += content.lower().count(word)
    true_base = 30
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
                content = tweet["text"]
                word_obj.distribution[true_base + i-1][int(tweet_time.strftime("%H"))] += content.lower().count(word)
    save_word(word_obj)

def normalise_word(word: Word):
    total = 0
    for i in range(61):
        total += sum(word.distribution[i])
    if total > 1 :
        average = total/(61*24)
        sd_sum = 0
        for i in range(61):
            for j in range(24):
                sd_sum = (word.distribution[i][j] - average)**2
        sd = math.sqrt(sd_sum/(61*24))
        for i in range(61):
            for j in range(24):
                word.distribution[i][j] = (word.distribution[i][j] - average)/sd
    return word

if __name__ == "__main__":
    #nltk.download("stopwords")
    search_word("arctic monkeys")
    print(normalise_word(open_word("arctic monkeys")))



