#import nltk
import datetime
import json
import pickle
import math
import numpy as np

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

class WordDay(Word):
    def __init__(self, word:str):
        self.word = word
        self.distribution = [0]*61

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

def containing_tweets(word:str)->[dict]:
    out = []
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                content = tweet["text"]
                if content.lower().count(word) > 0:
                    out.append(tweet)
    true_base = 29
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                content = tweet["text"]
                if content.count(word) > 0:
                    out.append(tweet)
    return out
                

def save_word(word: Word):
    with open(word.word+".pkl", "wb") as f:
        pickle.dump(word,f, protocol = pickle.HIGHEST_PROTOCOL)

def save_words(words : [Word], filename : str):
    with open(filename, "wb") as f:
        pickle.dump(words, f, protocol = pickle.HIGHEST_PROTOCOL)

def save_word_day(word: WordDay):
    with open(word.word+"day.pkl", "wb") as f:
        pickle.dump(word,f, protocol = pickle.HIGHEST_PROTOCOL)

def open_word(word:str)->[Word]:
    with open(word+".pkl", "rb") as f:
        word_obj =  pickle.load(f)
    return word_obj

def open_word_day(word:str)->[WordDay]:
    with open(word+"day.pkl", "rb") as f:
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
    true_base = 29
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
                content = tweet["text"]
                word_obj.distribution[true_base + i][int(tweet_time.strftime("%H"))] += content.lower().count(word)
    save_word(word_obj)

def search_word_day(word:str)->Word:
    word_obj = WordDay(word)
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
                content = tweet["text"]
                word_obj.distribution[i-1] += content.lower().count(word)
    true_base = 29
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                tweet_time =  datetime.datetime.strptime(tweet["created_at"],"%Y-%m-%dT%H:%M:%S.000Z")
                content = tweet["text"]
                word_obj.distribution[true_base+i] += content.lower().count(word)
    return word_obj

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

def normalise_word_day(word: WordDay)->WordDay:
    mean = np.mean(word.distribution)
    sd = np.std(word.distribution)
    for i in range(61):
        word.distribution[i] = (word.distribution[i]-mean)/sd
    return word

def residual_base():
    out = []
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            out.append(len(tweets))
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            out.append(len(tweets))
    mean = np.mean(out)
    sd = np.std(out)
    for i in range(61):
        out[i] = (out[i]-mean)/sd
    return out

def residual_pattern(base: [int], word:WordDay)->WordDay:
    for i in range(61):
        word.distribution[i] -= base[i]
    return word
        
def similar_words(tweets:[dict], input_word:str)->dict:
    unwanted = nltk.corpus.stopwords.words("italian")
    out = {}
    for tweet in tweets:
        words = split_tweet(tweet["text"])
        for word in words:
            if word not in unwanted and word.lower().count(input_word) == 0:
                if word in out.keys():
                    out[word] += 1
                else:
                    out[word] = 1
    return out

def extra_tweets(start_word:str)->int:
    unwanted = nltk.corpus.stopwords.words("italian")+nltk.corpus.stopwords.words("english")+["I'm", "w/","Milan","AC"]
    word_dict = {}
    tweets = containing_tweets(start_word)
    seen_ids = set()
    for tweet in tweets:
        words = split_tweet(tweet["text"])
        seen_ids.add(tweet["id"])
        for word in words:
            if word not in unwanted and word.lower().count(start_word) == 0:
                if word in word_dict.keys():
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    word_set = set()
    extra = 0
    for key in word_dict.keys():
        if word_dict[key] > 6:
            word_set.add(key)
    for word in word_set:
        print(word+":"+ str(sum(search_word_day(word.lower()).distribution)))
    print(word_dict)
    #return extra
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                if not tweet["id"] in seen_ids:
                    for word in word_set:
                        if tweet["text"].count(word) > 0:
                            extra += 1
                            break
    true_base = 29
    for i in range(1,32):
        with open(str(i).zfill(2)+"1213tweets.json","r") as f:
            tweets = json.load(f)["tweets"]
            for tweet in tweets:
                if not tweet["id"] in seen_ids:
                    for word in word_set:
                        if tweet["text"].count(word) > 0:
                            extra += 1
                            break
    return extra

if __name__ == "__main__":
    #nltk.download("stopwords")
    word = search_word_day("bob dylan")
    word = normalise_word_day(word)
    base_pattern = residual_base()
    #word = open_word_day("#inter")
    word = residual_pattern(base_pattern,word)
    save_word_day(word)
    #tweets = containing_tweets("internazionale")
    #print(similar_words(tweets, "internazionale"))
    #print(extra_tweets("internazionale"))




