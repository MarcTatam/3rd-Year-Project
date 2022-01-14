import matplotlib.pyplot as plt
import matplotlib.dates as pltdate
import datetime
import json
import numpy as np
def graph_tweets_per_day():
    """Graphs the number of tweets per day from the 1st of December 2013 to the 31st of December 2013"""
    xs = []
    ys = []
    tweet_dict ={}
    for i in range(1,32):
        xs.append(datetime.datetime(2013,12,i,00,00))
        with open(str(i).zfill(2)+"1213tweets.json", "r") as f:
            tweet_dict = json.load(f)
        ys.append(len(tweet_dict["tweets"]))

    dates = pltdate.date2num(xs)
    fig, ax = plt.subplots()
    ax.plot_date(dates,ys,linestyle="-")
    ax.set_title("Number of Tweets per Day")
    ax.set_ylabel("Number of Tweets")
    ax.set_xlabel("Date")
    plt.show()

def graph_tweets_per_hour():
    """Graphs the number of tweets per hour of the day from the 1st of December 2013 to the 31st of December 2013"""
    xs = []
    for i in range(0,24):
        xs.append(i)
    ys = [0] *24
    tweet_dict ={}
    for i in range(1,32):
        print(i)
        with open(str(i).zfill(2)+"1213tweets.json", "r") as f:
            tweet_dict = json.load(f)
        tweets = tweet_dict["tweets"]
        for tweet in tweets:
            hour = int(tweet["created_at"][11:13])
            ys[hour] += 1
    fig, ax = plt.subplots()
    ax.plot(xs,ys,linestyle="-")
    ax.set_title("Number of Tweets per hour of the day")
    ax.set_ylabel("Number of Tweets")
    ax.set_xlabel("Hour")
    plt.xticks(np.arange(0, 24, 1.0))
    plt.show()

def graph_word_usage(word:str):
    """Graphs the usage of a word over time
    
    Args
    word - the word to graph usage of"""
    xs = []
    ys = []
    tweet_dict ={}
    for i in range(1,32):
        print(i)
        day = [0]*24
        for j in range(0,24):
            xs.append(datetime.datetime(2013,12,i,j,00))
        with open(str(i).zfill(2)+"1213tweets.json", "r") as f:
            tweet_dict = json.load(f)
        tweets = tweet_dict["tweets"]
        for tweet in tweets:
            tweet = tweet.lower()
            tweet_content = tweet["text"]
            tweet_split = tweet_content.split(" ")
            hour = int(tweet["created_at"][11:13])
            for tweet_word in tweet_split:
                tweet_word = ''.join(e for e in tweet_word if e.isalnum())
                if tweet_word == word:
                    day[hour] +=1
        ys = ys + day
    dates = pltdate.date2num(xs)
    fig, ax = plt.subplots()
    ax.plot_date(dates,ys,linestyle="-")
    ax.set_title("Number of Times " + word + " was tweeted on an hourly timescale")
    ax.set_ylabel("Number of Tweets")
    ax.set_xlabel("Date and Time")
    plt.show()


def most_common_word()-> (str,int):
    """Find the most popular word in the month of december 2013
    
    Return
    String representing most common word and an integer for the number of times that word waas used"""
    word_dict = {}
    tweet_dict ={}
    for i in range(1,31):
        print(i)
        with open(str(i).zfill(2)+"1113tweets.json", "r") as f:
            tweet_dict = json.load(f)
        tweets = tweet_dict["tweets"]
        for tweet in tweets:
            tweet_content = tweet["text"]
            tweet_split = tweet_content.split(" ")
            for word in tweet_split:
                word = ''.join(e for e in word if e.isalnum()).lower()
                if word[0:4] == "http" or len(word) == 1 or len(word) == 0:
                    pass
                elif word in word_dict.keys():
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    highest_count = 0
    highest = ""
    print(word_dict)
    for key in word_dict.keys():
        if word_dict[key] > highest_count:
            highest_count = word_dict[key]
            print("Key: " +key)
            highest = key
    return highest, highest_count

if __name__ == "__main__":
    #graph_word_usage("christmas")
    most_common_word()
