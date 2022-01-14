import json
import matplotlib.pyplot as plt

class IncorrectLengthException(Exception):
    """Exception thrown if a list is not at the correct resolution"""
    pass

def get_hourly_frequency(word:str)->[int]:
    """Gets the usage of a word over time
    
    Args
    word - the word to graph usage of
    
    Returns
    List of ints for the usage of a word over each hour of the time period"""
    hours = []
    tweet_dict ={}
    for i in range(1,31):
        day = [0]*24
        with open(str(i).zfill(2)+"1113tweets.json", "r") as f:
            tweet_dict = json.load(f)
        tweets = tweet_dict["tweets"]
        for tweet in tweets:
            tweet_content = tweet["text"]
            tweet_split = tweet_content.split(" ")
            hour = int(tweet["created_at"][11:13])
            for tweet_word in tweet_split:
                tweet_word = ''.join(e for e in tweet_word if e.isalnum())
                if tweet_word == word:
                    day[hour] +=1
        hours = hours + day
    return hours

def get_base():
    """Gets the base frequency pattern over a day
    
    Returns
    List of integers for the number of tweets"""
    hours = [0] *24
    tweet_dict ={}
    for i in range(1,31):
        with open(str(i).zfill(2)+"1113tweets.json", "r") as f:
            tweet_dict = json.load(f)
        tweets = tweet_dict["tweets"]
        for tweet in tweets:
            hour = int(tweet["created_at"][11:13])
            hours[hour] += 1
    return hours

def compare_day(day:[int], baseline:[int])->[float]:
    """Compares each hour of a day to the baseline

    Args
    day - list of integers to compare to the baseline
    baseline - baseline for comparison
    
    Returns
    List of floats representing the compared version"""
    if len(day) != 24 or len(baseline) != 24:
        raise IncorrectLengthException
    out = []
    for i in range(24):
        out.append(day[i]/baseline[i])
    return out

if __name__ == "__main__":
    day = get_hourly_frequency("di")
    baseline = get_base()
    y = compare_day(day[0:24],baseline)
    x = range(744)
    for i in range(1,31):
        print(i)
        y = y + compare_day(day[i*24:i*24+24],baseline)
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()


