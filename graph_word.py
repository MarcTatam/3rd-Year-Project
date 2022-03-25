from tweet_processing import Word, open_word, normalise_word, WordDay, open_word_day, containing_tweets, similar_words, residual_base, residual_pattern, normalise_word_day, search_word_day
import matplotlib.pyplot as plt
import project_utils as util
import pandas as pd

def flatten_word(word:Word)->[float]:
    """Flattens the distribution of the word into a plottable list
    
    Args
    word - Word object to be flattened
    
    Returns
    List of floats demonstrating word or phrase usage"""
    out = []
    for i in range(61):
        out += word.distribution[i]
    return out

def graph_word(word:str, event:str):
    """Graphs the usage of a word, highlighting the relevant events
    
    Args
    word - word to plot the usage of
    event - event to highlight"""
    df = pd.read_csv("Events.csv")
    fig, ax = plt.subplots()
    word_obj = open_word(word)
    for index, row in df.loc[df["Name"] == event].iterrows():
        day = util.convert_to_project(row["Date"])
        start = 24*day
        stop = 24*(day+1)-1
        ax.axvspan(start, stop, color = "salmon")
    y = flatten_word(word_obj)
    x = []
    for i in range(1,1465):
        x.append(i)
    ax.plot(x,y)
    ax.set_ylabel("Word Usage")
    ax.set_xlabel("Hours Since Project Epoch")
    plt.savefig(word+'.png')

def graph_word_me(word:str, events:[str]):
    """Graphs the usage of a word, highlighting the relevant events
    
    Args
    word - word to plot the usage of
    events - events to highlight"""
    df = pd.read_csv("Events.csv")
    fig, ax = plt.subplots()
    word_obj = open_word(word)
    for event in events:
        for index, row in df.loc[df["Name"] == event].iterrows():
            day = util.convert_to_project(row["Date"])
            start = 24*day
            stop = 24*(day+1)-1
            ax.axvspan(start, stop, color = "salmon")
    y = flatten_word(word_obj)
    x = []
    for i in range(1,1465):
        x.append(i)
    ax.plot(x,y)
    ax.set_ylabel("Word Usage")
    ax.set_xlabel("Hours Since Project Epoch")
    ax.set_title(word)
    plt.savefig(word+'.png')

def graph_word_day_me(word:str, events:[str]):
    """Graphs the usage of a word, highlighting the relevant events
    
    Args
    word - word to plot the usage of
    events - events to highlight"""
    df = pd.read_csv("Events.csv")
    fig, ax = plt.subplots()
    word_obj = open_word_day(word)
    for event in events:
        for index, row in df.loc[df["Name"] == event].iterrows():
            day = util.convert_to_project(row["Date"])
            ax.axvline(day, color = "red", linestyle = "--")
    y = word_obj.distribution
    x = []
    for i in range(61):
        x.append(i)
    ax.plot(x,y)
    ax.set_ylabel("Word Usage")
    ax.set_xlabel("Days Since Project Epoch")
    ax.set_title(word)
    plt.savefig(word+'day.png')

def plot_related_frequency(word :str):
    tweets = containing_tweets(word)
    similar_dict = similar_words(tweets, word)
    dict2 = {}
    for key in similar_dict.keys():
        if similar_dict[key] > 5:
            dict2[key] = similar_dict[key]
    x = list(dict2.keys())
    y = []
    for key in x:
        y.append(dict2[key])
    fig, ax = plt.subplots()
    ax.bar(x,y)
    plt.xticks(fontsize=8, rotation = 90)
    fig.tight_layout()
    plt.savefig(word+'freq.png')

def plot_related_time(word:str, events: [str]):
    df = pd.read_csv("Events.csv")
    fig, ax = plt.subplots()
    base = residual_base()
    tweets = containing_tweets(word)
    word_obj = search_word_day(word)
    normalised_obj = normalise_word_day(word_obj)
    residual = residual_pattern(base, normalised_obj)
    x = []
    for i in range(61):
        x.append(i)
    ax.plot(x, residual.distribution)
    for event in events:
        for index, row in df.loc[df["Name"] == event].iterrows():
            day = util.convert_to_project(row["Date"])
            ax.axvline(day, color = "red", linestyle = "--")
    ax.axhline(0, color = "black", linestyle = "--")
    similar_dict = similar_words(tweets, word)
    dict2 = {}
    lines = []
    labels = []
    for key in similar_dict.keys():
        if similar_dict[key] > 5:
            word_obj = search_word_day(key)
            normalised_obj = normalise_word_day(word_obj)
            residual = residual_pattern(base, normalised_obj)
            line, = ax.plot(x, residual.distribution)
            lines.append(line)
            labels.append(key)
    ax.set_ylabel("Residual Error from Z-Score")
    ax.set_xlabel("Time from Project Epoch")
    ax.set_title("Z-Score Residual Difference of words relating to " + word)
    plt.legend(lines,labels,prop={'size': 5}, loc="upper right")
    plt.savefig(word+'related.png')

def graph_alcatraz():
    data = pd.read_csv("AlcatrazMilano.csv",parse_dates=True)
    df = pd.read_csv("Events.csv")
    ax = data.plot()
    for index, row in df.loc[df["Venue"] == "Alcatraz"].iterrows():
        day = util.convert_to_project(row["Date"])
        ax.axvline(day, color = "red", linestyle = "--")
    ax.set_title("Number of Tweets from @AlcatrazMilano")
    ax.set_ylabel("Number of Tweets")
    ax.set_xlabel("Hours Since Project Epoch")
    plt.savefig("AlcatrazTweets.png")

if __name__ == "__main__":
    #graph_word_day_me("internazionale", ["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan"])
    #graph_word_day_me("ac milan", ["Inter vs Milan","Milan vs Genoa","Milan vs Fiorentina","Milan vs Roma", "Milan vs Ajax"])
    #graph_word_day_me("vs", ["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan","Milan vs Genoa","Milan vs Fiorentina","Milan vs Roma"])
    #graph_word_day_me("arctic monkeys", ["Arctic Monkeys"])
    #plot_related_time("internazionale",["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan"])
    #graph_word_day_me("giuseppe meazza", ["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan","Milan vs Genoa","Milan vs Fiorentina","Milan vs Roma","Milan vs Ajax", "Italy vs Germany"])
    graph_word_day_me("bob dylan", ["Bob Dylan"])
    #plot_related_frequency("internazionale")
    #graph_alcatraz()