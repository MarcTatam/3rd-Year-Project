from tweet_processing import Word, open_word, normalise_word, WordDay, open_word_day, containing_tweets, similar_words, residual_base, residual_pattern, normalise_word_day, search_word_day
import matplotlib.pyplot as plt
import project_utils as util
import pandas as pd
import numpy as np

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
    ax.xticks(np.arange(-4, 6+1, 2))
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
    ax.set_yticks(np.arange(-4, 6+1, 2))
    ax.set_ybound(-4,7)
    ax.set_ylabel("Word Usage")
    ax.set_xlabel("Days Since Project Epoch")
    ax.set_title(word)
    plt.savefig(word+'day.png')

def plot_related_frequency(word :str):
    """Plots word frequency of related words
    
    Args
    word - base word to plot related words"""
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

if __name__ == "__main__":
    #graph_word_day_me("internazionale", ["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan"])
    graph_word_day_me("ac milan", ["Inter vs Milan","Milan vs Genoa","Milan vs Fiorentina","Milan vs Roma", "Milan vs Ajax"])
    #graph_word_day_me("vs", ["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan","Milan vs Genoa","Milan vs Fiorentina","Milan vs Roma"])
    #graph_word_day_me("arctic monkeys", ["Arctic Monkeys"])
    #plot_related_time("internazionale",["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan"])
    #graph_word_day_me("giuseppe meazza", ["Inter vs Livorno", "Inter vs Sampdoria", "Inter vs Parma", "Inter vs Milan","Milan vs Genoa","Milan vs Fiorentina","Milan vs Roma","Milan vs Ajax", "Italy vs Germany"])
    #graph_word_day_me("pixies", ["Pixies"])
    #plot_related_frequency("internazionale")
    #graph_alcatraz()