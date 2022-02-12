from tweet_processing import Word, open_word, normalise_word
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

if __name__ == "__main__":
    graph_word_me("skrillex", ["Skrillex"])