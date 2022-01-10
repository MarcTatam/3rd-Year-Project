import tweetdata as td
import matplotlib.pyplot as plt
import math

def get_normal_curve():
    """Get normalised tweet pattern"""
    hours = td.get_base()
    maxx = max(hours)
    minn = min(hours)
    scaled = [x/maxx for x in hours]
    return scaled

def scale_curve(word_distro:[int], base_curve:[int])->[float]:
    out = []
    mean_base = sum(base_curve)/24
    for i in range(int(len(word_distro)/24)):
        temp = word_distro[i:i+24]
        mean = sum(temp)/24
        if mean == 0:
            scale_factor = 0
        else:
            scale_factor = mean_base/mean
        for i in range(len(temp)):
            temp[i] = temp[i]*scale_factor
        out = out + temp
    return out

def compare():
    word_curve = td.get_hourly_frequency("milano")
    base = td.get_base()
    word_curve = scale_curve(word_curve,base)
    errors = []
    for i in range(len(word_curve)):
        errors.append(math.sqrt((word_curve[i]-base[i%24])**2))
    x = [x for x in range(720)]
    fig, ax = plt.subplots()
    ax.plot(x,errors,linestyle="-")
    ax.axhline(y=0, color="black", linestyle="--")
    plt.show()

def compare2():
    word_curve = td.get_hourly_frequency("stadio")
    base = td.get_base()
    word_curve = scale_curve(word_curve,base)
    errors = []
    for i in range(len(word_curve)):
        errors.append(word_curve[i]-base[i%24])
    x = [x for x in range(720)]
    fig, ax = plt.subplots()
    ax.plot(x,errors,linestyle="-")
    #ax.plot(x,word_curve,color="green",linestyle="-")
    ax.plot(x,base*30,color="red",linestyle="-")
    ax.axhline(y=0, color="black", linestyle="--")
    plt.show()

def three_hour():
    word_curve = td.get_hourly_frequency("milano")
    base = td.get_base()
    word_curve = scale_curve(word_curve,base)
    errors = []
    for i in range(2,len(word_curve)):
        errors.append((abs((word_curve[i-2]-base[(i-2)%24]))+abs((word_curve[i-1]-base[(i-1)%24]))+abs((word_curve[i]-base[i%24])))/3)
    x = [x for x in range(len(errors))]
    fig, ax = plt.subplots()
    ax.plot(x,errors,linestyle="-")
    ax.axhline(y=0, color="black", linestyle="--")
    plt.show()

if __name__ == "__main__":
    compare2()
    
