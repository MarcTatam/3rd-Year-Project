from moviepy.editor import *
from datetime import datetime, timedelta
import os

def first_render():
    """Renders the first video"""
    image1 = ImageClip("./images/sms2013-11-01-00-00.png")
    image2 = ImageClip("./images/sms2013-11-01-00-10.png")
    image3 = ImageClip("./images/sms2013-11-01-00-20.png")
    image1 = image1.set_duration(0.1)
    image2 = image2.set_duration(0.1)
    image3 = image3.set_duration(0.1)
    image1 = image1.set_start(0)
    image2 = image2.set_start(0.1)
    image3 = image3.set_start(0.2)
    final = CompositeVideoClip([image1,image2,image3])
    final = final.set_duration(0.3)
    final.write_videofile("heatmap.mp4",fps=10)

def render(start:int, stop:int):
    """Renders a video in a given time range
    Deprecated due to it creating poor video quality
    
    Args
    start - unix timestamp for a given start time
    stop - unix timestamp for a given stop time"""
    this = start
    while this <= stop:
        print(this)
        dt = datetime.fromtimestamp(int(this/1000))
        clip = VideoFileClip("heatmap.mp4")
        dur = clip.duration
        imageclip = ImageClip("./images/sms"+dt.strftime('%Y-%m-%d-%H-%M')+".png")
        imageclip = imageclip.set_duration(0.1)
        imageclip = imageclip.set_start(dur)
        print(imageclip.start)
        final = CompositeVideoClip([clip,imageclip])
        final = final.set_duration(dur+0.1)
        final.write_videofile("temp.mp4",fps=10)
        os.remove("heatmap.mp4")
        os.rename("temp.mp4", "heatmap.mp4")
        this += 600000

def render_day(date:str):
    """Renders a video for a given date
    
    Args
    date - date to render in string form"""
    start = 0
    images = []
    for hour in range(24):
        for minute in range(6):
            imageclip = ImageClip("./images/sms"+date+"-"+str(hour).zfill(2)+"-"+str(minute*10).zfill(2)+".png")
            imageclip = imageclip.set_duration(0.1)
            imageclip = imageclip.set_start(start)
            start += 0.1
            print(start)
            images.append(imageclip)
    final = CompositeVideoClip(images)
    final.set_duration(14.4)
    final.write_videofile("./videos/"+date+".mp4",fps=10)

def render_all():
    """Renders all videos"""
    for day in range(1,31):
        print(day)
        render_day("2013-11-"+str(day).zfill(2))
    for day in range(1,32):
        print(day)
        render_day("2013-12-"+str(day).zfill(2))


def join_all():
    """Joins all videos into a single video"""
    clips = []
    start = 0
    for day in range(1,31):
        print(day)
        clip = VideoFileClip("./videos/2013-11-"+str(day).zfill(2)+".mp4")
        clip = clip.set_start(start)
        start+=14.4
        clips.append(clip)
    for day in range(1,32):
        print(day)
        clip = VideoFileClip("./videos/2013-12-"+str(day).zfill(2)+".mp4")
        clip = clip.set_start(start)
        start+=14.4
        clips.append(clip)
    clip = VideoFileClip("./videos/2014-01-01.mp4")
    clip = clip.set_start(start)
    clips.append(clip)
    final = CompositeVideoClip(clips)
    final.write_videofile("heatmap.mp4", fps = 10)


if __name__ == "__main__":
    join_all()