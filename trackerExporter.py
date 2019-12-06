import os
import json
from collections import defaultdict

import pfpy


def getKeypoints():
    data = defaultdict(list)
    clip = pfpy.getClipRef(0)
    start = clip.getInPoint()
    end = clip.getOutPoint()
    num = pfpy.getNumTrackers()
    cnt = 0
    for i in range(0, num):
        tracker = pfpy.getTrackerRef(i)
        for frame in range(start, end+1):
            x, y = tracker.getTrackPosition(frame)
            data['Tracker'+str(cnt+1)].append([x, y])
        cnt += 1

    return data


data = getKeypoints()
pipe_dir = 'C:/Users/Ambrose/Documents/pipe/ml_projects/'
proj_dir = 'faceAutoTrackers/data/'
name = 'data.json'


def keyToDirectory(path, name):
    if os.path.exists:
        with open(os.path.join(path, name), 'w') as f:
            json.dump(data, f)
        print("file exported")
    else:
        print("Path does not exist")


keyToDirectory(pipe_dir+proj_dir, name)
