import os

import cv2
import numpy as np
import logging


logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def display_frame(frame,diff):
    cv2.imshow('Frame', frame)
    cv2.imshow('Diff', diff)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

def calculate_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    print(fps)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_frame, gray)
        score = np.sum(diff)
        motion_scores.append(score)
        prev_frame = gray

        # display_frame(frame, diff)

    cap.release()
    cv2.destroyAllWindows()
    return motion_scores

def getVideoNames(path):
    return path.split("/")[-1].split(".")[0]



def diffCalForTrainData():
    video_paths = os.listdir('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos')
    output_file = open('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/PreProcessedData/motion_diff.csv','w')
    output_file.write('video_name,motion_diff\n')
    for video_path in video_paths:
        logger.info("Calculating motion for train video: "+video_path)
        motion_stats = calculate_motion('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos/'+video_path)
        video_name=getVideoNames(video_path)
        output_file.write(video_name+','+str(motion_stats)+'\n')
    logger.info("Done calculating motion for train videos")


def diffCalForTestData(videoPath):
    logger.info("Calculating motion for test video: "+videoPath)
    motion_stats = calculate_motion(videoPath)
    logger.info("Done calculating motion for test video")
    return motion_stats


# Usage
# video_path = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Queries/video1_1.mp4'  # Replace with your video path
# video_path='/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos/video1.mp4'
# motion_stats = calculate_motion(video_path)
# print(motion_stats)


# uncomment to process train data and
if __name__=='__main__':
    diffCalForTrainData()

    # diffCalForTestData('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Queries/video1_1.mp4')
    # print(diff)
