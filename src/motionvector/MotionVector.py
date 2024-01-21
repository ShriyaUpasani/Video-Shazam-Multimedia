

import cv2
import numpy as np
from multiprocessing import Pool

# Function to calculate the difference between two frames
def calculate_frame_diff(args):
    frame1, frame2 = args
    diff = cv2.absdiff(frame1, frame2)
    score = np.sum(diff)
    return score

def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return frames

def calculate_motion(video_path):
    frames = read_frames(video_path)

    # Pair frames for processing
    frame_pairs = [(frames[i], frames[i+1]) for i in range(len(frames)-1)]

    # Use multiprocessing to calculate frame differences
    with Pool() as pool:
        motion_scores = pool.map(calculate_frame_diff, frame_pairs)

    return motion_scores

def main():
    video_path = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Queries/video1_1.mp4'
    motion_scores = calculate_motion(video_path)
    print(motion_scores)

if __name__=='__main__':
    main()