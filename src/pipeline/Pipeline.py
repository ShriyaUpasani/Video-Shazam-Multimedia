import time
from functools import wraps
import cv2
import numpy as np
import os
import sys

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

project_path = "/Users/sms/USC/MS-SEM2/multimedia/video-shazam"
src_path = os.path.join(project_path, "src")
sys.path.insert(0, src_path)


# Now you can import your modules
from motionvector.Motion import diffCalForTestData
from signature.Model import preprocess_data
from videoplayer.VideoPlayer import VideoPlayer
from constants import FilePaths as constants


total_time=0

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global total_time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Time taken by {func.__name__}: {elapsed_time} seconds")
        return result
    return wrapper
class Pipeline():
    def __init__(self, testFilePath):
        logger.info("Initializing Pipeline")
        self.testFilePath=testFilePath
        self.class_indices= None
        self.labels=None
        self.model=None
        self.testFrames =list()
        self.testMotionResidue=None
        self.trainMotionResidue=dict()
        self.newWidth=100
        self.newHeight=100
        self.videoPaths=dict()
        self.audioPaths=dict()

    @time_it
    def load_video_paths(self,directory):
        logger.info("Loading video paths")
        video_paths = {}
        for filename in os.listdir(directory):
            if filename.endswith(".mp4"):  # Ensuring only .mp4 files are considered
                # Extracting the video label (e.g., 'video1' from 'video1.mp4')
                label = os.path.splitext(filename)[0]
                # Constructing the full path
                full_path = os.path.join(directory, filename)
                # Adding to the dictionary
                video_paths[label] = full_path
        logger.info("Done loading video paths")
        self.videoPaths=video_paths
        return video_paths

    @time_it
    def load_audio_paths(self,directory):
        logger.info("Loading audio paths")
        audio_paths = {}
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):  # Ensuring only .wav files are considered
                # Extracting the audio label (e.g., 'video1' from 'video1.wav')
                label = os.path.splitext(filename)[0]
                # Constructing the full path
                full_path = os.path.join(directory, filename)
                # Adding to the dictionary
                audio_paths[label] = full_path
        logger.info("Done loading audio paths")
        self.audioPaths=audio_paths
        return audio_paths
    @time_it
    def loadLabels(self,trainDir,testDir):
        logger.info("Loading Labels")
        train_generator, test_generator = preprocess_data(
            trainDir,
            testDir)

        self.class_indices = train_generator.class_indices
        self.labels = {v: k for k, v in self.class_indices.items()}

    @time_it
    def loadPreTrainedModel(self,savedModelPath):
        logger.info("Loading Pretrained Model")
        self.model = load_model(savedModelPath)

    @time_it
    def predict(self,frame):
        logger.info("Predicting video class")
        img = cv2.resize(frame, (100, 100))
        img = np.reshape(img, (1, 100, 100, 1))
        img = img / 255.0
        # Make a prediction
        prediction = self.model.predict(img)

        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = self.labels[predicted_class[0]]
        logger.info("Predicted class: "+str(predicted_class[0]))
        return predicted_label
        

    def downscale_image(self,original_frame, new_width, new_height):
        img_ycrcb = cv2.cvtColor(original_frame, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycrcb)
        downscaled_image = cv2.resize(Y, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return downscaled_image

    @time_it
    def extract_frames(self):
        logger.info("Extracting frames from video")
        vidcap = cv2.VideoCapture(self.testFilePath)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        count = 0

        while success:
            # Save frame every second
            if count % int(fps) == 0:
                image = self.downscale_image(image, self.newWidth, self.newHeight)
                self.testFrames.append(image)
            success, image = vidcap.read()
            count += 1

        vidcap.release()
        logger.info("Done extracting Frames.")

    @time_it
    def loadTrainMotionResidue(self,path):
        logger.info("Loading Pretrained Motion Residue")
        self.trainMotionDiff = dict()
        with open(path, "r") as filestream:
            colnames=filestream.readline()
            logger.info("colnames: "+colnames)
            for line in filestream.readlines():
                label, residueX = line.split("[")
                label = label.strip(",")
                residue = eval("[" + residueX)
                self.trainMotionDiff[label] = residue
        logger.info("Done loading Pretrained Motion Residue")
    @time_it
    def extractMotionResidue(self):
        logger.info("Extracting Motion Residue")
        self.testMotionResidue=diffCalForTestData(self.testFilePath)
        logger.info("Done extracting Motion Residue")
        return self.testMotionResidue
    


    @time_it
    def find_matching_frames(self, query_array,predicted_label):
    # Step 1: Indexing the train array

        train_array = self.trainMotionDiff.get(predicted_label, [])
        train_index = {}
        for i, value in enumerate(train_array):
            if value not in train_index:
                train_index[value] = []
            train_index[value].append(i)

    # Step 2: Finding matching values
        first_query, last_query = query_array[0], query_array[-1]
        if first_query in train_index and last_query in train_index:
            # Step 3: Checking frame difference
            query_diff = len(query_array)-1
            for first_pos in train_index[first_query]:
                for last_pos in train_index[last_query]:
                    if last_pos - first_pos == query_diff:
                        return first_pos, last_pos
                    

    def playVideo(self,label, position):
        logger.info("Playing video from position: "+str(position)+" for label: "+label)
        vPath=self.videoPaths[label]
        aPath=self.audioPaths[label]
        VideoPlayer( "Video Player", vPath, aPath, position)


    def validateOffsetAndPlayVideo(self,positons,predicted_label):
        if positons:
            logger.info("Match found in "+predicted_label+" at positions: "+str(positons))
        else:
            logger.info("No match found in "+predicted_label)
            return
        startOffset,endOffset=positons
        self.playVideo(predicted_label, int(startOffset))




if __name__ == "__main__":

    pipeline=Pipeline(constants.QUERY_PATH)
    pipeline.load_audio_paths(constants.AUDIO_FILE_PATH)
    pipeline.load_video_paths(constants.VIDEO_FILE_PATH)
    pipeline.loadLabels(constants.TRAIN_DATA_PATH,constants.TEST_DATA_PATH)
    pipeline.loadPreTrainedModel(constants.PRE_TRAINED_MODEL_PATH)
    pipeline.loadTrainMotionResidue(constants.MOTION_RESIDUE_PATH)
    while(True):
        queryPath=input("Enter the path of the query file:\n")
        startTime = time.time()
        pipeline.testFilePath=queryPath
        pipeline.extract_frames()
        query_motion_residue = pipeline.extractMotionResidue()
        predicted_label = pipeline.predict(pipeline.testFrames[0])
        match_positions = pipeline.find_matching_frames(query_motion_residue,predicted_label)
        endTime=time.time()
        pipeline.validateOffsetAndPlayVideo(match_positions,predicted_label)
        print("Total time taken: " + str(endTime - startTime))
    # logger.info("Total time taken: "+str(total_time))








