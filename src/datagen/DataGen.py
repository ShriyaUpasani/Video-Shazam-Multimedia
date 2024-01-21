import cv2
import os
from PIL import Image
import logging

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def downscale_image(original_frame, new_width, new_height):
    img_ycrcb = cv2.cvtColor(original_frame, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)
    downscaled_image = cv2.resize(Y, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return downscaled_image

def extract_frames(video_path, output_folder,new_width,new_height,label):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Frames per second
    success, image = vidcap.read()
    count = 0

    while success:
        # Save frame every second
        if count % int(fps) == 0:
            frame_file = os.path.join(output_folder, f"frame_{count // int(fps)}_{label}.jpg")
            image=downscale_image(image,new_width,new_height)
            cv2.imwrite(frame_file, image)     # save frame as JPEG file
            print(f"Saved {frame_file}")

        success, image = vidcap.read()

        count += 1

    vidcap.release()
    print("Done extracting Frames.")


def extractLabel(filePath):
    return filePath.split("/")[-1].split(".")[0]
def main():
    rootPath="/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Videos"
    videFileName=os.listdir(rootPath)
    for file in videFileName:
        filePath=rootPath+"/"+file
        logger.info("Extracting frames from video")
        label=extractLabel(filePath)
        os.makedirs('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train/'+label, exist_ok=True)
        output_folder = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train/'+label
        extract_frames(filePath, output_folder,100,100,label)
        logger.info("Done extracting frames from video")

def extractQueriesLabel(filePath):
    return filePath.split("/")[-1].split(".")[0].split("_")[0]
def mainForQueries():
    rootPath="/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Queries"
    videFileName=os.listdir(rootPath)
    for file in videFileName:
        if(file.endswith(".mp4")):

            filePath=rootPath+"/"+file
            logger.info("Extracting frames from video")
            label=extractQueriesLabel(filePath)
            os.makedirs('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Valid/'+label, exist_ok=True)
            output_folder = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Valid/'+label
            extract_frames(filePath, output_folder,100,100,label)
            logger.info("Done extracting frames from video")

if __name__ == "__main__":
    main()
    # mainForQueries()