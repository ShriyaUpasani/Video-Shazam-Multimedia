import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import os


# Preprocessing the Data
def preprocess_data(train_dir, test_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(100, 100),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(100, 100),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')

    return train_generator, test_generator


# Building the Model
def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Main function
def main(train_dir, test_dir):
    train_generator, test_generator = preprocess_data(train_dir, test_dir)

    model = build_model(num_classes=train_generator.num_classes)
    model.fit(train_generator, epochs=50, validation_data=test_generator)
    model.save('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/SavedModel/signatureDetect.h5')


def load_and_predict(image_path):
    # Load the saved model
    model = load_model('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/SavedModel/signatureDetect.h5')

    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    img = np.reshape(img, (1, 100, 100, 1))
    img = img / 255.0
    # Make a prediction
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class


if __name__ == "__main__":
    # train
    # train_dir = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train'
    # test_dir = '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Test'
    # main(train_dir, test_dir)

    train_generator, test_generator = preprocess_data(
        '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train',
        '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Test')

    class_indices = train_generator.class_indices
    valuesToLabels = {v: k for k, v in class_indices.items()}
    print(class_indices)
    # test
    key=load_and_predict(
        "/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train/video10/frame_10_video10.jpg")
    print(valuesToLabels[key[0]])
    key=load_and_predict(
        "/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train/video10/frame_11_video10.jpg")
    print(valuesToLabels[key[0]])

    key = load_and_predict(
        "/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train/video19/frame_0_video19.jpg")
    print(valuesToLabels[key[0]])



