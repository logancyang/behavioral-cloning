import csv
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from preprocess import generate_new_image

DRIVING_LOG = './data/driving_log.csv'
IMG_PATH = './data/'
STEERING_COEFFICIENT = 0.23

lines = []
with open(DRIVING_LOG) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Remove header line
lines = lines[1:]
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


def get_images(batch_size=64):
    """
    Randomly pick left, center, right images with equal probability
    :param batch_size: Size of the image batch
    :return: A list of (image, steering_angle)
    """
    data = pd.read_csv(DRIVING_LOG)
    n_imgs = len(data)
    random_indices = np.random.randint(0, n_imgs, batch_size)
    imgs_and_angles = []
    for index in random_indices:
        random_type = np.random.randint(0, 3)
        if random_type == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_COEFFICIENT
            imgs_and_angles.append((img, angle))
        elif random_type == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            imgs_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_COEFFICIENT
            imgs_and_angles.append((img, angle))
    return imgs_and_angles


def generate_next_batch(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = get_images(batch_size)
        for image, angle in images:
            raw_image = plt.imread(IMG_PATH + image)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size must be True.'
        yield np.array(X_batch), np.array(y_batch)


def save_model(model, json_name='model.json', model_name='model.h5'):
    with open(json_name, 'w') as outfile:
        json.dump(model.to_json(), outfile)
    model.save(model_name)



