#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json


def draw_boxes_with_angle(image, boxes, labels, max_labels, angle_discretization):
    for box in boxes:
        cv2.rectangle(image, (box.xmin, box.ymin),
                      (box.xmax, box.ymax), (0, 255, 0), 1)

        exp_label = box.get_label()
        label = int(exp_label / (360. / angle_discretization))
        angle = (exp_label % (360. / angle_discretization)) * \
            angle_discretization

        cv2.putText(image, "{} -> {} =  {}".format(exp_label, label, int(angle)),
                    (box.xmin, box.ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image.shape[0],
                    (0, 255, 0), 1)
    return image


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '--obj_th',
    default=0.3,
    type=float)

argparser.add_argument(
    '--nms_th',
    default=0.3,
    type=float)


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       20.0,
                                       (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()
    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image, args.obj_th, args.nms_th)
        image = draw_boxes_with_angle(
            image, boxes, config['model']['labels'], 90, 20.0)

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
