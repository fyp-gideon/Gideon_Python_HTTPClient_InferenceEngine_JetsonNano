import tensorflow as tf
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.losses import binary_crossentropy
import numpy
import os
import numpy as np
import cv2 as cv
import argparse
import time
import threading 
import datetime
import requests

# How to invoke: 
# python3 gideon_cloud.py --input=/home/jupyter/.../input_video.mp4 --output=/home/jupyter/.../output_060520_63m_70_v1.mp4

############################### ARGUMENT PARSER ###############################
parser = argparse.ArgumentParser(description='Use this script to run action recognition using 3D ResNet34',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', '-i', help='Path to input video file. Skip this argument to capture frames from a camera.')
parser.add_argument("-o", "--output", required=True, help="path to our output video")
#parser.add_argument('--model', required=True, help='Path to model.')
#parser.add_argument('--classes', default=findFile('action_recongnition_kinetics.txt'), help='Path to classes list.')


############################### Loading Keras MODEL from the Disk ###############################
# Loading JSON Model from the disk:
json_file = open('/home/hassan/usb_drive/gideon/Activity_Accidents_16m_04032020_arch_ucf_accidents_16mp.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print ('[INFO] >> Model Loaded JSON File from the Disk')

# Populate Weights into the JSON Model:
model.load_weights('/home/hassan/usb_drive/gideon/Activity_Accidents_16m_04032020_weights_ucf_accidents_16mp.h5')
print ('[INFO] >> Model Loaded Successfully from the Disk')

# Model Summary:
model.summary()

# Model Compilation:
le = 0.001
opt = SGD(lr=le, momentum=0.009, decay=le)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print ('[INFO] >> Model Compiled Successfully !\n\n')

#print('********************** Model Weights (Format 1)**********************')
#model_weights = model.get_weights()
#print(model_weights)

print('********************** Model Configuration **********************')
total_layers = 0
for layer in model.layers:
    layer_weights = layer.get_weights()
    total_layers += 1
    print('Layer Name:',layer.name)
    print('Layer Configuration:',layer.get_config())
    print('Layer Input Shape:', layer.input_shape)
    print('Layer Output Shape:', layer.output_shape)
    #print('Layer Weights',layer_weights)
print('Total Layers in Network: ', total_layers)

############################### INFERENCE USING KERAS MODEL ###############################
#def get_class_names(path):
def get_class_names():
    class_names = ['Abnormal Activity', 'Normal Activity']
    #with open(path) as f:
    #    for row in f:
    #        class_names.append(row[:-1])
    #return class_names

#def classify_video(video_path, net_path):
def classify_video(video_path, output_path):
    SAMPLE_DURATION = 16 #Frames Batch Size for input to network for inference
    SAMPLE_SIZE = 112
    #mean = (114.7748, 107.7354, 99.4750)
    #class_names = get_class_names(args.classes)
    class_names = get_class_names()

    # Image/Input Characteristics:
    img_height = 100
    img_width = 100
    channels = 3
    batch_size = 16
    # ----------------------------------------- I M P O R T A N T -----------------------------------------
    #winName = 'Deep learning image classification in OpenCV'
    #cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
        
    # initialize the image mean for mean subtraction along with the
    # predictions queue
    cap = cv.VideoCapture(video_path)
    writer = None
    (W, H) = (None, None)
    # loop over frames from the video file stream
    frames_set = 0
    start_time = time.time()
    event_serial = 0
    count = 0
    while cv.waitKey(1) < 0:
    # constructing video frames batch for Conv3D input:
        frames = []            
        frames_copy = []
        frames_count = 0
            
        for _ in range(SAMPLE_DURATION):
            hasFrame, frame = cap.read()
            if not hasFrame:
                exit(0)
            # clone the output frame, then convert it from BGR to RGB
            output = frame.copy()
            frames_copy.append(output)
                
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                
            frame = cv.resize(frame, (100, 100))
            frame = frame /255
            frames.append(frame)
            frames_count += 1
        frames_set += 1
        
        print("16-Frames Set# ", frames_set)
        print('Frames Count in processed set: ', frames_count)
            #inputs = cv.dnn.blobFromImages(frames, 1, (SAMPLE_SIZE, SAMPLE_SIZE), mean, True, crop=True)
            #inputs = np.transpose(inputs, (1, 0, 2, 3))
            #inputs = np.expand_dims(inputs, axis=0)
            #net.setInput(inputs)
            #outputs = net.forward()
            #class_pred = np.argmax(outputs)
            #label = class_names[class_pred]
            
        # Performing Prediction on batch of 16 frames:
        frames = np.array(frames).reshape(-1, 16, 100, 100, 3)
        #_ = input('[INFO] >>> Press enter to predict !!!')
        pred_npy = model.predict(frames, batch_size=batch_size, verbose=0)
        #print('Shape of Prediction List: ', pred_npy.shape)
        pred = pred_npy.argmax(axis=1)
        #label = class_names[pred]

        if frames_set == 1:
            start_time = time.time()

        threshold = 0.70
        if pred_npy[0][0] >= threshold:
            event = 'Abnromality Observed !'
            event_occured = True
            count += 1
        else:
            event = 'Normal'
            event_occured = False
            count = 0

        print('Probability (Accident):', pred_npy[0][0])
        print('Probability (Normal):', pred_npy[0][1])
        print('Prediction: >> ', event)
            
            # clone the output frame, then convert it from BGR to RGB
            # ordering, resize the frame to a fixed 100x100, and then:
        for frame in frames_copy:
                # draw the activity on the output frame
            text = "Event:  {}".format(event)
            cv.putText(frame, text, (20, 20), cv.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 0), 2)
            text = "Accident Probability: {0:0.2f}".format(pred_npy[0][0])
            cv.putText(frame, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
            text = "Normal Probability: {0:0.2f}".format(pred_npy[0][1])
            cv.putText(frame, text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

                # check if the video writer is None
            if writer is None:
                    # initialize our video writer
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(output_path, fourcc, 30, (W, H), True)

                # write the output frame to disk
            writer.write(frame)
        
        # POSTing event_image to NodeJS server for Database insertion:
        if (event_occured == True and frames_set % 16 == 0):
            # Updating Event Serial:
            event_serial = event_serial + 1
            # Saving image to disk:
            event_image = frames_copy[8]
            image_name = '/home/hassan/Downloads/gideon_node_api/public/image_' + str(event_serial) + '.jpg'
            cv.imwrite(image_name, event_image)
            # Creating event_image location string:
            image_url = "http://192.168.10.11:3000/image_" + str(event_serial) + '.jpg'
            x = datetime.datetime.now()
            print(x.strftime("%A") + " " + str(x.date()) + " @ " + str(x.time()))
            url = 'http://192.168.10.5:3900/api/events/'
            
            event = {
                    "event_label": "Dummy Event",
                    "event_serial": event_serial,
                    "date_time": x.strftime("%A") + " " + str(x.date()) + " @ " + str(x.time()),
                    "normal_probability": 0.35,
                    "abnormal_probability": 0.65,
                    "image_url": image_url,
                    "clip_url": "http://localhost:3900/Brendon.mp4"
                }

            x = requests.post(url, json= event)
            print(x.text) 

        # Calculating Inference Metrics:
        end_time = time.time()
        print ('[INFO] >>> Total time for Inference: ', end_time - start_time)
        print ('[INFO] >>> Total Frames processed: ', frames_set * 16 - 16)
        print ('[INFO] >>> Inference FPS: ', ((frames_set*16)-16)/(end_time - start_time))
        print ('\n\n')
    print("[INFO] cleaning up system resources...")
    writer.release()
    cap.release()
            

if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    classify_video(args.input if args.input else 0, args.output)