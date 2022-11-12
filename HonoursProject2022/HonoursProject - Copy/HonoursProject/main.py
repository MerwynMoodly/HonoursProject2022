# COMP 700 Honours Project
# Author: Merwyn Moodly 219095202

import numpy as np
import cv2
import os
import imutils
import playsound
#from ffpyplayer.player import MediaPlayer
import multiprocessing as mp

from pygame import mixer

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

# Starting the mixer
mixer.init()

# Loading the song
mixer.music.load("beeping Yellow.mp3")

# Setting the volume
mixer.music.set_volume(0.7)


def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    # ensure at least one detection exists
    if len(idzs) > 0:
        # loop over the indexes we are keeping
        for i in idzs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    # return the list of results
    return results




def Scan(running, beeping, enterRed, enterYellow, enterGreen, red , yellow, green):

    running.value=1
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    weights_path = "yolov4-tiny.weights"
    config_path = "yolov4-tiny.cfg"

    model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    layer_name = model.getLayerNames()
    layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]
    cap = cv2.VideoCapture(0)
    # player = MediaPlayer("beeping Yellow.mp3")
    writer = None

    alpha = 0.3  # more transparent the closer to 0 it is

    while True:
        (grabbed, image) = cap.read()

        # (audio_frame, val) = player.get_frame()

        if not grabbed:
            break
        image = imutils.resize(image, width=1000)
        results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
        imgCopy = image.copy()

        #Adding to this increases the size of everything
        sizeFactor = 50+75+40 +50

        redZoneEnd = (image.shape[1], image.shape[0])
        redZoneStart = (0, redZoneEnd[1] - (sizeFactor))

        amberZoneStart = (redZoneStart[0], (redZoneStart[1] - (sizeFactor) +150))
        amberZoneEnd = (redZoneEnd[0], (redZoneEnd[1] - sizeFactor))

        greenZoneStart = (amberZoneStart[0], (amberZoneStart[1] - sizeFactor) + 200)
        greenZoneEnd = (amberZoneEnd[0], (amberZoneEnd[1] - (sizeFactor) +175))

        for res in results:
            cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

            cv2.rectangle(imgCopy, redZoneStart, redZoneEnd, (0, 0, 255), -1)  # green warning

            cv2.rectangle(imgCopy, amberZoneStart, amberZoneEnd, (0, 191, 255), -1)  # amber warning BGR

            cv2.rectangle(imgCopy, greenZoneStart, greenZoneEnd, (0, 255, 128), -1)  # amber warning BGR

            if (res[1][3] > redZoneStart[1]):
                #print("RED ZONE!")
                enterYellow.value = 0
                enterGreen.value = 0

                red.value = 1
                yellow.value = 0
                green.value = 0


                beeping.value = 1
                enterRed.value += 1

                if (enterRed.value == 1000):
                    enterRed.value = 5

            elif (res[1][3] > amberZoneStart[1]):
                #print("AMBER ZONE!")
                enterGreen.value = 0
                enterRed.value = 0

                red.value = 0
                yellow.value = 1
                green.value=0


                beeping.value = 1
                enterYellow.value += 1

                if (enterYellow.value == 1000):
                    enterYellow.value = 5
                #print(beeping)
            elif (res[1][3] > greenZoneStart[1]):
                #print("GREEN ZONE!")
                enterYellow.value = 0
                enterRed.value = 0

                red.value = 0
                yellow.value = 0
                green.value = 1


                beeping.value=1
                enterGreen.value +=1


                if(enterGreen.value==1000):
                    enterGreen.value=5

            else:
                beeping.value=0
                enterGreen.value=0
                #print(beeping)

            cv2.addWeighted(imgCopy, alpha, image, 1 - alpha, 0, image)

        cv2.imshow("Detection", image)

        #    if val != 'eof' and audio_frame is not None:
        #        # audio
        #        img, t = audio_frame

        key = cv2.waitKey(1)
        if key == 27:
            print("Width: ", (image.shape[1]))
            print("Height: ", (image.shape[0]))
            break

    cap.release()
    cv2.destroyAllWindows()
    running.value=0


def Beep(running,beeping,enterRed, enterYellow, enterGreen, red , yellow, green):

    while running.value==1:

        #print(enter.value)

        #this part causes error


        if(beeping.value==1):

            if (red.value == 1 and enterRed.value==1):

                if(mixer.Channel(1).get_busy()==True):
                    mixer.Channel(1).stop()

                mixer.Channel(0).play(mixer.Sound('beeping Red.mp3'), -1)

            elif (yellow.value == 1 and enterYellow.value==1):

                if (mixer.Channel(2).get_busy() == True):
                    mixer.Channel(2).stop()
                elif(mixer.Channel(0).get_busy() == True):
                    mixer.Channel(0).stop()

                mixer.Channel(1).play(mixer.Sound('beeping Yellow.mp3'), -1)

            elif (green.value == 1 and enterGreen.value==1):

                if (mixer.Channel(1).get_busy() == True):
                    mixer.Channel(1).stop()

                mixer.Channel(2).play(mixer.Sound('beeping Green.mp3'), -1)

        else:
            if(mixer.Channel(2).get_busy()==True and enterGreen.value==0):
                mixer.Channel(2).stop()


    print("Beep Ended")




if __name__ == '__main__':
    running = mp.Value('i', 1)
    beeping = mp.Value('i', 0)
    enterRed = mp.Value('i', 0)
    enterGreen = mp.Value('i', 0)
    enterYellow = mp.Value('i', 0)

    red = mp.Value('i', 0)
    yellow = mp.Value('i', 0)
    green = mp.Value('i', 0)


    process1 = mp.Process(target=Scan, args=(running, beeping, enterRed, enterYellow, enterGreen, red, yellow, green))
    process2 = mp.Process(target=Beep, args=(running, beeping, enterRed, enterYellow, enterGreen, red, yellow, green))

    process1.start()
    process2.start()
    process1.join()
    process2.join()




