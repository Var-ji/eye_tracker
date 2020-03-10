from scipy.spatial import distance as dist
import face_recognition_models as FRM
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import math
from keras.models import model_from_json


def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 4

COUNTER = 0
TOTAL = 0

directions = ['down', 'left', 'center', 'right', 'up']

shape_predictor = FRM.pose_predictor_model_location()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

with open('./eye_tracker_arch.json') as file:
    json_string = file.read()

eye_tracker = model_from_json(json_string)
eye_tracker.load_weights('eye_tracker.h5')
eye_tracker.summary()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
fileStream = False
inference_flag = False

while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame1 = frame.copy()
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = EAR(leftEye)
        rightEAR = EAR(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        roi_1 = np.array([leftEye], dtype=np.int32)
        roi_2 = np.array([rightEye], dtype=np.int32)
        eye_1 = frame1[max(leftEye[1][1], leftEye[2][1])-5:min(leftEye[4][1], leftEye[5][1])+5, min(leftEye[0][0], leftEye[3][0])-5:max(leftEye[0][0], leftEye[3][0])+5]
        eye_2 = frame1[max(rightEye[1][1], rightEye[2][1])-5:min(rightEye[4][1], rightEye[5][1])+5, min(rightEye[0][0], rightEye[3][0])-5:max(rightEye[0][0], rightEye[3][0])+5]
        pair = [eye_1, eye_2]
        eye_left = cv2.resize(eye_1, (224, 224))
        eye_right = cv2.resize(eye_2, (224, 224))
        cv2.imshow("Left Eye", eye_left)
        cv2.imshow("Right Eye", eye_right)

        if inference_flag is True:
            y1 = eye_tracker.predict(np.asarray([eye_left]))[0]
            y2 = eye_tracker.predict(np.asarray([eye_right]))[0]
            net = np.divide(np.add(y1, y2), 2.0)
            direc = np.argmax(net)
            if direc == 0:
                direc = np.argmax(net[1:])
            else:
                print("Looking %s" % (directions[direc]))

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('i'):
        inference_flag = not inference_flag

cv2.destroyAllWindows()
vs.stop()
