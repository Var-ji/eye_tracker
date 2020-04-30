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

def eye_iris_size(frame):
    """Returns the percentage of space that the iris takes up on
    the surface of the eye.

    Argument:
        frame (numpy.ndarray): Binarized iris frame
    """
    frame = frame[20:-20, 20:-20]
    height, width = frame.shape[:2]
    nb_pixels = height * width
    nb_blacks = nb_pixels - cv2.countNonZero(frame)
    return nb_blacks / nb_pixels

def image_processing(eye_frame, threshold):
    """Performs operations on the eye frame to isolate the iris

    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        threshold (int): Threshold value used to binarize the eye frame

    Returns:
        A frame with a single element representing the iris
    """
    kernel = np.ones((3, 3), np.uint8)
    if len(eye_frame.shape) > 2:
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

    new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
    new_frame = cv2.erode(new_frame, kernel, iterations=3)
    _, new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)

    return new_frame

def find_best_threshold(eye_frame):
    average_iris_size = 0.48 # Originally 0.48
    trials = {}

    for thr in range(5, 100, 5):
        iris_frame = image_processing(eye_frame, thr)
        trials[thr] = eye_iris_size(iris_frame)

    best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
    return best_threshold

def detect_iris(eye_frame, threshold=80):
    """Detects the iris and estimates the position of the iris by
    calculating the centroid.

    Arguments:
        eye_frame (numpy.ndarray): Frame containing an eye and nothing else
    """
    x = None
    y = None
    success = False
    iris_frame = image_processing(eye_frame, threshold)

    iris_frame = iris_frame[20:-20, 20:-20]

    contours, _ = cv2.findContours(iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    contours = sorted(contours, key=cv2.contourArea)

    try:
        moments = cv2.moments(contours[-2])
        x = int(moments['m10'] / moments['m00']) + 20
        y = int(moments['m01'] / moments['m00']) + 20
        success = True
    except (IndexError, ZeroDivisionError):
        x = None
        y = None
        success = False

    return success, x, y

def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear


def track_eye_centers(webcam=0):
#-----------------------------------------<CONSTANTS AND SHAPE PREDICTORS>---------------------------------------#

    CALIBRATED = False
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3

    COUNTER = 0
    TOTAL = 0

    shape_predictor = FRM.pose_predictor_model_location()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    vs = VideoStream(src=webcam).start()
    thresholds = [80, 80]

#----------------------------------------</CONSTANTS AND SHAPE PREDICTORS>---------------------------------------#

    while True:

        try:
            frame = vs.read()
        except:
            print("There was an error")
            vs.stop()
            return
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

            eye_1 = frame1[max(leftEye[1][1], leftEye[2][1])-5:min(leftEye[4][1], leftEye[5][1])+5, min(leftEye[0][0], leftEye[3][0])-5:max(leftEye[0][0], leftEye[3][0])+5]
            eye_2 = frame1[max(rightEye[1][1], rightEye[2][1])-5:min(rightEye[4][1], rightEye[5][1])+5, min(rightEye[0][0], rightEye[3][0])-5:max(rightEye[0][0], rightEye[3][0])+5]
            eye_1 = cv2.resize(eye_1, (187, 100))
            eye_2 = cv2.resize(eye_2, (187, 100)) # Preserve aspect ratio of eye (1.87 : 1)
            pair = [eye_1, eye_2]

            if not CALIBRATED:
                if True:
                    thresholds[0] = find_best_threshold(eye_1)
                    thresholds[1] = find_best_threshold(eye_2)
                    CALIBRATED = True
                    print(thresholds)
                else:
                    thresholds = [80, 80]
                    CALIBRATED = False

            if True:
                for i in range(len(pair)):
                    #print("Processing eye %d" % (i))
                    success, x, y = detect_iris(pair[i], thresholds[i])
                    if not success:
                        break
                    else:
                        cv2.circle(pair[i], (x, y), 3, (0, 255, 0), 2)
            #except:
            #    pass

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            cv2.imshow("Left eye", pair[0])
            cv2.imshow("Right eye", pair[1])

            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    track_eye_centers(0)