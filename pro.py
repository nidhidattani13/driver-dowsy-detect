from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
from twilio.rest import Client
import argparse
import imutils
import time
import dlib
import cv2
import os
import win32com.client as wincom

# Twilio credentials
account_sid = 'ACd04cf4beacade61b12e4b48dfaff5aab'
auth_token = 'fe06381284e233101e4a8a223b77a769'
twilio_phone_number = '+12762862519'  # Replace with your Twilio phone number
driver_phone_number = '+917863015585'  # Replace with the driver's phone number

client = Client(account_sid, auth_token)

def make_twilio_call(message):
    call = client.calls.create(
        twiml=f'<Response><Say>{message}</Say></Response>',
        to=driver_phone_number,  # Driver's phone number
        from_=twilio_phone_number  # Your Twilio phone number
    )
    print(f'Twilio Call SID: {call.sid}')

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying
    speak = wincom.Dispatch("SAPI.SpVoice")

    while True:
        if alarm_status:
            print('Calling driver to wake up')
            make_twilio_call("Please wake up!")

        if alarm_status2:
            print('Calling driver to take some fresh air')
            make_twilio_call("Please take some fresh air, sir")

        time.sleep(60)  # Adjust this delay as needed

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

import getpass
import os
USER_NAME = getpass.getuser()

def add_to_startup(file_path=""):
    if file_path == "":
        file_path = os.path.dirname(os.path.realpath(__file__))
    bat_path = r'C:\Users\%s\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup' % USER_NAME
    with open(bat_path + '\\' + "open.bat", "w+") as bat_file:
        bat_file.write(r'start "" "%s"' % file_path)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor("C:\\Users\\datta\\OneDrive\\Documents\\python vs\\shape_predictor_68_face_landmarks.dat")

print("-> Starting Video Stream")
vs = VideoStream(0).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (12, 255, 12), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (12, 255, 12), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 12), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=('Wake up, sir!',))
                    print("Wake Up")
                    t.daemon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('Take some fresh air, sir!',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
