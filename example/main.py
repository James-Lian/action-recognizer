# code to gather computer vision + pose data to train machine learning model
import traceback
import time
import math

import multiprocessing as multip
from multiprocessing import Manager
from collections import deque

from Vector3 import Vector3
from lstm_model import *
model_lstm = ActionRecognitionModel()
model_lstm.load_model("ArmsRecog_v2_best.keras", label_filepath="ArmsRecog_v2.pkl")
mode = "arms"
class_names = model_lstm.label_encoder.classes_
frame_no = 0
frame_folder = []
just_joints_and_time = []
action_time = 0

### Extracting joint positions ###
import numpy as np
import cv2
FONT_BOLD = 2
FONT_COLOUR = (0, 0, 0)
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.pose
model = mp_holistic.Pose(model_complexity=0)

data = [] # array of dictionaries
curr_recording_data = [] # timestamps and joint positions for current action being recorded

# all joints
all_joints = {}

def return_landmark(landmark):
    return [landmark.x, landmark.y, landmark.z]

def process_landmarks(no, action, timestamp):
    snapshot = {
        "action_no": no,
        "action": action,
        "timestamp": timestamp,
    }
    for joint in all_joints:
        snapshot[joint + "_x"] = all_joints[joint][0]
        snapshot[joint + "_y"] = all_joints[joint][1]
        snapshot[joint + "_z"] = all_joints[joint][2]
    
    return snapshot

cap = cv2.VideoCapture(0)
ptime = 0
action_class = ""
probabilities = ""
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = time.time()

    h, w, _channels = frame.shape

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    model_results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = True

    landmarks = model_results.pose_world_landmarks.landmark

    # face
    all_joints["nose"] = return_landmark(landmarks[mp_holistic.PoseLandmark.NOSE.value])
    all_joints["l_ear"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value])
    all_joints["r_ear"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value])
    all_joints["l_eye_inner"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EYE_INNER.value])
    all_joints["r_eye_inner"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value])
    all_joints["l_eye"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EYE.value])
    all_joints["r_eye"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_EYE.value])
    all_joints["l_eye_outer"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value])
    all_joints["r_eye_outer"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value])
    all_joints["l_mouth"] = return_landmark(landmarks[mp_holistic.PoseLandmark.MOUTH_LEFT.value])
    all_joints["r_mouth"] = return_landmark(landmarks[mp_holistic.PoseLandmark.MOUTH_RIGHT.value])

    # legs
    all_joints["l_hip"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value])
    all_joints["r_hip"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value])
    all_joints["l_knee"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value])
    all_joints["r_knee"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value])
    all_joints["l_ankle"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value])
    all_joints["r_ankle"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value])
    all_joints["l_heel"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_HEEL.value])
    all_joints["r_heel"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_HEEL.value])
    all_joints["l_toe"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value])
    all_joints["r_toe"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value])

    # arms
    all_joints["l_shoulder"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value])
    all_joints["r_shoulder"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value])
    all_joints["l_elbow"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value])
    all_joints["r_elbow"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value])
    all_joints["l_wrist"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value])
    all_joints["r_wrist"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value])
    all_joints["l_thumb"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_THUMB.value])
    all_joints["r_thumb"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_THUMB.value])
    all_joints["l_index"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value])
    all_joints["r_index"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value])
    all_joints["l_pinky"] = return_landmark(landmarks[mp_holistic.PoseLandmark.LEFT_PINKY.value])
    all_joints["r_pinky"] = return_landmark(landmarks[mp_holistic.PoseLandmark.RIGHT_PINKY.value])

    if model_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            model_results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    image = cv2.flip(image, 1)

    if frame_no == 20:
        frame_folder = np.array(frame_folder)
        frame_folder = np.expand_dims(frame_folder, axis=0)
        action_class, probabilities = model_lstm.predict(frame_folder)
        action_class = str(action_class[0])
        probabilities = probabilities[0].tolist()
        print(action_class, probabilities)

        frame_no = 0
        frame_folder = []
        action_time = time.time()
        
    else:
        flist = [ctime-action_time]
        just_legs = all_joints.copy()
        joints_to_remove = []
        for joint in just_legs:
            if mode == "legs" and joint not in ["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel", "l_toe", "r_toe"]:
                joints_to_remove.append(joint)
            elif mode == "arms" and joint not in ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_thumb", "r_thumb", "l_index", "r_index", "l_pinky", "r_pinky"]:
                joints_to_remove.append(joint)
        for joint in joints_to_remove:
            del just_legs[joint]

        glist = [value for sublist in just_legs.values() for value in sublist]
        print(len(glist))
        flist.extend(glist)
        flist = np.array(flist)
        frame_folder.append(np.array(flist))
    

    # display FPS
    cv2.putText(image, "FPS: " + str(int(fps)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)

    probability = ""
    if action_class in list(class_names):
        probability = probabilities[list(class_names).index(action_class)]
    cv2.putText(image, action_class + " " + str(probability), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, FONT_COLOUR, FONT_BOLD)
    cv2.putText(image, str(probabilities), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, FONT_COLOUR, FONT_BOLD)


    cv2.imshow("feed", image)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
    
    frame_no += 1

cv2.destroyAllWindows()
