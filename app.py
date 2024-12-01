import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import os
import datetime
import threading
import time
import requests 
import train as train
import winsound
from MAR import mouth_aspect_ratio  
from EAR import eye_aspect_ratio

def yawn(mouth):
    return (euclideanDist(mouth[2], mouth[10]) + euclideanDist(mouth[4], mouth[8])) / (
                2 * euclideanDist(mouth[0], mouth[6]))


def getFaceDirection(shape, size):
    image_points = np.array([
        shape[33],  
        shape[8],  
        shape[45],  
        shape[36],  
        shape[54], 
        shape[48]  
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  
        (0.0, -330.0, -65.0),  
        (-225.0, 170.0, -135.0),  
        (225.0, 170.0, -135.0),  
        (-150.0, -150.0, -125.0),  
        (150.0, -150.0, -125.0)  

    ])

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return translation_vector[1][0]


def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


def ear(eye):
    return (euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2 * euclideanDist(eye[0], eye[3]))


def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    cv2.imwrite('left-eye.jpg', img[y1:y2, x1:x2])
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    cv2.imwrite('right-eye.jpg', img[y1:y2, x1:x2])

alert_sound = 'alert-sound.wav'
take_break_sound = 'take_a_break.wav'
focus_sound = 'focus.wav'

frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5

close_thresh = 0.3  
flag = 0
yawn_countdown = 0
map_counter = 0
map_flag = 1

capture = cv2.VideoCapture(0)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

DRIVER_INFO = {
    'id': 'USER001',
    'name': 'John Doe',
}

CAPTURE_DIR = 'captured_drowsy'
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

class APIHandler:
    def __init__(self, login_url, credentials):
        self.login_url = login_url
        self.credentials = credentials
        self.session = requests.Session()
        self.user_id = None
        self.login()

    def login(self):
        response = self.session.post(self.login_url, json=self.credentials)
        if response.status_code == 200:
            print("Login successful.")
            self.user_id = response.json().get('id')  
        else:
            print(f"Login failed: {response.status_code}, {response.text}")

    def upload_image(self, image_path):
        if not self.user_id:
            print("User ID is required for image upload.")
            return
        with open(image_path, 'rb') as img_file:
            files = {'violate_photo': (image_path, img_file, 'image/png')}
            response = self.session.post('http://103.77.209.93:3001/api/violate/add', files=files)
            print(f"Response from API: {response.status_code}, {response.text}")


def initialize_api_handler():
    global api_handler
    LOGIN_API_URL = "http://103.77.209.93:3001/api/login/user-login"
    LOGIN_CREDENTIALS = {
    "email": "nguyn@gmail.com",
    "password": "123456789"
}
    api_handler = APIHandler(LOGIN_API_URL, LOGIN_CREDENTIALS)

first_drowsy_time = None  
consecutive_drowsy_count = 0  
CONSECUTIVE_DROWSY_INTERVAL = 5 * 60  
REPEATED_DROWSY_INTERVAL = 5 * 60  

def save_drowsy_image(frame, detection_type):
    global first_drowsy_time, consecutive_drowsy_count
    
    current_time = datetime.datetime.now()
    
    if first_drowsy_time is None:
        first_drowsy_time = current_time
        consecutive_drowsy_count = 1
    else:
        time_difference = (current_time - first_drowsy_time).total_seconds()
        
        if time_difference < CONSECUTIVE_DROWSY_INTERVAL:
            consecutive_drowsy_count += 1
            return False
        
        else:
            first_drowsy_time = current_time
            consecutive_drowsy_count = 1
            
    return True  

DROWSY_THRESHOLD = 5  
drowsy_start_time = None
last_capture_time = None
CAPTURE_INTERVAL = 5  

def play_yawn_sequence(sound_files, repeat=3):
    for _ in range(repeat):
        if not (yawn_countdown):
            break
        for sound in sound_files:
            if not (yawn_countdown): 
                break
            winsound.PlaySound(sound, winsound.SND_FILENAME)
            time.sleep(0.1)


def play_continuous_alert(stop_event):
    while not stop_event.is_set():
        winsound.PlaySound(alert_sound, winsound.SND_FILENAME)
        time.sleep(0.1)

stop_alert = threading.Event()
alert_thread = None
yawn_thread = None

MOUTH_AR_THRESH = 1.2  

initialize_api_handler()

while (True):
    ret, frame = capture.read()
    if not ret:
        break  

    gray = frame
    rects = detector(gray, 0)
    if len(rects):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        mouth = shape[mStart:mEnd]  

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        leftEAR = eye_aspect_ratio(leftEye)  
        rightEAR = eye_aspect_ratio(rightEye)  
        avgEAR = (leftEAR + rightEAR) / 2.0
        eyeContourColor = (255, 255, 255)

        mouthMAR = mouth_aspect_ratio(mouth)

        if mouthMAR > MOUTH_AR_THRESH: 
            cv2.putText(gray, "Yawn Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            yawn_countdown = 1
            if yawn_thread is None or not yawn_thread.is_alive():
                yawn_thread = threading.Thread(
                    target=play_yawn_sequence, 
                    args=([focus_sound, take_break_sound],),
                    kwargs={'repeat': 3}
                )
                yawn_thread.daemon = True
                yawn_thread.start()
        else:
            yawn_countdown = 0 

        if avgEAR < close_thresh:
            flag += 1
            eyeContourColor = (0, 255, 255)
            if drowsy_start_time is None:
                drowsy_start_time = datetime.datetime.now()
            
            current_time = datetime.datetime.now()
            drowsy_duration = (current_time - drowsy_start_time).total_seconds() if drowsy_start_time else 0
            
            if drowsy_duration >= DROWSY_THRESHOLD:
                if last_capture_time is None or (current_time - last_capture_time).total_seconds() >= CAPTURE_INTERVAL:
                    if save_drowsy_image(frame, "Drowsy State"):
                        timestamp = current_time.strftime('%Y%m%d_%H%M%S')
                        filename = f"{CAPTURE_DIR}/{timestamp}.png"     
                        frame_with_info = frame.copy()
                        
                        info_text = [
                            f"Driver_ID: {DRIVER_INFO['id']}",
                            f"Driver: {DRIVER_INFO['name']}",
                            f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
                            f"Detection: Drowsy State",
                            f"Consecutive Drowsy: {consecutive_drowsy_count}"
                        ]
                        
                        y_position = 30
                        for text in info_text:
                            cv2.putText(frame_with_info, text, (10, y_position),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            y_position += 30
                        
                        cv2.imwrite(filename, frame_with_info)
                        print(f"Saved drowsy detection image: {filename}")
                        api_handler.upload_image(filename)
                        last_capture_time = current_time  
                if alert_thread is None or not alert_thread.is_alive():
                    stop_alert.clear()
                    alert_thread = threading.Thread(target=play_continuous_alert, args=(stop_alert,))
                    alert_thread.daemon = True
                    alert_thread.start()

        else:  
            if flag > 0:  
                flag = 0
                drowsy_start_time = None
                
                if alert_thread and alert_thread.is_alive():
                    stop_alert.set()
                    alert_thread = None
                
                yawn_countdown = 0
                if yawn_thread and yawn_thread.is_alive():
                    yawn_thread = None

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)

    cv2.imshow('Driver', gray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

if alert_thread and alert_thread.is_alive():
    stop_alert.set()
if yawn_thread and yawn_thread.is_alive():
    yawn_countdown = 0
