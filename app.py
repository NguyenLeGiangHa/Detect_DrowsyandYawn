import cv2
import math
import numpy as np
import dlib
from imutils import face_utils
import os
import datetime
import threading
import time

import train as train
import winsound

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

    # Camera internals

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return translation_vector[1][0]


def euclideanDist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# EAR -> Eye Aspect ratio
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

close_thresh = 0.3  # (close_avg+open_avg)/2.0
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
    'license': 'ABC123'
}

# Create directory for storing captured images
CAPTURE_DIR = 'captured_drowsy'
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

def save_drowsy_image(frame, detection_type):
    """
    Save the frame with timestamp and driver information
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{CAPTURE_DIR}/drowsy_{DRIVER_INFO['id']}_{timestamp}.jpg"
    
    # Create a copy of the frame to add information
    frame_with_info = frame.copy()
    
    # Add information to the image
    info_text = [
        f"Driver ID: {DRIVER_INFO['id']}",
        f"Name: {DRIVER_INFO['name']}",
        f"License: {DRIVER_INFO['license']}",
        f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Detection: {detection_type}"
    ]
    
    # Add text to image
    y_position = 30
    for text in info_text:
        cv2.putText(frame_with_info, text, (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_position += 30
    
    # Save the image
    cv2.imwrite(filename, frame_with_info)
    print(f"Saved drowsy detection image: {filename}")


DROWSY_THRESHOLD = 5  # seconds before alert triggers
drowsy_start_time = None
last_capture_time = None
CAPTURE_INTERVAL = 5  

# Add this function to play yawn sequence
def play_yawn_sequence(sound_files, repeat=3):
    for _ in range(repeat):
        if not (yawn_countdown):  # Stop if no longer in yawn state
            break
        for sound in sound_files:
            if not (yawn_countdown):  # Stop if no longer in yawn state
                break
            winsound.PlaySound(sound, winsound.SND_FILENAME)
            time.sleep(0.1)

# Add this function to play continuous alert
def play_continuous_alert(stop_event):
    while not stop_event.is_set():
        winsound.PlaySound(alert_sound, winsound.SND_FILENAME)
        time.sleep(0.1)

# Add these variables at the global scope
stop_alert = threading.Event()
alert_thread = None
yawn_thread = None

while (True):
    ret, frame = capture.read()
    size = frame.shape

    gray = frame
    rects = detector(gray, 0)
    if (len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEAR = ear(leftEye)  
        rightEAR = ear(rightEye)  
        avgEAR = (leftEAR + rightEAR) / 2.0
        eyeContourColor = (255, 255, 255)

        if (yawn(shape[mStart:mEnd]) > 0.6):
            cv2.putText(gray, "Yawn Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            yawn_countdown = 1
            # Play yawn sequence in a separate thread if not already playing
            if yawn_thread is None or not yawn_thread.is_alive():
                try:
                    yawn_thread = threading.Thread(
                        target=play_yawn_sequence, 
                        args=([focus_sound, take_break_sound],),
                        kwargs={'repeat': 3}
                    )
                    yawn_thread.daemon = True
                    yawn_thread.start()
                except Exception as e:
                    print(f"Error playing yawn sequence: {e}")
        else:
            yawn_countdown = 0  # Reset yawn state when no yawn detected

        if (avgEAR < close_thresh):
            flag += 1
            eyeContourColor = (0, 255, 255)
            
            if drowsy_start_time is None:
                drowsy_start_time = datetime.datetime.now()
            
            current_time = datetime.datetime.now()
            drowsy_duration = (current_time - drowsy_start_time).total_seconds() if drowsy_start_time else 0
            
            if drowsy_duration >= DROWSY_THRESHOLD:
                if (last_capture_time is None or 
                    (current_time - last_capture_time).total_seconds() >= CAPTURE_INTERVAL):
                    save_drowsy_image(frame, "Drowsy State")
                    last_capture_time = current_time
                
                if alert_thread is None or not alert_thread.is_alive():
                    stop_alert.clear()
                    alert_thread = threading.Thread(target=play_continuous_alert, args=(stop_alert,))
                    alert_thread.daemon = True
                    alert_thread.start()
            
        else:  # Eyes are open
            if flag > 0:  # Only if we were previously in drowsy state
                print("Flag reset to 0")
                flag = 0
                drowsy_start_time = None
                
                # Stop the alert sound
                if alert_thread and alert_thread.is_alive():
                    stop_alert.set()
                    alert_thread = None
                
                # Also stop any yawn sounds if they're still playing
                yawn_countdown = 0
                if yawn_thread and yawn_thread.is_alive():
                    yawn_thread = None

        if (map_counter >= 5):
            map_flag = 1
            map_counter = 0
            winsound.PlaySound(alert_sound, winsound.SND_ASYNC | winsound.SND_FILENAME)

            # webbrowser.open("https://www.google.com/maps/search/hotels+or+motels+near+me")

            

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
    cv2.imshow('Driver', gray)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):

        break

capture.release()
cv2.destroyAllWindows()

# Cleanup at the end of the script
if alert_thread and alert_thread.is_alive():
    stop_alert.set()
if yawn_thread and yawn_thread.is_alive():
    yawn_countdown = 0
