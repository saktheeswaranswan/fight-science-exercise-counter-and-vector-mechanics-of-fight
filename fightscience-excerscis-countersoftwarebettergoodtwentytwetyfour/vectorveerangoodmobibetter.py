import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('videoplayback.mp4')

# Screen resolution
screen_width, screen_height = 640, 840
blue_line_y = screen_height // 3  # Blue line
yellow_line_y = (2 * screen_height) // 3  # Yellow line
blue_line_angle = 0
yellow_line_angle = 0
rotation_angle = 0

cross_count = 0  # Counter for pose keypoints crossing lines

def move_line(up, line):
    global blue_line_y, yellow_line_y
    if line == 'blue':
        if up and blue_line_y > 10:
            blue_line_y -= 10
        elif not up and blue_line_y < screen_height - 10:
            blue_line_y += 10
    elif line == 'yellow':
        if up and yellow_line_y > 10:
            yellow_line_y -= 10
        elif not up and yellow_line_y < screen_height - 10:
            yellow_line_y += 10

def rotate_frame(frame, angle):
    center = (screen_width // 2, screen_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, rotation_matrix, (screen_width, screen_height))

def draw_vector(frame, start, angle, magnitude, color):
    end_x = int(start[0] + magnitude * math.cos(math.radians(angle)))
    end_y = int(start[1] - magnitude * math.sin(math.radians(angle)))
    cv2.arrowedLine(frame, start, (end_x, end_y), color, 2)

def draw_elliptical_arc(frame, center, axis_length, angle, start_angle, end_angle, color):
    cv2.ellipse(frame, center, axis_length, angle, start_angle, end_angle, color, 2)

def count_crossings(keypoints):
    global cross_count
    for x, y in keypoints.values():
        if abs(y - blue_line_y) < 5 or abs(y - yellow_line_y) < 5:
            cross_count += 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (screen_width, screen_height))
    frame = rotate_frame(frame, rotation_angle)
    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        keypoints = {landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h)) for landmark in mp_pose.PoseLandmark}
        
        count_crossings(keypoints)
        
        # Draw keypoints
        for x, y in keypoints.values():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw rotated lines
        cv2.line(frame, (0, blue_line_y), (screen_width, blue_line_y), (255, 0, 0), 2)
        cv2.line(frame, (0, yellow_line_y), (screen_width, yellow_line_y), (0, 255, 255), 2)
        
        # Foot vector perpendicular
        if 'LEFT_ANKLE' in keypoints and 'LEFT_KNEE' in keypoints:
            ankle_x, ankle_y = keypoints['LEFT_ANKLE']
            knee_x, knee_y = keypoints['LEFT_KNEE']
            leg_angle = math.degrees(math.atan2(knee_y - ankle_y, knee_x - ankle_x))
            draw_vector(frame, (ankle_x, ankle_y), leg_angle + 90, 50, (255, 0, 0))
            draw_elliptical_arc(frame, (knee_x, knee_y), (30, 15), leg_angle, 0, 180, (255, 255, 0))
        
        # Fist vectors
        if 'LEFT_WRIST' in keypoints and 'LEFT_ELBOW' in keypoints:
            wrist_x, wrist_y = keypoints['LEFT_WRIST']
            elbow_x, elbow_y = keypoints['LEFT_ELBOW']
            arm_angle = math.degrees(math.atan2(elbow_y - wrist_y, elbow_x - wrist_x))
            draw_vector(frame, (wrist_x, wrist_y), arm_angle + 90, 30, (0, 0, 255))
            draw_vector(frame, (wrist_x, wrist_y), arm_angle - 90, 30, (0, 0, 255))
            draw_elliptical_arc(frame, (elbow_x, elbow_y), (25, 10), arm_angle, 0, 180, (0, 255, 255))
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.putText(frame, f'Cross Count: {cross_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Biomechanics Visualization', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        move_line(True, 'blue')
    elif key == ord('2'):
        move_line(False, 'blue')
    elif key == ord('3'):
        move_line(True, 'yellow')
    elif key == ord('4'):
        move_line(False, 'yellow')
    elif key == ord('6'):
        yellow_line_angle = (yellow_line_angle + 10) % 360
    elif key == ord('7'):
        yellow_line_angle = (yellow_line_angle - 10) % 360
    elif key == ord('8'):
        blue_line_angle = (blue_line_angle + 10) % 360
    elif key == ord('9'):
        blue_line_angle = (blue_line_angle - 10) % 360
    elif key == ord('r'):
        rotation_angle = (rotation_angle + 10) % 360

cap.release()
cv2.destroyAllWindows()
