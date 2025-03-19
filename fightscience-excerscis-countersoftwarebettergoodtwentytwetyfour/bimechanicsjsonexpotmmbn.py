import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging
import math
import time
import json

# ---------------------------
# Setup: Suppress Logs and Initialize Modules
# ---------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Initialize MediaPipe Pose (for single-person detection per ROI)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(model_complexity=1,
                             static_image_mode=False,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load MobileNet SSD for person detection
prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"
person_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# ---------------------------
# Video and Force Parameters
# ---------------------------
# Resolution for vertical video (9:16 aspect ratio)
screen_width, screen_height = 420, 640

# Force calculation parameters
weight = 10         # in kg
g = 9.81            # gravitational acceleration (m/s^2)
force_per_leg = (weight * g) / 2  # Force per leg (N)
impulse_force = 20  # Additional impulse force (N)
impulse_duration = 0.5  # Duration for impulse (seconds)
leg_impulse_time = 0.3  # Placeholder initial value (will be updated on keypress)
fist_impulse_time = 0.3  # Placeholder initial value (will be updated on keypress)

# ---------------------------
# Data Export Setup
# ---------------------------
export_data = []  # To store keypoints and joint angles per frame
frame_idx = 0     # Frame counter

# ---------------------------
# Utility Functions
# ---------------------------
def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b given three points a, b, and c.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def draw_angle_arc(frame, a, b, c, angle):
    """
    Draw an arc at joint b representing the angle between vectors (b->a) and (b->c).
    """
    center = tuple(b)
    radius = 30
    start_angle = int(math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])))
    end_angle = start_angle + int(angle)
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)

def process_pose_roi(roi, offset_x, offset_y, original_frame):
    """
    Process a cropped region of interest (ROI) using MediaPipe Pose.
    The keypoints are adjusted to the coordinates of the original frame using offset_x and offset_y.
    Also draws the pose landmarks on the original frame.
    Returns a dictionary of keypoints if detection is successful, else None.
    """
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(roi_rgb)
    if not results.pose_landmarks:
        return None
    
    h_roi, w_roi, _ = roi.shape
    keypoints = {}
    for landmark in mp_pose.PoseLandmark:
        lm = results.pose_landmarks.landmark[landmark.value]
        x = int(lm.x * w_roi) + offset_x
        y = int(lm.y * h_roi) + offset_y
        keypoints[landmark.name] = (x, y)
    
    # Draw the pose landmarks on the original frame (with coordinate offset)
    mp_drawing.draw_landmarks(original_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2))
    return keypoints

# ---------------------------
# Main Loop: Process Video and Detect Multiple People
# ---------------------------
def main():
    global leg_impulse_time, fist_impulse_time, frame_idx, export_data
    cap = cv2.VideoCapture('sdfgdfg.mp4')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate (if needed) and resize the frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (screen_width, screen_height))
        h_frame, w_frame, _ = frame.shape

        # Run MobileNet SSD person detector
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w_frame, h_frame), 127.5)
        person_net.setInput(blob)
        detections = person_net.forward()

        # List to store data for each detected person in the current frame
        persons_data = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            # Check if the detection is of a person and has sufficient confidence
            if confidence > 0.5 and class_id == 15:
                box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
                startX, startY, endX, endY = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w_frame - 1, endX), min(h_frame - 1, endY)

                # Crop ROI for the detected person
                roi = frame[startY:endY, startX:endX]
                if roi.size == 0:
                    continue

                # Process the ROI with MediaPipe Pose and adjust keypoints
                keypoints = process_pose_roi(roi, startX, startY, frame)
                if keypoints is None:
                    continue

                # Calculate joint angles for knees (if keypoints are available)
                joint_angles = {}
                if all(k in keypoints for k in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']):
                    left_angle = calculate_angle(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE'])
                    joint_angles['Left Knee'] = (left_angle, 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')
                if all(k in keypoints for k in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']):
                    right_angle = calculate_angle(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE'])
                    joint_angles['Right Knee'] = (right_angle, 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')

                # Visualize joint angles: draw arcs and arrows
                for joint, (angle, a, b, c) in joint_angles.items():
                    if a in keypoints and b in keypoints and c in keypoints:
                        draw_angle_arc(frame, keypoints[a], keypoints[b], keypoints[c], angle)
                        cv2.arrowedLine(frame, tuple(keypoints[b]), tuple(keypoints[a]), (0, 0, 255), 2)
                        cv2.arrowedLine(frame, tuple(keypoints[b]), tuple(keypoints[c]), (255, 0, 0), 2)

                # Visualize force vectors on feet
                for foot in ['LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
                    if foot in keypoints:
                        foot_x, foot_y = keypoints[foot]
                        total_angle = sum(angle for angle, _, _, _ in joint_angles.values()) if joint_angles else 0
                        resultant_force = int(force_per_leg * total_angle / 180)
                        if time.time() - leg_impulse_time < impulse_duration:
                            resultant_force += impulse_force
                        cv2.arrowedLine(frame, (foot_x, foot_y), (foot_x, foot_y - resultant_force), (0, 0, 255), 3)

                # Visualize hand force vectors
                for wrist in ['LEFT_WRIST', 'RIGHT_WRIST']:
                    if wrist in keypoints:
                        wrist_x, wrist_y = keypoints[wrist]
                        impulse_offset = impulse_force if time.time() - fist_impulse_time < impulse_duration else 0
                        cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x + 30 + impulse_offset, wrist_y), (255, 0, 0), 3)
                        cv2.arrowedLine(frame, (wrist_x, wrist_y), (wrist_x, wrist_y - 30 - impulse_offset), (0, 255, 0), 3)

                # Visualize kinetic linking (moving rings) between hip and ankle for each knee
                for joint, (_, a, b, c) in joint_angles.items():
                    if a in keypoints and c in keypoints:
                        mid_x = (keypoints[a][0] + keypoints[c][0]) // 2
                        mid_y = (keypoints[a][1] + keypoints[c][1]) // 2
                        cv2.circle(frame, (mid_x, mid_y), 10, (0, 255, 0), -1)

                # Display joint angle values near the detected person’s bounding box
                y_offset = startY + 20
                for joint, (angle, _, _, _) in joint_angles.items():
                    cv2.putText(frame, f"{joint}: {int(angle)} deg", (startX, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20

                # Save data for this person in the current frame
                persons_data.append({
                    "person_id": i,
                    "keypoints": keypoints,
                    "joint_angles": {k: v[0] for k, v in joint_angles.items()}
                })

        # Append the frame data to the export list
        export_data.append({"frame": frame_idx, "persons": persons_data})
        
        # Display the annotated frame
        cv2.imshow("Multiple Person Biomechanics Visualization", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            leg_impulse_time = time.time()
        elif key == ord('f'):
            fist_impulse_time = time.time()
            
        frame_idx += 1
        # Save export file every 100 frames
        if frame_idx % 100 == 0:
            with open("output_coordinates.json", "w") as outfile:
                json.dump(export_data, outfile, indent=4)
                
    cap.release()
    cv2.destroyAllWindows()
    # Final export once the video processing is complete
    with open("output_coordinates.json", "w") as outfile:
        json.dump(export_data, outfile, indent=4)

if __name__ == "__main__":
    main()

