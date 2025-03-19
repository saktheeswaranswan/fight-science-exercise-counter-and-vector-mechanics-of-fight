import cv2
import mediapipe as mp
import numpy as np
import os
import absl.logging
import math
import json

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

# Increase thresholds for better inference
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Initialize MediaPipe Pose with higher confidence thresholds
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1,
                    static_image_mode=False,
                    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture('Sequencebestmaa.mp4')

# Constants for weight and foot reaction force
person_weight = 50  # in kg
g = 9.81          # gravitational acceleration in m/s^2
foot_force_magnitude = int(person_weight * g)  # approx. 490 N

# Function to calculate angle between three points (in degrees)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Function to compute the reaction vector from knee to ankle.
# Returns a normalized vector (magnitude 1) and its angle (in degrees).
def compute_reaction_vector(knee, ankle):
    vector = (ankle[0] - knee[0], ankle[1] - knee[1])
    norm = math.sqrt(vector[0]**2 + vector[1]**2)
    if norm == 0:
        return (0, 0), 0
    normalized_vector = (vector[0] / norm, vector[1] / norm)
    angle = math.degrees(math.atan2(normalized_vector[1], normalized_vector[0]))
    return normalized_vector, angle

# Function to draw an arc around a joint showing the measured angle.
def draw_angle_arc(frame, joint, p1, p2, color, radius=30, thickness=2):
    angle1 = math.degrees(math.atan2(p1[1]-joint[1], p1[0]-joint[0])) % 360
    angle2 = math.degrees(math.atan2(p2[1]-joint[1], p2[0]-joint[0])) % 360
    diff = (angle2 - angle1) % 360
    if diff > 180:
        start_angle = angle2
        end_angle = angle1 + 360
    else:
        start_angle = angle1
        end_angle = angle2
    cv2.ellipse(frame, joint, (radius, radius), 0, start_angle, end_angle, color, thickness)

# Button control for video rotation
rotate_video = False

# Set output video resolution (16:9 aspect ratio)
screen_width, screen_height = 1280, 720

frame_idx = 0  # Frame counter
export_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate frame if toggle is active
    if rotate_video:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame, (screen_width, screen_height))
    h, w, _ = frame.shape

    # Process frame with MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Support multiple detections (if available)
    lm_list = []
    if results.pose_landmarks:
        if isinstance(results.pose_landmarks, list):
            lm_list = results.pose_landmarks
        else:
            lm_list = [results.pose_landmarks]

    for person_id, person_landmarks in enumerate(lm_list):
        lm = person_landmarks.landmark
        # Map each landmark name to its pixel coordinates.
        keypoints = {
            landmark.name: (int(lm[landmark].x * w), int(lm[landmark].y * h))
            for landmark in mp_pose.PoseLandmark
        }

        # --- Leg Biomechanics ---
        left_leg_vector, left_leg_angle = compute_reaction_vector(
            keypoints[mp_pose.PoseLandmark.LEFT_KNEE.name],
            keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.name]
        )
        right_leg_vector, right_leg_angle = compute_reaction_vector(
            keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.name],
            keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.name]
        )
        left_knee_angle = calculate_angle(
            keypoints[mp_pose.PoseLandmark.LEFT_HIP.name],
            keypoints[mp_pose.PoseLandmark.LEFT_KNEE.name],
            keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.name]
        )
        right_knee_angle = calculate_angle(
            keypoints[mp_pose.PoseLandmark.RIGHT_HIP.name],
            keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.name],
            keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.name]
        )

        # --- Arm Biomechanics ---
        left_elbow_angle = calculate_angle(
            keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.name],
            keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.name],
            keypoints[mp_pose.PoseLandmark.LEFT_WRIST.name]
        )
        right_elbow_angle = calculate_angle(
            keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.name],
            keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.name],
            keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.name]
        )

        # --- Foot Reaction (Horizontal Force) ---
        left_foot_force_vector = (-foot_force_magnitude, 0)  # leftward force
        right_foot_force_vector = (foot_force_magnitude, 0)  # rightward force
        left_foot_angle = math.degrees(math.atan2(left_foot_force_vector[1], left_foot_force_vector[0]))
        right_foot_angle = math.degrees(math.atan2(right_foot_force_vector[1], right_foot_force_vector[0]))
        left_foot = keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.name]
        right_foot = keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.name]

        # --- Foot Perpendicular Reaction (Force Perpendicular to Foot Orientation) ---
        # Compute foot orientation using heel and foot index landmarks.
        left_heel = keypoints[mp_pose.PoseLandmark.LEFT_HEEL.name]
        right_heel = keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.name]
        # Left foot
        lf_dir = (left_foot[0] - left_heel[0], left_foot[1] - left_heel[1])
        lf_norm = math.sqrt(lf_dir[0]**2 + lf_dir[1]**2)
        if lf_norm != 0:
            lf_dir_norm = (lf_dir[0]/lf_norm, lf_dir[1]/lf_norm)
            # Rotate by 90° (counterclockwise)
            left_foot_perp_vector = (-lf_dir_norm[1] * foot_force_magnitude, lf_dir_norm[0] * foot_force_magnitude)
            left_foot_perp_angle = math.degrees(math.atan2(left_foot_perp_vector[1], left_foot_perp_vector[0]))
        else:
            left_foot_perp_vector, left_foot_perp_angle = (0, 0), 0

        # Right foot
        rf_dir = (right_foot[0] - right_heel[0], right_foot[1] - right_heel[1])
        rf_norm = math.sqrt(rf_dir[0]**2 + rf_dir[1]**2)
        if rf_norm != 0:
            rf_dir_norm = (rf_dir[0]/rf_norm, rf_dir[1]/rf_norm)
            right_foot_perp_vector = (-rf_dir_norm[1] * foot_force_magnitude, rf_dir_norm[0] * foot_force_magnitude)
            right_foot_perp_angle = math.degrees(math.atan2(right_foot_perp_vector[1], right_foot_perp_vector[0]))
        else:
            right_foot_perp_vector, right_foot_perp_angle = (0, 0), 0

        # --- Store Export Data with Unique Person ID ---
        export_data.append({
            "frame": frame_idx,
            "id": person_id,
            "keypoints": keypoints,
            "reaction_vectors": {
                "left_leg": {
                    "vector": left_leg_vector,
                    "vector_angle": left_leg_angle,
                    "knee_angle": left_knee_angle
                },
                "right_leg": {
                    "vector": right_leg_vector,
                    "vector_angle": right_leg_angle,
                    "knee_angle": right_knee_angle
                },
                "foot": {
                    "left": {
                        "vector": left_foot_force_vector,
                        "vector_angle": left_foot_angle,
                        "magnitude": foot_force_magnitude,
                        "perpendicular": {
                            "vector": left_foot_perp_vector,
                            "vector_angle": left_foot_perp_angle,
                            "magnitude": foot_force_magnitude
                        }
                    },
                    "right": {
                        "vector": right_foot_force_vector,
                        "vector_angle": right_foot_angle,
                        "magnitude": foot_force_magnitude,
                        "perpendicular": {
                            "vector": right_foot_perp_vector,
                            "vector_angle": right_foot_perp_angle,
                            "magnitude": foot_force_magnitude
                        }
                    }
                }
            },
            "joint_angles": {
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle
            }
        })

        # --- Visualization ---
        # Draw landmarks and connections (unique color per person)
        color = (np.random.randint(0, 255),
                 np.random.randint(0, 255),
                 np.random.randint(0, 255))
        mp_drawing.draw_landmarks(frame, person_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=color))
        cv2.putText(frame, f"ID: {person_id}", (20, 50 + person_id * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Leg reaction vectors (arrow from knee to ankle)
        left_knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.name]
        right_knee = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.name]
        left_leg_arrow_end = (int(left_knee[0] + left_leg_vector[0] * 50),
                              int(left_knee[1] + left_leg_vector[1] * 50))
        right_leg_arrow_end = (int(right_knee[0] + right_leg_vector[0] * 50),
                               int(right_knee[1] + right_leg_vector[1] * 50))
        cv2.arrowedLine(frame, left_knee, left_leg_arrow_end, (0, 255, 0), 2)
        cv2.arrowedLine(frame, right_knee, right_leg_arrow_end, (0, 255, 0), 2)
        cv2.putText(frame, f"L Leg: {left_leg_vector} @ {int(left_leg_angle)} deg",
                    (left_knee[0]-60, left_knee[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, f"R Leg: {right_leg_vector} @ {int(right_leg_angle)} deg",
                    (right_knee[0]-60, right_knee[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Foot reaction vectors (horizontal force)
        left_foot_arrow_end = (int(left_foot[0] + left_foot_force_vector[0] * 0.1),
                               int(left_foot[1] + left_foot_force_vector[1] * 0.1))
        right_foot_arrow_end = (int(right_foot[0] + right_foot_force_vector[0] * 0.1),
                                int(right_foot[1] + right_foot_force_vector[1] * 0.1))
        cv2.arrowedLine(frame, left_foot, left_foot_arrow_end, (255, 0, 0), 2)
        cv2.arrowedLine(frame, right_foot, right_foot_arrow_end, (255, 0, 0), 2)
        cv2.putText(frame, f"L Foot: {foot_force_magnitude}N, {int(left_foot_angle)} deg",
                    (left_foot[0]-100, left_foot[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(frame, f"R Foot: {foot_force_magnitude}N, {int(right_foot_angle)} deg",
                    (right_foot[0]+20, right_foot[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Foot perpendicular reaction vectors (computed from foot orientation)
        left_foot_perp_arrow_end = (int(left_foot[0] + left_foot_perp_vector[0] * 0.1),
                                    int(left_foot[1] + left_foot_perp_vector[1] * 0.1))
        right_foot_perp_arrow_end = (int(right_foot[0] + right_foot_perp_vector[0] * 0.1),
                                     int(right_foot[1] + right_foot_perp_vector[1] * 0.1))
        cv2.arrowedLine(frame, left_foot, left_foot_perp_arrow_end, (0, 255, 255), 2)
        cv2.arrowedLine(frame, right_foot, right_foot_perp_arrow_end, (0, 255, 255), 2)
        cv2.putText(frame, f"L Foot Perp: {foot_force_magnitude}N, {int(left_foot_perp_angle)} deg",
                    (left_foot[0]-150, left_foot[1]-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(frame, f"R Foot Perp: {foot_force_magnitude}N, {int(right_foot_perp_angle)} deg",
                    (right_foot[0]+20, right_foot[1]-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # Draw joint angle arcs for knee and elbow joints.
        draw_angle_arc(frame,
                       keypoints[mp_pose.PoseLandmark.LEFT_KNEE.name],
                       keypoints[mp_pose.PoseLandmark.LEFT_HIP.name],
                       keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.name],
                       (0, 255, 0), radius=30, thickness=2)
        draw_angle_arc(frame,
                       keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.name],
                       keypoints[mp_pose.PoseLandmark.RIGHT_HIP.name],
                       keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.name],
                       (0, 255, 0), radius=30, thickness=2)
        draw_angle_arc(frame,
                       keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.name],
                       keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.name],
                       keypoints[mp_pose.PoseLandmark.LEFT_WRIST.name],
                       (0, 0, 255), radius=30, thickness=2)
        draw_angle_arc(frame,
                       keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.name],
                       keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.name],
                       keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.name],
                       (0, 0, 255), radius=30, thickness=2)
        cv2.putText(frame, f"L Knee: {int(left_knee_angle)} deg",
                    (left_knee[0]-50, left_knee[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, f"R Knee: {int(right_knee_angle)} deg",
                    (right_knee[0]-50, right_knee[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        left_elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.name]
        right_elbow = keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.name]
        cv2.putText(frame, f"L Elbow: {int(left_elbow_angle)} deg",
                    (left_elbow[0]-50, left_elbow[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(frame, f"R Elbow: {int(right_elbow_angle)} deg",
                    (right_elbow[0]-50, right_elbow[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow('Biomechanics Visualization', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        rotate_video = not rotate_video  # Toggle rotation

    frame_idx += 1
    # Periodically save export data to JSON
    if frame_idx % 100 == 0:
        with open("output_coordinates.json", "w") as outfile:
            json.dump(export_data, outfile, indent=4)

cap.release()
cv2.destroyAllWindows()

# Save the final export data after processing
with open("output_coordinates.json", "w") as outfile:
    json.dump(export_data, outfile, indent=4)

