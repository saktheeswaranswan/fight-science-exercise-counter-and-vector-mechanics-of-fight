import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle safely
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Prevent division by zero by returning 0 if either vector is zero-length.
    if norm_ba == 0 or norm_bc == 0:
        return 0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clip the cosine value to the valid range [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def calculate_reaction_force(knee_angle, torso_angle):
    base_force = 75  # Base force in kg
    # Using 1 as minimum knee angle to avoid division by zero;
    # Reaction force increases as knee angle decreases (more flexion) and is modulated by torso rotation.
    reaction_force = base_force * (90 / max(knee_angle, 1)) * np.cos(np.radians(torso_angle))
    return reaction_force

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Get required keypoints from hips to feet
        keypoints = {
            "left_hip": (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                         int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)),
            "right_hip": (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                          int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h)),
            "left_knee": (int(lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w),
                          int(lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h)),
            "right_knee": (int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                           int(lm[mp_pose.PoseLandmark.RIGHT_KNEE].y * h)),
            "left_ankle": (int(lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                           int(lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)),
            "right_ankle": (int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
                            int(lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h)),
            "left_foot": (int(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * w),
                          int(lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * h)),
            "right_foot": (int(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w),
                           int(lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h))
        }
        
        # Calculate knee angles
        left_knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        right_knee_angle = calculate_angle(keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"])
        
        # Calculate torso turn angle using a reference point slightly offset from right_hip
        torso_angle = calculate_angle(keypoints["left_hip"], keypoints["right_hip"],
                                      (keypoints["right_hip"][0] + 1, keypoints["right_hip"][1]))
        
        # Calculate reaction forces at feet
        left_reaction_force = calculate_reaction_force(left_knee_angle, torso_angle)
        right_reaction_force = calculate_reaction_force(right_knee_angle, torso_angle)
        
        # Handle potential NaN values (skip drawing if force is not a number)
        if np.isnan(left_reaction_force):
            left_reaction_force = 0
        if np.isnan(right_reaction_force):
            right_reaction_force = 0
        
        # Draw reaction force vectors at the bottom of the feet in yellow
        left_force_vector = (keypoints["left_foot"][0], keypoints["left_foot"][1] - int(left_reaction_force))
        right_force_vector = (keypoints["right_foot"][0], keypoints["right_foot"][1] - int(right_reaction_force))
        
        cv2.arrowedLine(frame, keypoints["left_foot"], left_force_vector, (0, 255, 255), 5)
        cv2.arrowedLine(frame, keypoints["right_foot"], right_force_vector, (0, 255, 255), 5)
        
        # Draw center of gravity between the thighs (using hips) as a green dot
        center_of_gravity = ((keypoints["left_hip"][0] + keypoints["right_hip"][0]) // 2,
                             (keypoints["left_hip"][1] + keypoints["right_hip"][1]) // 2 + 20)
        cv2.circle(frame, center_of_gravity, 7, (0, 255, 0), -1)
        
        # Draw pose landmarks for visualization
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Biomechanics Visualization', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

