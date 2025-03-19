import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Set up the video capture at 1280 x 720.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Lists to store wrist (hand) positions for tracing.
left_hand_trace = []
right_hand_trace = []

# Record the start time (in seconds) for time tracking.
start_time = time.time()

# Function to safely calculate angle between three points.
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to calculate reaction force based on joint angle and torso angle.
def calculate_reaction_force(joint_angle, torso_angle, base_force=75):
    # Increase force as joint angle decreases (more flexion), modulated by torso orientation.
    reaction_force = base_force * (90 / max(joint_angle, 1)) * np.cos(np.radians(torso_angle))
    return reaction_force

# Helper function to draw an arc representing the angle at a joint.
def draw_angle_arc(frame, a, b, c, color=(255, 0, 0), radius=30, thickness=2):
    # b is the vertex.
    angle_a = np.degrees(np.arctan2(a[1]-b[1], a[0]-b[0])) % 360
    angle_c = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0])) % 360

    # Determine the smaller arc.
    if angle_c < angle_a:
        angle_a, angle_c = angle_c, angle_a
    if (angle_c - angle_a) > 180:
        start_angle = angle_c
        end_angle = angle_a + 360
    else:
        start_angle = angle_a
        end_angle = angle_c

    # Draw the arc at point b.
    cv2.ellipse(frame, b, (radius, radius), 0, start_angle, end_angle, color, thickness)

# Helper function to compute the arc length (in pixels) from a list of points.
def compute_arc_length(trace):
    if len(trace) < 2:
        return 0
    length = 0
    for i in range(1, len(trace)):
        pt1 = np.array(trace[i-1])
        pt2 = np.array(trace[i])
        length += np.linalg.norm(pt2 - pt1)
    return length

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the image.
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Define keypoints for arms, legs, and torso.
        keypoints = {
            "left_shoulder": (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                              int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)),
            "right_shoulder": (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                               int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)),
            "left_elbow": (int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w),
                           int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)),
            "right_elbow": (int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w),
                            int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)),
            "left_wrist": (int(lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                           int(lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h)),
            "right_wrist": (int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                            int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)),
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

        # Calculate joint angles.
        left_knee_angle = calculate_angle(keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        right_knee_angle = calculate_angle(keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"])
        left_elbow_angle = calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        right_elbow_angle = calculate_angle(keypoints["right_shoulder"], keypoints["right_elbow"], keypoints["right_wrist"])

        # Compute torso angle using mid-points of hips and shoulders.
        mid_hip = ((keypoints["left_hip"][0] + keypoints["right_hip"][0]) // 2,
                   (keypoints["left_hip"][1] + keypoints["right_hip"][1]) // 2)
        mid_shoulder = ((keypoints["left_shoulder"][0] + keypoints["right_shoulder"][0]) // 2,
                        (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) // 2)
        dx = mid_shoulder[0] - mid_hip[0]
        dy = mid_shoulder[1] - mid_hip[1]
        torso_angle = abs(np.degrees(np.arctan2(dx, dy)))  # 0° means vertical

        # For a tall person (~6.2 ft), you might adjust the base force if desired.
        # Here we simply use the same base force but note that limb lengths may be larger.

        # Calculate reaction forces.
        left_leg_force = calculate_reaction_force(left_knee_angle, torso_angle)
        right_leg_force = calculate_reaction_force(right_knee_angle, torso_angle)
        left_arm_force = calculate_reaction_force(left_elbow_angle, torso_angle)
        right_arm_force = calculate_reaction_force(right_elbow_angle, torso_angle)

        # Draw reaction force vectors (yellow) for legs.
        left_leg_vector = (keypoints["left_foot"][0], keypoints["left_foot"][1] - int(left_leg_force))
        right_leg_vector = (keypoints["right_foot"][0], keypoints["right_foot"][1] - int(right_leg_force))
        cv2.arrowedLine(frame, keypoints["left_foot"], left_leg_vector, (0, 255, 255), 5)
        cv2.arrowedLine(frame, keypoints["right_foot"], right_leg_vector, (0, 255, 255), 5)
        # Draw reaction force vectors (yellow) for arms.
        left_arm_vector = (keypoints["left_wrist"][0], keypoints["left_wrist"][1] - int(left_arm_force))
        right_arm_vector = (keypoints["right_wrist"][0], keypoints["right_wrist"][1] - int(right_arm_force))
        cv2.arrowedLine(frame, keypoints["left_wrist"], left_arm_vector, (0, 255, 255), 5)
        cv2.arrowedLine(frame, keypoints["right_wrist"], right_arm_vector, (0, 255, 255), 5)

        # Draw angle arcs at the knee and elbow joints.
        draw_angle_arc(frame, keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"])
        draw_angle_arc(frame, keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"])
        draw_angle_arc(frame, keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"])
        draw_angle_arc(frame, keypoints["right_shoulder"], keypoints["right_elbow"], keypoints["right_wrist"])

        # Update hand traces for left and right wrists.
        left_hand_trace.append(keypoints["left_wrist"])
        right_hand_trace.append(keypoints["right_wrist"])

        # Optionally, limit the trace length (e.g., last 300 points) to avoid overdraw.
        max_trace_length = 300
        if len(left_hand_trace) > max_trace_length:
            left_hand_trace = left_hand_trace[-max_trace_length:]
        if len(right_hand_trace) > max_trace_length:
            right_hand_trace = right_hand_trace[-max_trace_length:]

        # Draw the hand trace (locus) for left hand.
        for i in range(1, len(left_hand_trace)):
            cv2.line(frame, left_hand_trace[i-1], left_hand_trace[i], (255, 255, 0), 2)
        # Draw the hand trace for right hand.
        for i in range(1, len(right_hand_trace)):
            cv2.line(frame, right_hand_trace[i-1], right_hand_trace[i], (255, 255, 0), 2)

        # Compute arc lengths (total distance traveled by each hand in pixels).
        left_arc_length = compute_arc_length(left_hand_trace)
        right_arc_length = compute_arc_length(right_hand_trace)

        # Compute elapsed time in milliseconds.
        elapsed_time_ms = int((time.time() - start_time) * 1000)

        # Display angles and reaction force values.
        cv2.putText(frame, f"L Knee: {int(left_knee_angle)}", (keypoints["left_knee"][0]-50, keypoints["left_knee"][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"R Knee: {int(right_knee_angle)}", (keypoints["right_knee"][0]-50, keypoints["right_knee"][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"L Elbow: {int(left_elbow_angle)}", (keypoints["left_elbow"][0]-50, keypoints["left_elbow"][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"R Elbow: {int(right_elbow_angle)}", (keypoints["right_elbow"][0]-50, keypoints["right_elbow"][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Torso: {int(torso_angle)}", (mid_shoulder[0]-50, mid_shoulder[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display reaction forces.
        cv2.putText(frame, f"Leg Force L: {int(left_leg_force)}", (keypoints["left_foot"][0]-100, keypoints["left_foot"][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Leg Force R: {int(right_leg_force)}", (keypoints["right_foot"][0]-100, keypoints["right_foot"][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Arm Force L: {int(left_arm_force)}", (keypoints["left_wrist"][0]-100, keypoints["left_wrist"][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Arm Force R: {int(right_arm_force)}", (keypoints["right_wrist"][0]-100, keypoints["right_wrist"][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display hand trace arc lengths and elapsed time.
        cv2.putText(frame, f"L Hand Arc: {int(left_arc_length)} px", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"R Hand Arc: {int(right_arc_length)} px", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed_time_ms} ms", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw center of gravity (green dot) between hips.
        center_of_gravity = ((keypoints["left_hip"][0] + keypoints["right_hip"][0]) // 2,
                             (keypoints["left_hip"][1] + keypoints["right_hip"][1]) // 2 + 20)
        cv2.circle(frame, center_of_gravity, 7, (0, 255, 0), -1)

        # Draw pose landmarks.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Biomechanics Visualization', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

