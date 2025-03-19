import cv2
import numpy as np
import math
import json

# Set resolution (width, height)
screen_width, screen_height = 840, 420

# Load pre-saved keypoints timeline from JSON file
with open("output_coordinatilllies.json", "r") as f:
    frames_data = json.load(f)

# Sort frames by frame index (if not already sorted)
frames_data = sorted(frames_data, key=lambda x: x["frame"])

# Define custom skeleton connections
skeleton_connections = [
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE')
]

# Function to calculate the angle (in degrees) between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0
    cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

# Function to draw an arc representing the angle at joint 'b'
def draw_angle_arc(frame, a, b, c, angle):
    center = tuple(b)
    radius = 20  # Adjust for visualization
    start_angle = int(math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])))
    end_angle = start_angle + int(angle)
    cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 2)

# Function to draw a vector arrow between two points
def draw_vector_arrow(frame, p1, p2, color=(255, 255, 255)):
    cv2.arrowedLine(frame, tuple(p1), tuple(p2), color, 2)

# Open live webcam feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

json_index = 0
num_frames = len(frames_data)

while cap.isOpened():
    ret, live_frame = cap.read()
    if not ret:
        break
    
    # Resize live frame to desired resolution
    live_frame = cv2.resize(live_frame, (screen_width, screen_height))
    
    # Get next pre-saved skeleton frame from JSON (cycling through)
    skeleton_data = frames_data[json_index]
    json_index = (json_index + 1) % num_frames
    keypoints = skeleton_data["keypoints"]
    
    # Draw each keypoint as a small green circle
    for name, coord in keypoints.items():
        cv2.circle(live_frame, tuple(coord), 4, (0, 255, 0), -1)
    
    # Draw skeleton connections using arrowed lines (vector reconstruction)
    for pt1, pt2 in skeleton_connections:
        if pt1 in keypoints and pt2 in keypoints:
            draw_vector_arrow(live_frame, keypoints[pt1], keypoints[pt2])
    
    # Calculate and draw left knee angle if available
    if all(k in keypoints for k in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']):
        left_angle = calculate_angle(keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE'])
        draw_angle_arc(live_frame, keypoints['LEFT_HIP'], keypoints['LEFT_KNEE'], keypoints['LEFT_ANKLE'], left_angle)
        cv2.putText(live_frame, f"Left Knee: {int(left_angle)} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Calculate and draw right knee angle if available
    if all(k in keypoints for k in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']):
        right_angle = calculate_angle(keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE'])
        draw_angle_arc(live_frame, keypoints['RIGHT_HIP'], keypoints['RIGHT_KNEE'], keypoints['RIGHT_ANKLE'], right_angle)
        cv2.putText(live_frame, f"Right Knee: {int(right_angle)} deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Overlay timeline info from JSON
    cv2.putText(live_frame, f"Frame: {skeleton_data['frame']}", (10, screen_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display the live frame with overlaid skeleton reconstruction
    cv2.imshow("Live Webcam with Pre-saved Skeleton", live_frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

