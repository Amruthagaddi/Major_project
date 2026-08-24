import cv2
import os

# -----------------------------
# Input video
# -----------------------------
video_path = "datasets/running/person11_running_d1_uncomp.avi"

# -----------------------------
# Output folder
# -----------------------------
output_dir = "output/frames/running/person11_running_d1"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)

# -----------------------------
# Number of frames to extract
# -----------------------------
num_frames = 20

# Calculate frame positions
frame_indices = [
    int(i * total_frames / num_frames)
    for i in range(num_frames)
]

# -----------------------------
# Extract frames
# -----------------------------
for i, frame_index in enumerate(frame_indices):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame:", frame_index)
        continue

    # Resize frame
    frame = cv2.resize(frame, (224, 224))

    # Save frame
    output_path = os.path.join(
        output_dir,
        f"frame_{i+1:02d}.jpg"
    )

    cv2.imwrite(output_path, frame)

    print("Saved:", output_path)

cap.release()

print("Frame extraction completed!")