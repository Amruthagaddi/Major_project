import cv2
import os

video_path = "datasets/running/person11_running_d1_uncomp.avi"

output_dir = "output/frames/inspection"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

# Ranges around the frames where you saw the person
ranges = [
    (130, 180),
    (260, 310)
]

for start, end in ranges:

    print(f"Extracting frames {start} to {end}")

    for frame_number in range(start, end + 1, 5):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (320, 240))

        cv2.putText(
            frame,
            f"Frame {frame_number}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        filename = f"frame_{frame_number}.jpg"

        cv2.imwrite(
            os.path.join(output_dir, filename),
            frame
        )

cap.release()

print("Inspection frames created!")