import cv2
import os

video_path = "datasets/running/person11_running_d1_uncomp.avi"

output_dir = "output/sequences/running/person11_running_d1"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

# Visible segments we identified
segments = [
    (150, 169),
    (280, 299)
]

sequence_number = 1

for start_frame, end_frame in segments:

    sequence_dir = os.path.join(
        output_dir,
        f"sequence_{sequence_number:02d}"
    )

    os.makedirs(sequence_dir, exist_ok=True)

    print(
        f"Creating sequence {sequence_number}: "
        f"frames {start_frame}–{end_frame}"
    )

    frame_count = 1

    for frame_number in range(start_frame, end_frame + 1):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:
            print(
                "Could not read frame:",
                frame_number
            )
            continue

        # Resize for CNN input
        frame = cv2.resize(
            frame,
            (224, 224)
        )

        output_path = os.path.join(
            sequence_dir,
            f"frame_{frame_count:02d}.jpg"
        )

        cv2.imwrite(
            output_path,
            frame
        )

        frame_count += 1

    print(
        f"Saved {frame_count - 1} frames"
    )

    sequence_number += 1

cap.release()

print("Sequence creation completed!")