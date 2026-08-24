import cv2

video_path = "datasets/running/person11_running_d1_uncomp.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)
print("Press SPACE when you see the person.")
print("Press ESC to stop.")

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.imshow("KTH Video Scanner", frame)

    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        break

    if key == 32:
        print("Person visible around frame:", frame_number)

cap.release()
cv2.destroyAllWindows()