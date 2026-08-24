import cv2

video_path = "datasets/running/person11_running_d1_uncomp.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    cv2.imshow("KTH Running Video", frame)

    # Press ESC to stop
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()