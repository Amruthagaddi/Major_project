import cv2
import os

# -----------------------------------
# 1. Input video
# -----------------------------------
video_path = "datasets/running/person11_running_d1_uncomp.avi"

# -----------------------------------
# 2. Output folder
# -----------------------------------
output_dir = "output/tracked_videos"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "running_tracked.avi"
)

# -----------------------------------
# 3. Open video
# -----------------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video information:")
print("FPS:", fps)
print("Width:", width)
print("Height:", height)

# -----------------------------------
# 4. Create output video
# -----------------------------------
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# -----------------------------------
# 5. Variables
# -----------------------------------
paused = False
tracking_started = False
tracker = None

# -----------------------------------
# 6. Main video loop
# -----------------------------------
while True:

    # Read next frame only when playing
    if not paused:

        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

    # -----------------------------------
    # Tracking
    # -----------------------------------
    if tracking_started:

        success, bbox = tracker.update(frame)

        if success:

            x, y, w, h = [int(v) for v in bbox]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Tracking",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Tracking Failed",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    else:

        cv2.putText(
            frame,
            "Press SPACE when person appears",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

    # -----------------------------------
    # Save frame
    # -----------------------------------
    out.write(frame)

    # -----------------------------------
    # Display frame
    # -----------------------------------
    cv2.imshow(
        "KTH Human Activity Tracking",
        frame
    )

    # -----------------------------------
    # Keyboard controls
    # -----------------------------------
    key = cv2.waitKey(30) & 0xFF

    # ESC → exit
    if key == 27:
        break

    # SPACE → pause/play
    if key == 32:

        paused = not paused

        # -----------------------------------
        # Select person while paused
        # -----------------------------------
        if paused and not tracking_started:

            print()
            print("Video paused.")
            print("Draw a rectangle around the person.")
            print("Press ENTER or SPACE after selecting.")

            bbox = cv2.selectROI(
                "KTH Human Activity Tracking",
                frame,
                False
            )

            # Check selection
            if bbox[2] > 0 and bbox[3] > 0:

                print("Selected bounding box:", bbox)

                # Create tracker
                tracker = cv2.TrackerCSRT_create()

                # Initialize tracker
                tracker.init(frame, bbox)

                tracking_started = True

                print("Tracking started!")

            else:

                print("Invalid selection.")

            # Continue video
            paused = False


# -----------------------------------
# 7. Release resources
# -----------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("Tracking completed!")
print("Output saved to:")
print(output_path)