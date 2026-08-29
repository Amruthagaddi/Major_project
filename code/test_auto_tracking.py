import cv2
import os

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = (
    "datasets/running/"
    "person01_running_d1_uncomp.avi"
)

START_FRAME = 1
END_FRAME = 20

OUTPUT_DIR = "output/auto_tracking_test"

os.makedirs(
    OUTPUT_DIR, 
    exist_ok=True
)


# ============================================================
# CREATE COMPATIBLE OPENCV TRACKER
# ============================================================

def create_tracker():
    """Creates a supported OpenCV tracker instance with fallbacks."""
    tracker_types = [
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.TrackerMIL_create(),
        lambda: getattr(cv2, 'legacy', None).TrackerCSRT_create() if hasattr(cv2, 'legacy') else None,
    ]
    for create_fn in tracker_types:
        try:
            tracker = create_fn()
            if tracker is not None:
                return tracker
        except (AttributeError, Exception):
            continue
    return None


# ============================================================
# DETECT PERSON (FULL BODY)
# ============================================================

def detect_person(frame, background_subtractor):

    # --------------------------------------------------------
    # Background subtraction
    # --------------------------------------------------------

    mask = background_subtractor.apply(
        frame,
        learningRate=0.01
    )

    # --------------------------------------------------------
    # Clean noise
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # Connect separated body parts
    # --------------------------------------------------------

    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    mask = cv2.dilate(
        mask,
        kernel_large,
        iterations=2
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    frame_height, frame_width = frame.shape[:2]

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 30:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        boxes.append(
            (x, y, w, h, area)
        )

    if not boxes:
        return None

    # --------------------------------------------------------
    # Combine all moving body parts
    # --------------------------------------------------------

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[0] + box[2] for box in boxes)
    max_y = max(box[1] + box[3] for box in boxes)

    # --------------------------------------------------------
    # Estimate full body bounding box centered on person
    # --------------------------------------------------------

    detected_height = max_y - min_y

    target_height = max(int(detected_height * 1.35), 85)
    target_width = int(target_height * 0.45)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    x1 = max(0, min(center_x - target_width // 2, frame_width - target_width))
    y1 = max(0, min(center_y - target_height // 2, frame_height - target_height))

    w1 = min(target_width, frame_width - x1)
    h1 = min(target_height, frame_height - y1)

    if w1 <= 0 or h1 <= 0:
        return None

    return (x1, y1, w1, h1)


# ============================================================
# MAIN
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        "Could not open video:",
        VIDEO_PATH
    )

    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = os.path.join(OUTPUT_DIR, "tracked_person.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# ============================================================
# BACKGROUND SUBTRACTOR
# ============================================================

background_subtractor = (
    cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=40,
        detectShadows=False
    )
)


# ============================================================
# TRACKER VARIABLES
# ============================================================

tracker = None

tracking = False

bbox = None

fixed_width = None
fixed_height = None

frame_number = 0


# ============================================================
# PROCESS FRAMES
# ============================================================

while frame_number < END_FRAME:

    ret, frame = cap.read()

    if not ret:

        print(
            "Could not read frame:",
            frame_number + 1
        )

        break

    frame_number += 1

    # ========================================================
    # DETECT PERSON
    # ========================================================

    detected_bbox = detect_person(
        frame,
        background_subtractor
    )

    # ========================================================
    # START / UPDATE TRACKING (DYNAMIC CENTERING)
    # ========================================================

    if detected_bbox is not None:

        if not tracking:

            bbox = detected_bbox
            fixed_width = bbox[2]
            fixed_height = bbox[3]

            print(
                "Person detected at frame:",
                frame_number
            )

            print(
                "Initial bounding box:",
                bbox
            )

            print(
                "Fixed size:",
                fixed_width,
                "x",
                fixed_height
            )

            tracker = create_tracker()
            if tracker is not None:
                try:
                    tracker.init(frame, bbox)
                    tracking = True
                except Exception:
                    tracking = True
            else:
                tracking = True

        else:

            # Smoothly follow person across frames
            sx = int(0.6 * bbox[0] + 0.4 * detected_bbox[0])
            sy = int(0.6 * bbox[1] + 0.4 * detected_bbox[1])
            bbox = (sx, sy, detected_bbox[2], detected_bbox[3])

            if tracker is not None:
                try:
                    tracker.init(frame, bbox)
                except Exception:
                    pass

    elif tracking and tracker is not None:

        success, tracked_bbox = tracker.update(frame)

        if success:

            tracked_x = int(tracked_bbox[0])
            tracked_y = int(tracked_bbox[1])

            frame_height, frame_width = frame.shape[:2]

            new_x = max(0, min(tracked_x, frame_width - (fixed_width or 40)))
            new_y = max(0, min(tracked_y, frame_height - (fixed_height or 85)))

            bbox = (new_x, new_y, fixed_width or 40, fixed_height or 85)

        else:

            tracking = False
            tracker = None
            bbox = None

    # ========================================================
    # DRAW BOUNDING BOX
    # ========================================================

    display = frame.copy()

    if bbox is not None:

        x, y, w, h = bbox

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "PERSON",
            (
                x,
                max(
                    20,
                    y - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            display,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ========================================================
    # FRAME NUMBER
    # ========================================================

    cv2.putText(
        display,
        f"Frame: {frame_number}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SAVE FRAME & WRITE VIDEO
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"frame_{frame_number:02d}.jpg"
    )

    cv2.imwrite(
        output_path,
        display
    )
    out_writer.write(display)

    # ========================================================
    # SHOW
    # ========================================================

    try:
        cv2.imshow(
            "Automatic Person Tracking",
            display
        )
        key = cv2.waitKey(100) & 0xFF
        if key == 27:
            break
    except Exception:
        pass


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out_writer.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass

# Generate GIF output & sync with active conversation artifact directory
try:
    import glob
    import shutil
    from PIL import Image

    frame_paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'frame_*.jpg')))
    if frame_paths:
        gif_path = os.path.join(OUTPUT_DIR, 'tracked_person.gif')
        images = [Image.open(p) for p in frame_paths]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)

        artifact_dir = "/Users/amrutha/.gemini/antigravity-ide/brain/37e2e9f7-6e1c-485f-884e-1f1648e261bd"
        if os.path.exists(artifact_dir):
            shutil.copy(gif_path, os.path.join(artifact_dir, "tracked_person.gif"))
except Exception:
    pass

print()

print(
    "Test completed."
)

print(
    "Frames saved in:",
    OUTPUT_DIR
)