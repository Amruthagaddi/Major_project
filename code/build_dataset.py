import cv2
import os
import re
import numpy as np
import argparse
import json


# ============================================================
# CONFIGURATION
# ============================================================

ANNOTATION_FILE = "00sequences.txt"

DATASET_DIR = "datasets"

OUTPUT_DIR = "output/processed_dataset"

SEQUENCE_LENGTH = 20

IMAGE_SIZE = (224, 224)


# ============================================================
# ACTIVITY NAMES
# ============================================================

ACTIVITIES = {
    "walking",
    "jogging",
    "running",
    "boxing",
    "handwaving",
    "handclapping"
}


# ------------------------------------------------------------
# HOG person detector (single-frame detector used for cropping)
# ------------------------------------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_person_hog(frame):
    """Detect a single person in a frame using HOG + SVM.

    Returns bbox (x, y, w, h) for the largest detected person, or None.
    """
    if frame is None:
        return None

    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # If no detections, try upscaling the frame once to improve recall
    if len(rects) == 0:
        # upscale 2x
        small = frame
        try:
            up = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
            rects_u, _ = hog.detectMultiScale(up, winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(rects_u) > 0:
                # map first/largest rect back to original coordinates
                areas_u = [w * h for (x, y, w, h) in rects_u]
                idx_u = int(np.argmax(areas_u))
                xu, yu, wu, hu = rects_u[idx_u]
                # scale back
                x = int(xu // 2)
                y = int(yu // 2)
                w = int(wu // 2)
                h = int(hu // 2)
                rects = [(x, y, w, h)]
        except Exception:
            pass

    if len(rects) == 0:
        return None

    # pick largest detection
    areas = [w * h for (x, y, w, h) in rects]
    idx = int(np.argmax(areas))
    x, y, w, h = rects[idx]

    # expand a little to better include full body
    pad_h = int(0.25 * h)
    pad_w = int(0.15 * w)

    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_w, x + w + pad_w)
    y2 = min(frame_h, y + h + pad_h)

    return (x1, y1, x2 - x1, y2 - y1)


def detect_person_bg(frame, background_subtractor):
    """Detect person using background subtraction + morphology + contours.

    Returns bbox (x, y, w, h) or None.
    """
    if frame is None:
        return None

    mask = background_subtractor.apply(frame, learningRate=0.01)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel_large, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h, area))

    if not boxes:
        return None

    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[0] + box[2] for box in boxes)
    max_y = max(box[1] + box[3] for box in boxes)

    detected_height = max_y - min_y

    target_height = max(int(detected_height * 1.35), 85)
    target_width = int(target_height * 0.45)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, min(center_x - target_width // 2, frame_width - target_width))
    y1 = max(0, min(center_y - target_height // 2, frame_height - target_height))

    w1 = min(target_width, frame_width - x1)
    h1 = min(target_height, frame_height - y1)

    if w1 <= 0 or h1 <= 0:
        return None

    return (x1, y1, w1, h1)


# ============================================================
# SUBJECT SPLIT
# ============================================================

TRAIN_SUBJECTS = {
    "person11",
    "person12",
    "person13",
    "person14",
    "person15",
    "person16",
    "person17",
    "person18"
}

VALIDATION_SUBJECTS = {
    "person19",
    "person20",
    "person21",
    "person23",
    "person24",
    "person25",
    "person01",
    "person04"
}

TEST_SUBJECTS = {
    "person22",
    "person02",
    "person03",
    "person05",
    "person06",
    "person07",
    "person08",
    "person09",
    "person10"
}


# ============================================================
# DETERMINE DATASET SPLIT
# ============================================================

def get_split(person):

    if person in TRAIN_SUBJECTS:
        return "train"

    if person in VALIDATION_SUBJECTS:
        return "validation"

    if person in TEST_SUBJECTS:
        return "test"

    return None


# ============================================================
# PARSE ANNOTATION FILE
# ============================================================

def read_annotations():

    annotations = []

    with open(ANNOTATION_FILE, "r") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Example:
            #
            # person11_running_d1
            # frames 1-35, 140-180, 274-310, 415-450

            match = re.match(
                r"(person\d+)_([a-z]+)_(d[1-4])\s+frames\s+(.+)",
                line
            )

            if not match:
                continue

            person = match.group(1)

            activity = match.group(2)

            condition = match.group(3)

            ranges_text = match.group(4)

            if activity not in ACTIVITIES:
                continue

            split = get_split(person)

            if split is None:
                continue

            ranges = []

            for part in ranges_text.split(","):

                part = part.strip()

                start, end = map(
                    int,
                    part.split("-")
                )

                ranges.append(
                    (start, end)
                )

            annotations.append({
                "person": person,
                "activity": activity,
                "condition": condition,
                "ranges": ranges,
                "split": split
            })

    return annotations


# ============================================================
# PROCESS ONE ANNOTATED SEQUENCE
# ============================================================

def extract_sequences(
    video_path,
    person,
    activity,
    condition,
    ranges,
    split
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            "Could not open:",
            video_path
        )

        return 0

    sequence_number = 1

    total_saved = 0

    # initialize per-sequence stats counters
    stats = {
        "total_frames": 0,
        "detected": 0,
        "fallback": 0
    }

    # KLT/template tracker state (per-sequence)
    last_bbox = None
    last_gray = None
    last_template = None
    last_pts = None

    for start, end in ranges:

        length = end - start + 1

        # Need at least 20 frames
        if length < SEQUENCE_LENGTH:

            continue

        # Non-overlapping 20-frame windows
        current_start = start

        while current_start + SEQUENCE_LENGTH - 1 <= end:

            current_end = (
                current_start
                + SEQUENCE_LENGTH
                - 1
            )

            sequence_name = (
                f"{person}_{activity}_{condition}"
                f"_seq{sequence_number:03d}"
            )

            sequence_dir = os.path.join(
                OUTPUT_DIR,
                split,
                activity,
                sequence_name
            )

            os.makedirs(
                sequence_dir,
                exist_ok=True
            )

            frame_number = 1

            # Position video at sequence start and read sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_start - 1)

            for frame_index in range(current_start, current_end + 1):

                ret, frame = cap.read()

                if not ret:

                    print(
                        "Could not read frame:",
                        frame_index,
                        "from",
                        video_path
                    )

                    continue

                stats["total_frames"] += 1
                per_video = globals().setdefault("__per_video_stats", {})
                key = f"{person}_{activity}_{condition}"
                per_video.setdefault(key, {"total_frames": 0, "detected": 0, "fallback": 0})
                per_video[key]["total_frames"] += 1

                # Choose detector per invocation (hog or bg)
                detector = globals().get("__detector_choice", "hog")

                if detector == "bg":

                    # background subtractor must be created per-video
                    bg = globals().get("__bg_subtractor", None)

                    if bg is None:
                        # fallback to hog
                        detected = detect_person_hog(frame)
                    else:
                        detected = detect_person_bg(frame, bg)

                else:

                    # default: hog
                    detected = detect_person_hog(frame)

                # If detection failed, try KLT + template fallback using previous bbox
                if detected is None and last_bbox is not None:

                    # prepare gray frames
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    except Exception:
                        gray = None

                    tracked = False

                    # 1) KLT optical flow on tracked points
                    if last_gray is not None and last_pts is not None and gray is not None:
                        try:
                            new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                                last_gray, gray, last_pts, None,
                                winSize=(15, 15), maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                            )

                            if new_pts is not None and status is not None:
                                good_old = last_pts[status.flatten() == 1]
                                good_new = new_pts[status.flatten() == 1]

                                if len(good_old) >= 3:
                                    # estimate median displacement
                                    disps = good_new - good_old
                                    dx = int(np.median(disps[:, 0]))
                                    dy = int(np.median(disps[:, 1]))

                                    x, y, w, h = last_bbox
                                    frame_h, frame_w = frame.shape[:2]
                                    nx = max(0, min(frame_w - w, x + dx))
                                    ny = max(0, min(frame_h - h, y + dy))
                                    detected = (nx, ny, w, h)
                                    tracked = True
                        except Exception:
                            tracked = False

                    # 2) Template matching fallback within an expanded search window
                    if not tracked and last_template is not None:
                        try:
                            x, y, w, h = last_bbox
                            fh, fw = frame.shape[:2]
                            pad = int(max(w, h) * 1.5)
                            sx = max(0, x - pad)
                            sy = max(0, y - pad)
                            ex = min(fw, x + w + pad)
                            ey = min(fh, y + h + pad)

                            search = frame[sy:ey, sx:ex]
                            if search.size != 0 and last_template.size != 0:
                                search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
                                res = cv2.matchTemplate(search_gray, last_template, cv2.TM_CCOEFF_NORMED)
                                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                                # threshold can be tuned; 0.48 is a reasonable start
                                if max_val > 0.48:
                                    tx, ty = max_loc
                                    nx = sx + tx
                                    ny = sy + ty
                                    detected = (nx, ny, last_template.shape[1], last_template.shape[0])
                                    tracked = True
                        except Exception:
                            tracked = False

                if detected is not None:

                    stats["detected"] += 1
                    # record per-video stat
                    per_video = globals().setdefault("__per_video_stats", {})
                    key = f"{person}_{activity}_{condition}"
                    per_video.setdefault(key, {"total_frames": 0, "detected": 0, "fallback": 0})
                    per_video[key]["detected"] += 1

                    x, y, w, h = detected

                    # safe crop and clamp
                    x = int(max(0, x))
                    y = int(max(0, y))
                    w = int(max(1, w))
                    h = int(max(1, h))

                    # clamp to frame
                    fh, fw = frame.shape[:2]
                    x = min(x, fw - 1)
                    y = min(y, fh - 1)
                    w = min(w, fw - x)
                    h = min(h, fh - y)

                    crop = frame[y:y + h, x:x + w]

                    # update KLT/template state: compute good features in current crop
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        last_template = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        pts = cv2.goodFeaturesToTrack(last_template, maxCorners=50, qualityLevel=0.01, minDistance=4)
                        if pts is not None:
                            pts = pts.reshape(-1, 2)
                            pts[:, 0] += x
                            pts[:, 1] += y
                            last_pts = pts.reshape(-1, 1, 2).astype(np.float32)
                        else:
                            last_pts = None

                        last_gray = gray
                        last_bbox = (x, y, w, h)
                    except Exception:
                        pass

                else:

                    stats["fallback"] += 1
                    per_video = globals().setdefault("__per_video_stats", {})
                    key = f"{person}_{activity}_{condition}"
                    per_video.setdefault(key, {"total_frames": 0, "detected": 0, "fallback": 0})
                    per_video[key]["fallback"] += 1

                    # Fallback: center square crop
                    fh, fw = frame.shape[:2]
                    side = min(fh, fw)
                    cx = fw // 2
                    cy = fh // 2
                    x1 = max(0, cx - side // 2)
                    y1 = max(0, cy - side // 2)
                    crop = frame[y1:y1 + side, x1:x1 + side]

                frame = cv2.resize(crop, IMAGE_SIZE)

                frame_path = os.path.join(
                    sequence_dir,
                    f"frame_{frame_number:02d}.jpg"
                )

                cv2.imwrite(
                    frame_path,
                    frame
                )

                frame_number += 1

            # Only count complete sequences
            if frame_number == SEQUENCE_LENGTH + 1:

                total_saved += 1

            else:

                # Remove incomplete sequence
                for filename in os.listdir(
                    sequence_dir
                ):

                    os.remove(
                        os.path.join(
                            sequence_dir,
                            filename
                        )
                    )

                os.rmdir(sequence_dir)

            sequence_number += 1

            current_start += SEQUENCE_LENGTH

    cap.release()

    # expose last sequence stats for aggregation
    try:
        globals()["__last_sequence_stats"] = stats
    except Exception:
        pass

    return total_saved


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Build processed KTH dataset with person crops")
    parser.add_argument(
        "--detector",
        choices=("hog", "bg", "auto"),
        default="hog",
        help="Detection method: hog (HOG+SVM), bg (background subtractor), auto (hog)"
    )

    args = parser.parse_args()

    detector_choice = args.detector

    print("=" * 60)
    print(f"Using detector: {detector_choice}")
    print("Reading KTH annotations...")

    annotations = read_annotations()

    print("Annotation entries found:", len(annotations))
    print("=" * 60)

    total_sequences = 0

    for index, item in enumerate(
        annotations,
        start=1
    ):

        # make detector choice available to extract_sequences via globals
        globals()["__detector_choice"] = detector_choice

        person = item["person"]

        activity = item["activity"]

        condition = item["condition"]

        ranges = item["ranges"]

        split = item["split"]

        video_filename = (
            f"{person}_{activity}_{condition}"
            "_uncomp.avi"
        )

        video_path = os.path.join(
            DATASET_DIR,
            activity,
            video_filename
        )

        if not os.path.exists(video_path):

            print(
                "Missing video:",
                video_path
            )

            continue

        print(
            f"[{index}/{len(annotations)}]",
            person,
            activity,
            condition,
            split
        )

        # If using background-subtractor, create one per-video and expose it
        if detector_choice == "bg":
            bg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
            globals()["__bg_subtractor"] = bg
        else:
            globals().pop("__bg_subtractor", None)

        saved = extract_sequences(
            video_path,
            person,
            activity,
            condition,
            ranges,
            split
        )

        # Print per-video detection breakdown (HOG vs fallback) if available
        per_video = globals().get("__per_video_stats", {})
        key = f"{person}_{activity}_{condition}"
        if key in per_video:
            pv = per_video[key]
            tf = pv.get("total_frames", 0)
            det = pv.get("detected", 0)
            fb = pv.get("fallback", 0)
            pct_det = (det / tf * 100) if tf > 0 else 0.0
            pct_fb = (fb / tf * 100) if tf > 0 else 0.0
            print(f"  Per-video detection: {det}/{tf} frames detected ({pct_det:.1f}%), fallback {fb} ({pct_fb:.1f}%)")
            # If running with HOG/auto and fallback dominates, recommend bg
            if detector_choice in ("hog", "auto") and tf > 0 and det < tf * 0.5:
                print("  WARNING: HOG missed majority of frames for this video — consider re-running with --detector bg")

        # collect per-run stats if returned via globals (extract_sequences writes per-sequence stats)
        per_run_stats = globals().get("__detection_stats", None)
        if per_run_stats is None:
            globals()["__detection_stats"] = {
                "total_frames": 0,
                "detected": 0,
                "fallback": 0
            }
        # accumulate
        seq_stats = globals().pop("__last_sequence_stats", None)
        if seq_stats:
            s = globals()["__detection_stats"]
            s["total_frames"] += seq_stats.get("total_frames", 0)
            s["detected"] += seq_stats.get("detected", 0)
            s["fallback"] += seq_stats.get("fallback", 0)

        # Clear per-video bg subtractor
        globals().pop("__bg_subtractor", None)

        total_sequences += saved

    # After processing all videos, if detection stats exist, write per-video breakdown
    stats = globals().get("__detection_stats", None)
    if stats is not None:
        # also try to write per-video breakdown if available
        per_video = globals().get("__per_video_stats", None)
        if per_video is not None:
            try:
                with open("output/detection_stats_per_video.json", "w") as pvf:
                    json.dump(per_video, pvf, indent=2)
                print("Wrote per-video detection stats to: output/detection_stats_per_video.json")
            except Exception:
                pass

    print()
    print("=" * 60)

    print(
        "Dataset preparation completed!"
    )

    print(
        "Total 20-frame sequences:",
        total_sequences
    )

    print("=" * 60)

    # Write aggregated detection stats if available
    stats = globals().get("__detection_stats", None)
    if stats is not None:
        os.makedirs(os.path.dirname("output/detection_stats.json"), exist_ok=True)
        try:
            with open("output/detection_stats.json", "w") as sf:
                json.dump(stats, sf, indent=2)
            print("Wrote detection stats to: output/detection_stats.json")
        except Exception:
            pass


if __name__ == "__main__":

    main()