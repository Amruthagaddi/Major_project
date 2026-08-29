import cv2
import os
import re


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

            for frame_index in range(
                current_start,
                current_end + 1
            ):

                # KTH annotation uses frame numbers
                # starting at 1.
                # OpenCV uses frame index starting at 0.

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index - 1
                )

                ret, frame = cap.read()

                if not ret:

                    print(
                        "Could not read frame:",
                        frame_index,
                        "from",
                        video_path
                    )

                    continue

                frame = cv2.resize(
                    frame,
                    IMAGE_SIZE
                )

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

    return total_saved


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print("Reading KTH annotations...")

    annotations = read_annotations()

    print(
        "Annotation entries found:",
        len(annotations)
    )

    print("=" * 60)

    total_sequences = 0

    for index, item in enumerate(
        annotations,
        start=1
    ):

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

        saved = extract_sequences(
            video_path,
            person,
            activity,
            condition,
            ranges,
            split
        )

        total_sequences += saved

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


if __name__ == "__main__":

    main()