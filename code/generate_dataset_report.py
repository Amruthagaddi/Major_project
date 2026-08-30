import os
import csv
import shutil

ROOT = os.path.join(os.getcwd(), "output/processed_dataset")
OUT_DIR = os.path.join(os.getcwd(), "output/report_samples")
REPORT_CSV = os.path.join(os.getcwd(), "output/dataset_report.csv")

os.makedirs(OUT_DIR, exist_ok=True)

rows = []

if not os.path.exists(ROOT):
    print("Processed dataset not found at:", ROOT)
    raise SystemExit(1)

for split in sorted(os.listdir(ROOT)):
    split_dir = os.path.join(ROOT, split)
    if not os.path.isdir(split_dir):
        continue
    for activity in sorted(os.listdir(split_dir)):
        activity_dir = os.path.join(split_dir, activity)
        if not os.path.isdir(activity_dir):
            continue

        # sequence dirs
        seq_dirs = [d for d in sorted(os.listdir(activity_dir)) if os.path.isdir(os.path.join(activity_dir, d))]
        num_sequences = len(seq_dirs)
        total_frames = 0
        example_image = ""

        for i, seq in enumerate(seq_dirs):
            seq_path = os.path.join(activity_dir, seq)
            # count files in sequence
            files = [f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f))]
            total_frames += len(files)
            if i == 0:
                # pick first image if exists
                candidates = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if candidates:
                    example_image_src = os.path.join(seq_path, candidates[0])
                    example_image_dst = os.path.join(OUT_DIR, f"{split}_{activity}.jpg")
                    try:
                        shutil.copy(example_image_src, example_image_dst)
                        example_image = example_image_dst
                    except Exception:
                        example_image = example_image_src

        rows.append((split, activity, num_sequences, total_frames, example_image))

# write CSV
with open(REPORT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["split", "activity", "num_sequences", "num_frames", "example_image"])
    for r in rows:
        writer.writerow(r)

print("Report written to:", REPORT_CSV)
print("Sample images copied to:", OUT_DIR)
