import re

annotation_file = "00sequences.txt"

with open(annotation_file, "r") as file:
    lines = file.readlines()

sequence_count = 0

for line in lines:

    line = line.strip()

    # Ignore comments and empty lines
    if not line:
        continue

    if not line.startswith("person"):
        continue

    # Example:
    # person11_running_d1 frames 1-35, 140-180, 274-310, 415-450

    match = re.match(
        r"(person\d+)_([a-z]+)_(d[1-4])\s+frames\s+(.+)",
        line
    )

    if not match:
        print("Could not parse:", line)
        continue

    person = match.group(1)
    activity = match.group(2)
    condition = match.group(3)
    ranges_text = match.group(4)

    ranges = []

    for part in ranges_text.split(","):

        part = part.strip()

        start, end = map(
            int,
            part.split("-")
        )

        ranges.append((start, end))

    sequence_count += len(ranges)

    print(
        person,
        "|",
        activity,
        "|",
        condition,
        "|",
        ranges
    )

print()
print("Total annotated sequences:", sequence_count)