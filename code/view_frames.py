import cv2
import os
import math

folder = "output/frames/running/person11_running_d1"

files = sorted([
    f for f in os.listdir(folder)
    if f.endswith(".jpg")
])

images = []

for file in files:
    path = os.path.join(folder, file)

    image = cv2.imread(path)

    if image is not None:
        image = cv2.resize(image, (320, 240))

        # Add filename
        cv2.putText(
            image,
            file,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        images.append(image)

# 5 columns × 4 rows
cols = 5
rows = math.ceil(len(images) / cols)

sheet = []

for r in range(rows):
    row_images = images[r * cols:(r + 1) * cols]

    while len(row_images) < cols:
        row_images.append(
            255 * 
            (images[0] * 0 + 1).astype("uint8")
        )

    row = cv2.hconcat(row_images)
    sheet.append(row)

contact_sheet = cv2.vconcat(sheet)

cv2.imwrite(
    "output/frames/running/contact_sheet.jpg",
    contact_sheet
)

print("Contact sheet saved!")