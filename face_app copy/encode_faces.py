import face_recognition
import os
import pickle
from PIL import Image
import numpy as np

# Prevent PIL large image crash
Image.MAX_IMAGE_PIXELS = None

dataset_path = "dataset"

known_encodings = []
known_names = []

print("🔍 Scanning dataset...")

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"➡️ Processing: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            img = Image.open(image_path).convert("RGB")

            # Resize very large images
            if img.size[0] * img.size[1] > 10_000_000:
                img.thumbnail((1500, 1500))

            image = np.array(img)

            # Detect face
            face_locations = face_recognition.face_locations(image, model="hog")
            encodings = face_recognition.face_encodings(image, face_locations)

            # ✅ Only allow images with exactly ONE face
            if len(encodings) != 1:
                print(f"⚠️ Skipped (need 1 face): {image_name}")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person_name)

            print(f"✅ Encoded: {image_name}")

        except Exception as e:
            print(f"❌ Skipped {image_name} → {e}")

# Save encodings
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n🎉 Encoding complete!")
print(f"Total faces encoded: {len(known_encodings)}")