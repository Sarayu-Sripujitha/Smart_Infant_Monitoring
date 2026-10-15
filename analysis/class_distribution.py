import os
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/Baby Dataset"

class_counts = {}

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    if os.path.isdir(class_path):
        count = 0

        for root, dirs, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    count += 1

        class_counts[class_name] = count

print("\n========== CLASS DISTRIBUTION ==========")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

print("\nTOTAL:", sum(class_counts.values()))

plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values())

plt.xlabel("Cry Class")
plt.ylabel("Number of Audio Files")
plt.title("Baby Cry Dataset - Class Distribution")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("results/class_distribution.png")

plt.show()