import os
import glob
import matplotlib.pyplot as plt

# Define paths based on your data structure
# Class_0 = Non-Oil Spill, Class_1 = Oil Spill
CLASS_0_DIR = os.path.join("data", "Class_0")
CLASS_1_DIR = os.path.join("data", "Class_1")

# Count images
non_oil_images = glob.glob(os.path.join(CLASS_0_DIR, "*.*"))
oil_images = glob.glob(os.path.join(CLASS_1_DIR, "*.*"))

non_oil_count = len(non_oil_images)
oil_count = len(oil_images)
total_images = non_oil_count + oil_count

print(f"Non-Oil Spill (Class_0): {non_oil_count} images")
print(f"Oil Spill (Class_1): {oil_count} images")
print(f"Total Dataset Size: {total_images} images")

# Plot Class Distribution
labels = ['Non-Oil Spill (Class 0)', 'Oil Spill (Class 1)']
sizes = [non_oil_count, oil_count]
colors = ['#66b3ff', '#ff9999']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie Chart
axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Class Distribution (Percentage)')

# Bar Chart
axes[1].bar(labels, sizes, color=colors)
axes[1].set_title('Class Distribution (Count)')
axes[1].set_ylabel('Number of Images')

plt.tight_layout()
plt.show()