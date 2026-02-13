# n423.py
# English comments only
# Generate scene names from 0001 to 0423

import random

# Generate all 423 scene names (n001 to n423)
all_scenes = [f"n{i:03d}" for i in range(1, 424)]

# Define the number of scenes for the validation set (15% of 423 ≈ 63)
num_val_scenes = int(len(all_scenes) * 0.15)

# Ensure the split is the same every time
random.seed(42)

# Shuffle the list in place
random.shuffle(all_scenes)

# Split the list
val_detect = all_scenes[:num_val_scenes]  # First 63 scenes become validation
train_detect = all_scenes[num_val_scenes:] # Remaining scenes become training

print(f"Total scenes: {len(all_scenes)}")
print(f"Train scenes: {len(train_detect)}")
print(f"Validation scenes: {len(val_detect)}")

print(f"train scene {train_detect}")
print(f"val scene {val_detect}")