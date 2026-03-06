import os
import glob
import random
import shutil

# --- Configuration ---
SOURCE_DIR = '../gap_generation_output'
DEST_DIR = './test_data'

CONFIGURATIONS = [
    "2m_100f", "2m_300f", "2m_500f",
    "6m_100f", "6m_300f", "6m_500f",
    "10m_100f", "10m_300f", "10m_500f",
    "14m_100f", "14m_300f", "14m_500f"
]

FILES_PER_CONFIG = 12


def create_dataset_sample():
    # 1. Ensure destination directory exists
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory: {DEST_DIR}")

    total_copied = 0

    # 2. Iterate through each configuration
    for config in CONFIGURATIONS:
        # Construct search pattern (e.g., "*_2m_100f_c*.csv")
        search_pattern = os.path.join(SOURCE_DIR, f"*_{config}_c*.csv")
        matching_files = glob.glob(search_pattern)

        # 3. Check if we have enough files
        if len(matching_files) < FILES_PER_CONFIG:
            print(f"Error: Not enough files for {config}. Found {len(matching_files)}, need {FILES_PER_CONFIG}.")
            continue

        # 4. Randomly sample exactly 12 files
        selected_files = random.sample(matching_files, FILES_PER_CONFIG)

        # 5. Copy the selected files to the new folder
        for file_path in selected_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(DEST_DIR, filename)

            shutil.copy(file_path, dest_path)
            total_copied += 1

        print(f"Successfully sampled {FILES_PER_CONFIG} files for {config}.")

    print(f"\nDone! {total_copied} total files copied to '{DEST_DIR}'.")


if __name__ == "__main__":
    create_dataset_sample()
