import os
import json
import shutil
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

# --- Configuration ---
NUM_SAMPLES = 50
OUTPUT_DIR = "./coco_subset/"

def export_subset():
    """
    Downloads a subset of the COCO-2017 dataset and exports the images and
    their available segmentation annotations to a local directory.
    """
    print("Loading COCO dataset to export a local subset...")

    # Ensure the output directory is clean
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    annotations_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(images_dir)
    os.makedirs(annotations_dir)

    # Load the COCO-2017 validation split
    dataset = foz.load_zoo_dataset("coco-2017", split="validation")

    # A simple filter: just get samples that have *any* ground truth annotation
    annotated_view = dataset.exists("ground_truth")
    print(f"Found {len(annotated_view)} samples with annotations.")

    # Take a random sample
    sample_view = annotated_view.shuffle().take(NUM_SAMPLES)

    print(f"Exporting {len(sample_view)} samples to {OUTPUT_DIR}...")

    # Iterate over the random sample to save images and annotations
    for sample in sample_view:
        image_path = sample.filepath
        image_filename = os.path.basename(image_path)

        # Copy image to the new output directory
        shutil.copy(image_path, os.path.join(images_dir, image_filename))

        # Prepare annotations for JSON export
        annotations_data = []
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                # SIMPLE FIX: If a specific annotation has no mask, just skip it
                if det.mask is None:
                    continue

                # If a mask exists, process it
                mask_list = det.mask.tolist()
                annotations_data.append({
                    "label": det.label,
                    "bounding_box": det.bounding_box,
                    "mask": mask_list
                })

        # Write the collected annotations to a corresponding JSON file
        json_filename = os.path.splitext(image_filename)[0] + ".json"
        json_path = os.path.join(annotations_dir, json_filename)

        with open(json_path, 'w') as f:
            json.dump(annotations_data, f, indent=4)

    print("Export complete.")
    print(f"Data saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    export_subset()