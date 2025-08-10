import os
import random
import argparse
import json
from PIL import Image, ImageOps, ImageFont, ImageDraw
import numpy as np
import cv2
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from rembg import remove
from ultralytics import YOLO

# --- Device Configuration ---
# Auto-detect the best available device for PyTorch models
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# --- Core Implementation ---

def _color_transfer(source, target):
    """Transfers the color distribution from the source to the target image using LAB color space."""
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, a_mean_src, b_mean_src) = source.mean(axis=0).mean(axis=0)
    (l_std_src, a_std_src, b_std_src) = source.std(axis=0).std(axis=0)
    (l_mean_tar, a_mean_tar, b_mean_tar) = target.mean(axis=0).mean(axis=0)
    (l_std_tar, a_std_tar, b_std_tar) = target.std(axis=0).std(axis=0)

    (l, a, b) = cv2.split(target)
    l -= l_mean_tar
    a -= a_mean_tar
    b -= b_mean_tar

    l = (l_std_src / l_std_tar) * l
    a = (a_std_src / a_std_tar) * a
    b = (b_std_src / b_std_tar) * b

    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    return transfer

def _get_full_mask_from_detection(detection_data, frame_width, frame_height):
    """
    Converts a detection's instance mask (from JSON) into a full-frame binary mask.
    """
    full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if detection_data['mask'] is not None:
        x_rel, y_rel, w_rel, h_rel = detection_data['bounding_box']
        x = int(x_rel * frame_width)
        y = int(y_rel * frame_height)
        w = int(w_rel * frame_width)
        h = int(h_rel * frame_height)
        
        instance_mask_list = detection_data['mask']
        instance_mask = np.array(instance_mask_list, dtype=np.uint8)
        
        if instance_mask.size == 0 or w <= 0 or h <= 0:
            return full_mask

        resized_instance_mask = cv2.resize(instance_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        full_mask[y:y+h, x:x+w] = resized_instance_mask * 255
    return full_mask


def insert_object(scene_path: str, annotation_path: str, object_path: str, output_path: str, device: str):
    """
    Inserts an object from an image into a scene using a hierarchical semantic placement pipeline.
    """
    # --- Model Loading ---
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    pose_model = YOLO('yolov8n-pose.pt')

    # --- Step 1 & 2: Isolate Object and Get Semantic Label ---
    object_pil = Image.open(object_path)
    isolated_object_pil = remove(object_pil)
    object_label = classifier(object_pil)[0]['label'].split(',')[0]
    WEARABLE_ITEMS = ['cap', 'hat', 'sunglasses', 'watch', 'backpack', 'bonnet', 'helmet']

    # --- Step 3: Analyze Scene using Local Annotations ---
    scene_pil = Image.open(scene_path)
    with open(annotation_path, 'r') as f:
        detections = json.load(f)
    
    # *** FIX: Filter out small detections before processing ***
    min_area = (scene_pil.width * scene_pil.height) * 0.01 # Must be at least 1% of image area
    
    valid_detections = []
    for det in detections:
        _, _, w_rel, h_rel = det['bounding_box']
        area = (w_rel * scene_pil.width) * (h_rel * scene_pil.height)
        if area > min_area:
            valid_detections.append(det)

    if not valid_detections:
        print("No sufficiently large objects found in the scene. Cannot proceed.")
        return

    scene_labels = [det['label'] for det in valid_detections]

    # --- Step 4: Find Best Semantic Placement Target ---
    obj_embedding = st_model.encode(object_label)
    scene_embeddings = st_model.encode(scene_labels)
    similarities = util.cos_sim(obj_embedding, scene_embeddings)
    best_match_idx = torch.argmax(similarities).item()
    
    target_entity_label = scene_labels[best_match_idx]
    target_detection = valid_detections[best_match_idx]
    target_surface_mask_np = _get_full_mask_from_detection(target_detection, scene_pil.width, scene_pil.height)
    
    # --- Step 5: Refine Placement (Hierarchical Logic) ---
    is_wearable = any(item in object_label for item in WEARABLE_ITEMS)

    if is_wearable and target_entity_label == 'person':
        results = pose_model(scene_pil)
        target_person_box = target_detection['bounding_box']
        best_pose = None
        for r in results:
            if r.keypoints and r.keypoints.xy.numel() > 0:
                nose_kp = r.keypoints.xy[0][0]
                if (target_person_box[0] < nose_kp[0]/scene_pil.width < target_person_box[0] + target_person_box[2] and
                    target_person_box[1] < nose_kp[1]/scene_pil.height < target_person_box[1] + target_person_box[3]):
                    best_pose = r
                    break
        
        if best_pose:
            keypoints = best_pose.keypoints.xy[0]
            nose_kp = keypoints[0]
            person_box = cv2.boundingRect(target_surface_mask_np)
            head_width = person_box[2] * 0.25
            head_height = person_box[3] * 0.15
            center_x = int(nose_kp[0])
            center_y = int(nose_kp[1] - head_height * 0.25)
            placement_box = (center_x, center_y, int(head_width), int(head_height))
        else:
            (x, y, w, h) = cv2.boundingRect(target_surface_mask_np)
            center_x, center_y = x + w // 2, y + h // 6
            placement_box = (center_x, center_y, w, h)
    else:
        dist_transform = cv2.distanceTransform(target_surface_mask_np, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
        center_x, center_y = max_loc
        placement_box = cv2.boundingRect(target_surface_mask_np)

    # --- Step 6: Apply Geometric & Color Transforms ---
    (box_x, box_y, box_w, box_h) = placement_box
    scale_ratio = 0.8 if is_wearable and target_entity_label == 'person' else 0.4
    new_width = int(box_w * scale_ratio)
    aspect_ratio = isolated_object_pil.height / isolated_object_pil.width if isolated_object_pil.width > 0 else 1
    new_height = int(new_width * aspect_ratio)
    
    if new_width <= 10 or new_height <= 10:
        print("Invalid placement size (too small). Cannot proceed.")
        return

    resized_object_pil = isolated_object_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    scene_np = cv2.cvtColor(np.array(scene_pil), cv2.COLOR_RGB2BGR)
    object_bgr_pil = resized_object_pil.convert("RGB")
    object_np = cv2.cvtColor(np.array(object_bgr_pil), cv2.COLOR_RGB2BGR)
    
    crop_x1, crop_y1 = max(0, center_x - box_w//4), max(0, center_y - box_h//4)
    crop_x2, crop_y2 = min(scene_np.shape[1], center_x + box_w//4), min(scene_np.shape[0], center_y + box_h//4)
    scene_crop = scene_np[crop_y1:crop_y2, crop_x1:crop_x2]

    if scene_crop.size > 0 and object_np.size > 0:
        color_corrected_object_np = _color_transfer(scene_crop, object_np)
    else:
        color_corrected_object_np = object_np

    # --- Step 7: Blend and Finalize ---
    object_mask_np = np.array(resized_object_pil.split()[3]) 
    
    final_center_x = min(max(new_width // 2, center_x), scene_np.shape[1] - new_width // 2)
    final_center_y = min(max(new_height // 2, center_y), scene_np.shape[0] - new_height // 2)

    final_image_np = cv2.seamlessClone(
        color_corrected_object_np, scene_np, object_mask_np,
        (final_center_x, final_center_y), cv2.NORMAL_CLONE
    )

    final_image_pil = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
    final_image_pil.save(output_path)
    print(f"Successfully inserted '{object_label}' into scene. Saved to {output_path}")


def insert_text(scene_path: str, annotation_path: str, text_to_insert: str, output_path: str, device: str):
    """
    Finds a suitable surface in a scene and inserts text onto it with perspective warping.
    """
    # --- Step 1: Find a Suitable "Writable" Surface ---
    scene_pil = Image.open(scene_path)
    scene_np = cv2.cvtColor(np.array(scene_pil), cv2.COLOR_RGB2BGR)
    with open(annotation_path, 'r') as f:
        detections = json.load(f)
        
    best_surface = None
    max_score = -1

    for det in detections:
        # *** FIX: Filter out small detections ***
        _, _, w_rel, h_rel = det['bounding_box']
        area_pixels = (w_rel * scene_pil.width) * (h_rel * scene_pil.height)
        if area_pixels < 5000: continue

        mask_np = _get_full_mask_from_detection(det, scene_pil.width, scene_pil.height)
        (x, y, w, h) = cv2.boundingRect(mask_np.astype(np.uint8))
        surface_region = cv2.cvtColor(scene_np[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(surface_region, cv2.CV_64F).var()
        
        score = area_pixels / (lap_var + 1e-5) 

        if score > max_score:
            max_score = score
            best_surface = mask_np

    if best_surface is None:
        print("No suitable surface found for text insertion.")
        return

    # --- Step 2: Render Text Image ---
    try:
        font = ImageFont.truetype("arial.ttf", size=80)
    except IOError:
        font = ImageFont.load_default()
    
    text_size = font.getbbox(text_to_insert)
    text_img = Image.new('RGBA', (text_size[2], text_size[3]), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_img)
    draw.text((0, 0), text_to_insert, font=font, fill=(50, 50, 50, 255))

    # --- Step 3: Calculate and Apply Perspective Transform ---
    rect = cv2.minAreaRect(best_surface.astype(np.uint8))
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    src_pts = np.array([[0, 0], [text_img.width, 0], [text_img.width, text_img.height], [0, text_img.height]], dtype="float32")
    
    dst_pts = np.array(sorted(box, key=lambda x: x[0]), dtype="float32")
    if dst_pts[0][1] > dst_pts[1][1]: dst_pts[[0,1]] = dst_pts[[1,0]]
    if dst_pts[2][1] < dst_pts[3][1]: dst_pts[[2,3]] = dst_pts[[3,2]]

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_text_np = cv2.warpPerspective(np.array(text_img), M, (scene_np.shape[1], scene_np.shape[0]))

    # --- Step 4: Blend and Finalize ---
    warped_alpha = warped_text_np[:, :, 3] / 255.0
    warped_rgb = warped_text_np[:, :, :3]

    for c in range(0, 3):
        scene_np[:, :, c] = scene_np[:, :, c] * (1 - warped_alpha) + warped_rgb[:, :, c] * warped_alpha

    final_image_pil = Image.fromarray(cv2.cvtColor(scene_np, cv2.COLOR_BGR2RGB))
    final_image_pil.save(output_path)
    print(f"Successfully inserted text '{text_to_insert}'. Saved to {output_path}")


# --- Main Execution ---

def main():
    """
    Main function to drive the synthetic data generation process from a local dataset.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic data by inserting objects or text into images.")
    parser.add_argument("--mode", type=str, choices=['object', 'text'], default='object', help="Insertion mode.")
    args = parser.parse_args()

    DEFAULT_TEXT = "RESARO"
    DEFAULT_OBJECT_IMAGE_PATH = "./input_assets/cap.png"
    OUTPUT_DIR = "./output/"
    INPUT_ASSETS_DIR = "./input_assets/"
    LOCAL_DATASET_DIR = "./coco_subset/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_ASSETS_DIR, exist_ok=True)

    if not os.path.exists(DEFAULT_OBJECT_IMAGE_PATH):
        dummy_img = Image.new('RGBA', (100, 80), (255, 0, 0, 255))
        dummy_img.save(DEFAULT_OBJECT_IMAGE_PATH)

    # --- Load from local dataset ---
    images_dir = os.path.join(LOCAL_DATASET_DIR, "images")
    annotations_dir = os.path.join(LOCAL_DATASET_DIR, "annotations")

    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"Error: Local dataset not found or empty in '{LOCAL_DATASET_DIR}'.")
        print("Please run the 'export_coco_subset.py' script first.")
        return

    random_image_name = random.choice(os.listdir(images_dir))
    scene_path = os.path.join(images_dir, random_image_name)
    annotation_name = os.path.splitext(random_image_name)[0] + ".json"
    annotation_path = os.path.join(annotations_dir, annotation_name)

    if args.mode == 'object':
        output_filename = f"object_{random_image_name}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        insert_object(
            scene_path=scene_path,
            annotation_path=annotation_path,
            object_path=DEFAULT_OBJECT_IMAGE_PATH,
            output_path=output_path,
            device=DEVICE
        )
    
    elif args.mode == 'text':
        output_filename = f"text_{random_image_name}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        insert_text(
            scene_path=scene_path,
            annotation_path=annotation_path,
            text_to_insert=DEFAULT_TEXT,
            output_path=output_path,
            device=DEVICE
        )
        
    print("\nScript finished.")

if __name__ == "__main__":
    main()
