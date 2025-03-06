import os
import uuid
import cv2
import numpy as np
from datetime import datetime
import random
import shutil
import tempfile
import zipfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO

api_router = APIRouter()

# -------------------------------------------------------------------------
# GLOBAL MODELS (adjust paths as needed)
# -------------------------------------------------------------------------
ball_player_rim_model = YOLO("/home/nikos/master0.2/models/ball-player-rim.pt")
court_model = YOLO("/home/nikos/ml_web_app/basketball-tracking-app/backend/models/train4best.pt")

# -------------------------------------------------------------------------
# CLASS NAMES and COLORS
# -------------------------------------------------------------------------
CLASS_NAMES = {
    0: "ball",
    1: "player",
    2: "rim"
}
CLASS_COLORS = {
    0: (0, 0, 255),      # Red
    1: (0, 255, 0),      # Green
    2: (255, 200, 100)   # Light Blue
}

# -------------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------------
def is_inside_court(x, y, court_mask):
    if 0 <= x < court_mask.shape[1] and 0 <= y < court_mask.shape[0]:
        return court_mask[y, x] == 1
    return False

def create_dataset_folders(base_path):
    for folder in ['images', 'annotations', 'bbox']:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

# -------------------------------------------------------------------------
# DETECTION FUNCTION WITH COLOR BBOX + LABEL
# -------------------------------------------------------------------------
def detect_basketball_objects(image: np.ndarray, conf_threshold: float, base_path: str, img_name: str):
    create_dataset_folders(base_path)

    # 1. Save original
    original_path = os.path.join(base_path, 'images', img_name)
    cv2.imwrite(original_path, image)

    # 2. YOLO detection
    results = ball_player_rim_model.predict(image, conf=conf_threshold)

    # 3. YOLO annotations
    annot_path = os.path.join(base_path, 'annotations', img_name.replace('.png', '.txt'))
    with open(annot_path, 'w') as label_file:
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            img_h, img_w = image.shape[:2]

            # YOLO format
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            label_file.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

            # Draw BBox
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Label + confidence
            class_name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
            label_text = f"{class_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_w, text_h = text_size
            text_x = int(x1)
            text_y = int(y1) - 5
            if text_y < 0:
                text_y = int(y2) + text_h + 5

            # Background rect
            cv2.rectangle(image, (text_x, text_y - text_h), (text_x + text_w, text_y), color, -1)
            # White text
            cv2.putText(
                image, label_text, (text_x, text_y - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )

    # 4. Save bounding-boxed
    bbox_path = os.path.join(base_path, 'bbox', img_name)
    cv2.imwrite(bbox_path, image)

# -------------------------------------------------------------------------
# TRACK ENDPOINT
# -------------------------------------------------------------------------
@api_router.post("/track/")
async def track_video(file: UploadFile = File(...)):
    unique_id = uuid.uuid4().hex
    temp_video_path = f"temp_{unique_id}.mp4"

    try:
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not write video file: {e}")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        os.remove(temp_video_path)
        raise HTTPException(status_code=400, detail="Failed to open video.")

    player_positions = []
    ball_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_h, original_w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, 640))
        court_results = court_model(frame_resized)
        if court_results[0].masks is not None:
            court_mask = (court_results[0].masks.data[0].cpu().numpy() > 0.8).astype(np.uint8)
            court_mask = cv2.resize(court_mask, (original_w, original_h))
        else:
            continue

        results = ball_player_rim_model(frame)
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if is_inside_court(cx, cy, court_mask):
                    if int(cls) == 0:
                        ball_positions.append((cx, cy))
                    elif int(cls) == 1:
                        player_positions.append((cx, cy))

    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return {
        "message": "Tracking completed successfully",
        "player_positions": player_positions,
        "ball_positions": ball_positions
    }

# -------------------------------------------------------------------------
# CREATE DATASET FROM IMAGE
# -------------------------------------------------------------------------
@api_router.post("/dataset/create_image/")
async def create_dataset_from_image(
    base_name: str = Form(...),
    conf_threshold: float = Form(0.75),
    file: UploadFile = File(...)
):
    dataset_root = "/home/nikos/ml_web_app/basketball-tracking-app/backend/dataset"
    base_path = os.path.join(dataset_root, base_name)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data received.")

    np_image = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"image_{timestamp}.png"

    detect_basketball_objects(frame, conf_threshold, base_path, img_name)
    return {
        "message": f"Dataset created at {base_path}",
        "image_name": img_name,
        "confidence_threshold_used": conf_threshold
    }

# -------------------------------------------------------------------------
# CREATE DATASET FROM VIDEO
# -------------------------------------------------------------------------
@api_router.post("/dataset/create_video/")
async def create_dataset_from_video(
    base_name: str = Form(...),
    conf_threshold: float = Form(0.75),
    file: UploadFile = File(...)
):
    dataset_root = "/home/nikos/ml_web_app/basketball-tracking-app/backend/dataset"
    base_path = os.path.join(dataset_root, base_name)
    os.makedirs(base_path, exist_ok=True)

    unique_id = uuid.uuid4().hex
    temp_video_path = f"temp_{unique_id}.mp4"
    try:
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not write video file: {e}")

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        os.remove(temp_video_path)
        raise HTTPException(status_code=400, detail="Failed to open the uploaded video.")

    frame_index = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        img_name = f"frame_{frame_index:06d}.png"

        detect_basketball_objects(frame.copy(), conf_threshold, base_path, img_name)
        saved_frames += 1

    cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    return {
        "message": "Video dataset creation complete",
        "base_path": base_path,
        "total_frames_processed": frame_index,
        "frames_saved": saved_frames,
        "conf_threshold_used": conf_threshold
    }

# -------------------------------------------------------------------------
# SPLIT FUNCTION
# -------------------------------------------------------------------------
def create_final_dataset(base_name: str, dataset_root: str,
                         train_ratio: float = 0.7,
                         valid_ratio: float = 0.2,
                         test_ratio: float = 0.1):
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Train/valid/test must sum to 1.0, got {total}")

    temp_dataset_path = os.path.join(dataset_root, base_name)
    if not os.path.isdir(temp_dataset_path):
        raise FileNotFoundError(f"Temp dataset not found: {temp_dataset_path}")

    final_dataset_name = f"{base_name}_final_dataset"
    final_dataset_path = os.path.join(dataset_root, final_dataset_name)
    os.makedirs(final_dataset_path, exist_ok=True)

    # Copy bbox/
    bbox_src = os.path.join(temp_dataset_path, "bbox")
    bbox_dst = os.path.join(final_dataset_path, "bbox")
    if os.path.isdir(bbox_src):
        shutil.copytree(bbox_src, bbox_dst, dirs_exist_ok=True)
    else:
        print(f"[Warning] No bbox folder at {bbox_src}")

    # Gather images in images/ folder
    images_dir = os.path.join(temp_dataset_path, "images")
    annotations_dir = os.path.join(temp_dataset_path, "annotations")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images folder: {images_dir}")
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(f"No annotations folder: {annotations_dir}")

    all_images = [f for f in os.listdir(images_dir)
                  if os.path.isfile(os.path.join(images_dir, f))]
    random.shuffle(all_images)

    n_images = len(all_images)
    train_count = int(n_images * train_ratio)
    valid_count = int(n_images * valid_ratio)
    test_count = n_images - train_count - valid_count

    train_imgs = all_images[:train_count]
    valid_imgs = all_images[train_count:train_count+valid_count]
    test_imgs = all_images[train_count+valid_count:]

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(final_dataset_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(final_dataset_path, split, "annotations"), exist_ok=True)

    def copy_image_annot(img_file, split):
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(final_dataset_path, split, "images", img_file)

        annotation_file = os.path.splitext(img_file)[0] + ".txt"
        src_annot = os.path.join(annotations_dir, annotation_file)
        dst_annot = os.path.join(final_dataset_path, split, "annotations", annotation_file)

        shutil.copy2(src_img, dst_img)
        if os.path.isfile(src_annot):
            shutil.copy2(src_annot, dst_annot)
        else:
            print(f"[Warning] no annotation for {img_file}")

    for img in train_imgs:
        copy_image_annot(img, "train")
    for img in valid_imgs:
        copy_image_annot(img, "valid")
    for img in test_imgs:
        copy_image_annot(img, "test")

    print(f"Created final dataset at {final_dataset_path}")
    print(f"   train: {train_count} images")
    print(f"   valid: {valid_count} images")
    print(f"   test:  {test_count} images")

    return {
        "final_dataset_path": final_dataset_path,
        "train_count": train_count,
        "valid_count": valid_count,
        "test_count": test_count
    }

# -------------------------------------------------------------------------
# FINALIZE (JSON) + FINALIZE ZIP
# -------------------------------------------------------------------------
@api_router.post("/dataset/finalize/")
async def finalize_dataset(
    base_name: str = Form(...),
    train_ratio: float = Form(0.7),
    valid_ratio: float = Form(0.2),
    test_ratio: float = Form(0.1)
):
    """
    Creates {base_name}_final_dataset/ with train/valid/test splits + bbox,
    then returns a JSON response (no zip).
    """
    dataset_root = "/home/nikos/ml_web_app/basketball-tracking-app/backend/dataset"
    try:
        result = create_final_dataset(
            base_name=base_name,
            dataset_root=dataset_root,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )
        return {"message": "Final dataset creation complete", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Make sure to add these imports at the top:
# import tempfile, zipfile
from fastapi.responses import FileResponse

@api_router.post("/dataset/finalize_zip")
def finalize_and_zip_dataset(
    base_name: str = Form(...),
    train_ratio: float = Form(0.7),
    valid_ratio: float = Form(0.2),
    test_ratio: float = Form(0.1)
):
    """
    Creates {base_name}_final_dataset/, then zips it and returns the zip file.
    """
    dataset_root = "/home/nikos/ml_web_app/basketball-tracking-app/backend/dataset"
    try:
        result = create_final_dataset(
            base_name=base_name,
            dataset_root=dataset_root,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )
        final_path = result["final_dataset_path"]  # e.g. /home/nikos/.../my_image_dataset_final_dataset

        # Build a zip in a temp directory
        zip_filename = f"{base_name}_final_dataset.zip"
        import tempfile, zipfile
        temp_zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

        with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(final_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, final_path)
                    zipf.write(full_path, arcname=rel_path)

        return FileResponse(
            temp_zip_path,
            filename=zip_filename,
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





