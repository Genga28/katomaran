import os
from datetime import datetime
import cv2
import uuid

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def timestamp():
    return datetime.utcnow().isoformat()

def crop_from_bbox(frame, bbox, pad=0):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2]

def save_cropped(root, event_type, image, entry_image_size=(160, 160)):
    date = datetime.utcnow().strftime('%Y-%m-%d')
    folder = os.path.join(root, f"{event_type}s", date)
    ensure_dir(folder)
    fname = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(folder, fname)
    if entry_image_size:
        image = cv2.resize(image, tuple(entry_image_size))
    cv2.imwrite(path, image)
    return path

def make_face_uuid():
    return uuid.uuid4().hex
