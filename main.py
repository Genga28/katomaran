import argparse
import json
import cv2
import os

from detector import Detector
from embedder import Embedder
from tracker import Tracker
from db import DB
from logger_setup import setup_logger
from utils import crop_from_bbox, save_cropped, make_face_uuid

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    output_root = cfg.get('output_root','./outputs')
    os.makedirs(output_root, exist_ok=True)

    logger = setup_logger(output_root)
    db = DB(cfg.get('db_path', os.path.join(output_root,'faces.db')))
    det = Detector(min_confidence=cfg.get('min_face_confidence',0.35))
    embd = Embedder()
    trk = Tracker(max_age=cfg.get('tracker_max_age',30))

    track_face_map = {}
    cap = cv2.VideoCapture(cfg.get('video_source', 0))
    skip_frames = cfg.get('detection_skip_frames', 3)
    frame_index = 0

    logger.info('Starting processing')
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_index += 1

        detections = det.detect(frame) if frame_index % (skip_frames+1) == 0 else []
        tracks = trk.update(detections, frame)
        current_ids = set()

        for (x,y,x2,y2, track_id) in tracks:
            bbox = (x, y, x2, y2)
            current_ids.add(track_id)
            if track_id not in track_face_map:
                face_crop = crop_from_bbox(frame, bbox)
                embedding = embd.get_embedding(face_crop)
                if embedding is None: continue
                db_embs = db.get_all_embeddings()

                matched_uuid, best_sim = None, -1
                for (uuid, embb) in db_embs:
                    sim = embd.cos_sim(embedding, embb)
                    if sim > best_sim: best_sim, matched_uuid = sim, uuid

                if best_sim >= 0.45 and matched_uuid:
                    face_uuid = matched_uuid
                    logger.info(f'Recognized {face_uuid} track {track_id} (sim={best_sim:.2f})')
                else:
                    face_uuid = make_face_uuid()
                    logger.info(f'New face {face_uuid} track {track_id}')
                    db.register_face(face_uuid, embedding)
                    path = save_cropped(output_root, 'entry', face_crop, tuple(cfg.get('entry_image_size',[160,160])))
                    db.add_event(face_uuid, 'entry', path)
                track_face_map[track_id] = face_uuid

        exited = set(track_face_map.keys()) - current_ids
        for ex in list(exited):
            face_uuid = track_face_map.pop(ex)
            logger.info(f'Face {face_uuid} exited track {ex}')
            dummy = frame[0:50,0:50]  # fallback crop
            path = save_cropped(output_root, 'exit', dummy)
            db.add_event(face_uuid, 'exit', path)

        cv2.putText(frame, f'Unique Count: {db.unique_count()}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
        cv2.imshow('tracker', frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    main(args.config)
