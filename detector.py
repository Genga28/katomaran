import cv2
import numpy as np
import os

class Detector:
    def __init__(self, min_confidence=0.5):
        folder = os.path.dirname(os.path.abspath(__file__))
        protoPath = os.path.join(folder, "deploy.prototxt")
        modelPath = os.path.join(folder, "res10_300x300_ssd_iter_140000.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        self.min_confidence = min_confidence

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                bboxes.append((x1, y1, x2, y2, float(confidence)))
        return bboxes
