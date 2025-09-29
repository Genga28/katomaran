class Tracker:
    def __init__(self, max_age=30):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.deepsort = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        """
        detections: [(x1,y1,x2,y2,conf), ...] from OpenCV DNN
        Returns: [(x1,y1,x2,y2,track_id), ...]
        """
        # Prepare detections in DeepSort format
        # Each detection = [ [cx, cy, w, h], conf ]
        deep_sort_input = []
        for det in detections:
            if len(det) != 5:
                continue
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            deep_sort_input.append([[float(cx), float(cy), float(w), float(h)], float(conf)])

        # If no detections, return empty
        if len(deep_sort_input) == 0:
            return []

        tracks = self.deepsort.update_tracks(deep_sort_input, frame=frame)

        results = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = t.to_ltrb()
            results.append((int(l), int(t_), int(r), int(b), t.track_id))
        return results
