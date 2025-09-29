import logging
import os

def setup_logger(output_root):
    os.makedirs(output_root, exist_ok=True)
    logger = logging.getLogger("FaceTracker")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_root,"events.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
