👤 Smart Visitor Tracking System

A real-time Visitor Recognition System using DeepFace (facial embeddings), OpenCV, and DeepSort tracking.
The system can:

    Detect faces in a video stream

    Assign unique IDs to visitors

    Track entry and exit

    Store facial embeddings in SQLite for recognition

    Log visitor events

📂 Project Structure
        katomaran/
        
            │── main.py           # Entry point: runs detection + tracking + recognition
            │── detector.py       # Handles face detection
            │── tracker.py        # Handles DeepSort tracking
            │── embedder.py       # Extracts embeddings + cosine similarity
            │── db.py             # SQLite database handler
            │── faces.db          # SQLite database (auto-created on first run)
            │── requirements.txt  # Dependencies
            │── README.md         # Documentation

⚙️ Installation
1. Clone repo
       
        git clone https://github.com/<your-repo>/katomaran.git
        cd katomaran

2. Create virtual environment (Windows)
   
        python -m venv venv
        venv\Scripts\activate


(Mac/Linux)

    python3 -m venv venv
    source venv/bin/activate

3. Install dependencies
   
    pip install -r requirements.txt


requirements.txt

    opencv-python
    deepface
    numpy
    sqlite3-binary
    deep-sort-realtime

▶️ Running the System

Run the app with default camera:

    python main.py --config config.yaml


If you don’t have a config, just run:

    python main.py



## Video link:

    https://www.loom.com/share/c1070381140047a8b7b6be069f363521?sid=96e3acf2-e315-477f-8715-38828ee72b1f

# Sample config:

    {
    "video_source": "C:\\ZZZ Genga\\companies\\katomaran\\record_20250620_183903.mp4",
    "output_root": "./outputs",
    "db_path": "./outputs/faces.db",
    "detection_skip_frames": 3,
    "min_face_confidence": 0.35,
    "entry_image_size": [160, 160],
    "tracker_max_age": 30
    }

    
🛠 How It Works

Face Detection

    Uses DeepFace (retinaface backend) to detect faces in frames.

Tracking

    Each detected face is tracked using DeepSort (so the same visitor keeps the same ID while moving).

Embedding Extraction

    Each new face generates a 128-d embedding vector via DeepFace.

Database Storage

    Embeddings and unique visitor UUIDs are stored in SQLite (faces.db).

Recognition

    When a new face appears, its embedding is compared (cosine similarity) with stored embeddings.

    If similarity > threshold (e.g., 0.65), it is recognized as an existing visitor.

    Otherwise, it is registered as a new visitor.

Logging

    Each visitor’s entry and exit is logged in the console.

Future enhancement: log to a CSV or web dashboard.

📊 Database Schema

    SQLite DB: faces.db

Table: faces

    Column    |	Type	|  Description
    
    id        |	INTEGER	|  Auto-increment ID
    
    uuid      |	TEXT	|  Unique visitor UUID
    
    embedding |	BLOB	|  Numpy array (serialized)



✅ Example Console Output

    2025-09-26 22:13:38 | INFO | New face 218ea38d287f4296a7bffba0fb3317f6 track 4
    
    2025-09-26 22:14:10 | INFO | Face 218ea38d287f4296a7bffba0fb3317f6 exited track 4
    
    2025-09-26 22:15:22 | INFO | Recognized visitor (match: 0.82) - UUID: 218ea38d287f4296a7bffba0fb3317f6

🚀 Next Steps / Improvements

 Add GUI (Streamlit/Dash) to display live video + visitor logs

 Store visitor logs (entry/exit time) in SQLite

 Add REST API for integration with other systems

 Optimize performance for large crowds

🙌 Credits

DeepFace
 – Face recognition and embeddings

DeepSort
 – Object tracking

OpenCV
 – Video processing
