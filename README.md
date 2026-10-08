Face Detection and Recognition using InsightFace & ArcFace

A real-time face detection and recognition system built using Python, OpenCV, and InsightFace. The system detects faces from a webcam, generates facial embeddings using the ArcFace model, compares them with registered face embeddings using Cosine Similarity, and identifies the person when the similarity score is above the configured threshold.

🚀 Project Overview

This project performs real-time face recognition using a computer's webcam.

The system first loads images of known people from the face_recognition/images/ directory. It generates a facial embedding for each registered face using InsightFace.

During webcam operation, the system:

Captures live video frames.
Detects faces using InsightFace.
Generates an embedding for each detected face.
Compares the embedding with registered face embeddings.
Finds the closest matching face using Cosine Similarity.
Identifies the person if the similarity exceeds the threshold.
Displays the person's name and similarity score on the webcam.

If no sufficiently similar face is found, the system displays Unknown.

✨ Features
🎥 Real-time webcam face recognition
👤 Face detection using InsightFace
🧠 Face recognition using ArcFace
🔢 Facial embedding generation
📊 Cosine similarity-based matching
🏷️ Known and Unknown face identification
⚡ Frame skipping for better performance
👥 Support for multiple registered faces
💻 CPU-based inference
📈 Real-time similarity score display
🟩 Face bounding box visualization
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
OpenCV	Webcam access, image processing and visualization
InsightFace	Face detection and recognition
ArcFace	Facial embedding generation
NumPy	Numerical and vector operations
Cosine Similarity	Comparing face embeddings
📂 Project Structure
Face-Recognition/
│
├── face_recognition/
│   │
│   ├── images/
│   │   ├── person1.jpg
│   │   ├── person2.jpg
│   │   └── person3.jpg
│   │
│   └── face_detection.py
│
├── requirements.txt
├── .gitignore
└── README.md
face_recognition/images/

This folder contains the reference images of people who should be recognized.

The filename is automatically used as the person's name.

For example:

face_recognition/images/
├── Sumit.jpg
├── Rahul.jpg
└── Priya.jpg

The system will use:

Sumit
siddhu
satish
aditya

as the corresponding recognition labels.

Tip: Use clear, front-facing images containing one face for better recognition results.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/yourusername/face-recognition.git

Navigate to the project:

cd face-recognition
2. Create a Virtual Environment
Windows
python -m venv .venv

Activate it:

.venv\Scripts\activate
Linux/macOS
python3 -m venv .venv

Activate it:

source .venv/bin/activate
3. Install Dependencies

Install the required packages:

pip install -r requirements.txt

Example requirements.txt:

opencv-python
numpy
insightface
onnxruntime

The project uses CPUExecutionProvider, so onnxruntime is sufficient for CPU inference.

▶️ Running the Project

First, make sure the reference face images are present in:

face_recognition/images/

Then run the Python program:

python face_recognition/face_recognition.py

The webcam will open automatically.

The system will detect and recognize faces in real time.

Exit the application

Press:

Q

to close the webcam window.

🧠 System Workflow
              Reference Images
                     │
                     ▼
             Face Detection
                     │
                     ▼
            ArcFace Embedding
                     │
                     ▼
           Store Face Embeddings
                     │
                     │
                     ▼
              Webcam Input
                     │
                     ▼
             Detect Live Face
                     │
                     ▼
           Generate Embedding
                     │
                     ▼
          Cosine Similarity
                     │
              ┌──────┴──────┐
              │             │
           Match         No Match
              │             │
              ▼             ▼
        Person Name      "Unknown"
🔍 How the Code Works
1. Import Required Libraries
import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

These libraries provide:

Webcam and image processing
File handling
Numerical operations
Face detection and recognition
Vector normalization
2. Configure the Project
KNOWN_DIR = "face_recognition/images/"
THRESHOLD = 0.5
FRAME_SKIP = 5
KNOWN_DIR

Specifies the location of registered face images.

THRESHOLD

Defines the minimum similarity score required to recognize a person.

Similarity >= 0.5 → Recognized
Similarity < 0.5  → Unknown
FRAME_SKIP

Controls how frequently face detection is performed.

FRAME_SKIP = 5

means face detection is performed every fifth frame.

🤖 InsightFace and ArcFace

The project initializes InsightFace using:

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

The buffalo_l model provides the required face analysis capabilities.

The model is prepared with:

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)
👤 Registering Known Faces

The program reads every image from:

face_recognition/images/

Each image is processed using:

faces = app.get(img)

If a face is detected, its embedding is stored:

known_embeddings.append(faces[0].embedding)

The filename is stored as the person's name:

known_names.append(os.path.splitext(file)[0])

For example:

satish.jpg → Satish
Sumit.jpg → Sumit
📐 Cosine Similarity

The project compares face embeddings using Cosine Similarity.

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

Conceptually:

                 A · B
Similarity = ─────────────
             ||A|| × ||B||

The resulting score is used to determine which registered face is most similar to the detected face.

🎯 Face Matching

The system calculates similarity between the live face and every known face:

similarities = [
    cosine_similarity(emb, known_emb)
    for known_emb in known_embeddings
]

It then selects the highest score:

best_idx = np.argmax(similarities)

The corresponding person is selected if the score passes the threshold:

name = (
    known_names[best_idx]
    if best_score >= THRESHOLD
    else "Unknown"
)
⚡ Frame Skipping

Face recognition can be computationally expensive, especially when running on a CPU.

Therefore, the project uses:

FRAME_SKIP = 5

Instead of detecting faces on every frame, the program performs detection periodically and reuses the previous detection results between detection frames.

This helps improve the responsiveness of the webcam application.

📷 Real-Time Output

When a known person is detected, the system displays:

┌─────────────────────┐
│                     │
│      Face           │
│                     │
└─────────────────────┘
      Sumit (0.72)

The bounding box shows the detected face, while the label contains:

Person Name (Similarity Score)

For example:

Sumit (0.72)

If the similarity score is below the threshold:

Unknown (0.43)
📊 Recognition Logic

The recognition process can be summarized as:

Live Face
    │
    ▼
Generate Embedding
    │
    ▼
Compare with Known Embeddings
    │
    ▼
Find Highest Similarity
    │
    ▼
Is Score >= 0.5?
    │
 ┌──┴──┐
 │     │
Yes    No
 │     │
 ▼     ▼
Name  Unknown
🎯 Configuration

The main parameters can be modified according to your requirements.

Parameter	Current Value	Purpose
KNOWN_DIR	face_recognition/images/	Location of registered images
THRESHOLD	0.5	Recognition threshold
FRAME_SKIP	5	Number of frames between detection
det_size	(640, 640)	Face detection resolution
provider	CPU	Hardware execution provider
🔐 Privacy & Security

Facial recognition involves biometric information. For real-world applications:

Obtain consent before registering people's faces.
Store face images and embeddings securely.
Avoid collecting unnecessary personal information.
Do not use the system for unauthorized surveillance.
Follow applicable privacy and data-protection regulations.

This project is intended primarily for educational and development purposes.

🚀 Future Improvements

The current project can be extended with:

Face registration through webcam

Multiple reference images per person

SQLite/PostgreSQL face database

Attendance management

Automatic attendance logging

CSV/Excel attendance export

Web interface using Streamlit

GPU acceleration

Face anti-spoofing

Liveness detection

Recognition history

Unknown-face logging

Automatic threshold calibration

Improved handling of multiple faces

📚 Concepts Demonstrated

This project demonstrates practical implementation of:

Computer Vision
Face Detection
Face Recognition
Deep Learning
Face Embeddings
ArcFace
InsightFace
OpenCV
Cosine Similarity
Real-Time Video Processing
Python
Vector Similarity
💼 Internship Project

This project was developed as part of the CodSoft Artificial Intelligence Internship.

Task: Face Detection and Recognition

The project demonstrates the practical application of computer vision and deep-learning-based face recognition techniques.

👨‍💻 Author

Sumit Santosh Patil

Bachelor of Engineering (B.E.) in Computer Engineering
Dhole Patil College of Engineering, Pune
Expected Graduation: 2027

⭐ Acknowledgements

This project uses the following open-source technologies:

InsightFace
ArcFace
OpenCV
NumPy
📄 License

This project is intended for educational and learning purposes.
