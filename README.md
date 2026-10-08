# Face Detection and Recognition using InsightFace

A real-time **face detection and recognition system** built with Python, OpenCV, and InsightFace. The system detects faces from a webcam, generates face embeddings using the **ArcFace** recognition model, compares them with registered face embeddings using **Cosine Similarity**, and identifies the person when the similarity score exceeds a predefined threshold.

---

## 📌 Project Overview

This project provides real-time face recognition through a webcam.

The system works by:

1. Loading registered/known face images.
2. Detecting faces in the registered images.
3. Generating facial embeddings using InsightFace.
4. Capturing live video from the webcam.
5. Detecting faces in the video stream.
6. Generating embeddings for detected faces.
7. Comparing live embeddings with known embeddings.
8. Identifying the person using Cosine Similarity.
9. Displaying the person's name and similarity score on the video.

If no registered person matches the detected face above the configured threshold, the system labels the person as **Unknown**.

---

## ✨ Features

* 🎥 Real-time webcam face recognition
* 👤 Face detection using InsightFace
* 🧠 Face recognition using ArcFace
* 🔢 Face embedding generation
* 📊 Cosine similarity-based face matching
* 🏷️ Known/Unknown person identification
* ⚡ Frame skipping for improved performance
* 🖼️ Support for multiple registered faces
* 💻 CPU-based inference
* 📈 Displays recognition confidence/similarity score

---

## 🛠️ Technologies Used

| Technology        | Purpose                                      |
| ----------------- | -------------------------------------------- |
| Python            | Core programming language                    |
| OpenCV            | Webcam access, image processing and display  |
| InsightFace       | Face detection and recognition               |
| ArcFace           | Facial feature/embedding generation          |
| NumPy             | Numerical operations and vector calculations |
| Cosine Similarity | Comparing facial embeddings                  |

---

## 🧠 How It Works

The system follows the following pipeline:

```text
             ┌─────────────────────┐
             │  Known Face Images  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Face Detection     │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ ArcFace Embeddings  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Store Embeddings    │
             └──────────┬──────────┘
                        │
                        │
       ┌────────────────▼────────────────┐
       │        Webcam Video            │
       └────────────────┬────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Detect Live Face    │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Generate Embedding  │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Cosine Similarity   │
             └──────────┬──────────┘
                        │
                ┌───────┴────────┐
                │                │
             Match            No Match
                │                │
                ▼                ▼
        ┌──────────────┐   ┌───────────┐
        │ Person Name  │   │  Unknown  │
        └──────────────┘   └───────────┘
```

---

## 📂 Project Structure

```text
Face-Detection-Recognition/
│
├── task5/
│   │
│   ├── images/
│   │   ├── person1.jpg
│   │   ├── person2.jpg
│   │   └── person3.jpg
│   │
│   └── face_detection.py
│
├── README.md
└── requirements.txt
```

### `task5/images/`

This folder contains the images of people who should be recognized by the system.

The filename is used as the person's name.

For example:

```text
images/
├── Rahul.jpg
├── Sumit.jpg
└── Priya.jpg
```

The system will display:

```text
Rahul
Sumit
Priya
```

when the corresponding faces are recognized.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-detection-recognition.git
```

Move into the project directory:

```bash
cd face-detection-recognition
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
opencv-python
numpy
insightface
onnxruntime
```

For CPU-based execution, `onnxruntime` is sufficient.

---

# ▶️ Running the Project

Make sure your registered face images are inside:

```text
task5/images/
```

Then run:

```bash
python task5/face_detection.py
```

The webcam will open and start detecting and recognizing faces.

To stop the program, press:

```text
Q
```

---

# 🔍 Face Recognition Process

## 1. Load Known Faces

The program reads all images from:

```python
KNOWN_DIR = "task5/images/"
```

Each valid image is processed using InsightFace.

---

## 2. Generate Face Embeddings

InsightFace detects the face and generates a numerical representation called a **face embedding**.

```python
faces = app.get(img)
```

The embedding is extracted using:

```python
faces[0].embedding
```

These embeddings represent important facial characteristics.

---

## 3. Capture Webcam Input

OpenCV accesses the computer's webcam:

```python
cap = cv2.VideoCapture(0)
```

The system continuously reads frames from the webcam.

---

## 4. Detect Faces

InsightFace detects faces in the current frame:

```python
faces = app.get(frame)
```

Each detected face contains information such as:

* Bounding box
* Face embedding
* Detection information

---

## 5. Compare Face Embeddings

The live face embedding is compared with the stored embeddings.

The project uses **Cosine Similarity**:

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))
```

A higher similarity value indicates that the two face embeddings are more similar.

---

## 6. Identify the Person

The system finds the highest similarity score:

```python
best_idx = np.argmax(similarities)
```

Then it checks the threshold:

```python
THRESHOLD = 0.5
```

If:

```text
Similarity >= 0.5
```

the person is considered recognized.

Otherwise:

```text
Unknown
```

---

# ⚡ Performance Optimization

The project uses **frame skipping** to reduce the computational workload.

```python
FRAME_SKIP = 5
```

Instead of running face detection on every webcam frame, the system performs detection periodically and reuses the previously detected faces between detection frames.

This can improve real-time performance, especially when running InsightFace on a CPU.

---

# 📊 Recognition Output

The webcam displays a bounding box around detected faces.

Example:

```text
┌─────────────────────────────┐
│                             │
│       ┌───────────┐         │
│       │           │         │
│       │    Face   │         │
│       │           │         │
│       └───────────┘         │
│       Sumit (0.72)          │
│                             │
└─────────────────────────────┘
```

The displayed value represents the similarity score between the detected face and the closest known face.

---

# 🧮 Cosine Similarity

Cosine similarity measures the similarity between two vectors.

For two face embeddings **A** and **B**:

```text
Cosine Similarity =
(A · B) / (||A|| × ||B||)
```

In this project, the vectors are the face embeddings generated by ArcFace.

A higher value generally indicates greater similarity between the facial representations.

---

# 🎯 Threshold

The project uses:

```python
THRESHOLD = 0.5
```

The threshold determines whether the best matching face should be considered a known person.

```text
Similarity >= Threshold
        │
        ├── Yes → Recognized
        │
        └── No  → Unknown
```

The optimal threshold can vary depending on the dataset, image quality, camera conditions, and recognition requirements.

---

# 🔐 Privacy Considerations

This project processes faces from the local webcam and locally stored images.

For real-world deployment:

* Obtain appropriate consent before collecting facial data.
* Protect stored face images and embeddings.
* Avoid storing unnecessary biometric information.
* Follow applicable privacy and data-protection requirements.
* Do not use the system for unauthorized surveillance.

---

# 🚀 Future Improvements

The project can be extended with:

* [ ] Multiple images per person for better recognition
* [ ] Face database using SQLite/PostgreSQL
* [ ] Web-based interface using Streamlit
* [ ] GPU acceleration
* [ ] Face registration module
* [ ] Attendance management
* [ ] Recognition history
* [ ] CSV/Excel attendance export
* [ ] Real-time attendance dashboard
* [ ] Unknown-face logging
* [ ] Face anti-spoofing/liveness detection
* [ ] Improved threshold calibration
* [ ] Support for multiple faces in the same frame

---

# 📌 Applications

This technology can be used as a foundation for:

* Smart attendance systems
* Access-control prototypes
* Face-based authentication
* Personal identification systems
* Security research
* Computer vision projects
* AI/ML learning projects

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Computer Vision
* Face Detection
* Face Recognition
* Deep Learning
* Face Embeddings
* ArcFace
* Similarity Measurement
* Cosine Similarity
* Real-Time Video Processing
* OpenCV
* InsightFace
* Python

---

# 👨‍💻 Author

**Sumit Santosh Patil**

Bachelor of Engineering (B.E.) – Computer Engineering
Dhole Patil College of Engineering, Pune
Expected Graduation: 2027

---

# ⭐ Acknowledgements

This project uses the following open-source technologies:

* OpenCV
* InsightFace
* ArcFace
* NumPy

---

## 📄 License

This project is intended for educational and learning purposes.
