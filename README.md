# 🚀 Real-Time Face Recognition Security System

A real-time face detection and recognition web application built using Flask, OpenCV, and the face_recognition library. This system captures faces via webcam, identifies registered users, and logs access activity.

---

## 🎯 Features

✅ Real-time face detection using webcam  
✅ Face recognition with pre-trained encodings  
✅ Add new face (multi-image capture for better accuracy)  
✅ Green box → Recognized face  
✅ Red box → Unknown face  
✅ Access status display (GRANTED / DENIED)  
✅ Automatic logging system  
✅ View logs via web UI  
✅ Simple and clean frontend  

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Computer Vision:** OpenCV
- **Face Recognition:** face_recognition (dlib-based)
- **Frontend:** HTML, CSS
- **Database:** File-based (dataset + encodings.pkl)

---

## 📂 Project Structure
face_app copy/
├── templates/
├── dataset/
├── app.py
├── requirements.txt
└── encodings.pkl
