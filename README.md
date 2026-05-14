# 🧑‍💻 Python Head Tracking Module

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)  
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Live%20ML-orange)](https://developers.google.com/mediapipe)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

A real-time **head tracking system** powered by **OpenCV** and **MediaPipe**.  
This module detects the user’s head position via webcam and overlays a red laser-like dot on their face.  

Perfect for experiments, interactive installations, fun projects, or as a base for **AR/VR applications**.  

---



---

## 🚀 Features
- ✅ Real-time **head tracking** using webcam  
- ✅ **MediaPipe FaceMesh** for robust landmark detection  
- ✅ Red **laser dot overlay** at the head center  
- ✅ Adjustable **smoothing factor** to reduce jitter  
- ✅ Works fully **offline** — no internet required  
- ✅ Runs cross-platform (Windows, macOS, Linux)  

---

## 🛠️ Requirements
- Python **3.8+**  
- [OpenCV](https://opencv.org/) (`opencv-python`)  
- [MediaPipe](https://developers.google.com/mediapipe)  
- [NumPy](https://numpy.org/)  

### Install dependencies:
```bash
git clone https://github.com/RXDarkee/python-head-tracking-module.git

cd python-head-tracking-module

pip install -r requirements.txt
