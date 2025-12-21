# 🩺 Scan Quality Check

A convolutional neural network that evaluates ultrasound image quality at capture time.  
The model flags scans as **clear**, **unclear**, or **uncertain** to guide whether a retake is needed before interpretation.

This system focuses on **scan quality**, not diagnosis.

## 🌐 Live Demo  
https://huggingface.co/spaces/samhitak10/scan-quality-demo

## 🖼 Demo Preview
<img width="1894" height="984" alt="image" src="https://github.com/user-attachments/assets/a618a7bf-fb24-4e3b-b307-d79138d410c8" />

## 🖼 Demo Overview

Upload an ultrasound image to receive:
- A scan quality assessment  
- A confidence level  
- A recommended next action  

Example outputs:
- “Clear scan. Proceed with interpretation.”  
- “Unclear scan. Retake and adjust probe angle or brightness.”

## ⚡ Features  
- Web-based ultrasound image upload  
- Automatic grayscale preprocessing and normalization  
- Convolutional Neural Network for scan quality assessment  
- Conservative decision thresholds to reduce overconfident predictions  
- Action-oriented outputs designed for real-time feedback  
- Public demo deployed on Hugging Face Spaces  

## 🧠 Model Details  
- TensorFlow / Keras Convolutional Neural Network (CNN)  
- Input: grayscale ultrasound images (128 × 128)  
- Output: probability that a scan is unclear  
- Threshold-based mapping to:
  - Clear  
  - Uncertain  
  - Unclear  
- Designed to provide immediate guidance at the point of image capture  

## 🛠 Tech Stack  
- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow  
- Gradio  

## 📝 What I Learned  
- Building CNNs for medical image quality assessment  
- Preprocessing and standardizing ultrasound images  
- Translating model outputs into user-facing recommendations  
- Deploying interactive ML applications on Hugging Face Spaces  
