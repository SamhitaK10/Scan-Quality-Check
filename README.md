# 🩺 Scan Quality Check

A convolutional neural network that evaluates ultrasound image quality at capture time.  
The model flags scans as **clear**, **unclear**, or **uncertain** to guide whether a retake is needed before interpretation.

This system focuses on **scan quality**, not diagnosis.  
When a scan is not clearly acceptable, it also **explains why** using a **Retrieval-Augmented Generation (RAG)** pipeline grounded in clinical guidelines.

## 🌐 Live Demo  
https://huggingface.co/spaces/samhitak10/scan-quality-demo

## 🖼 Demo Preview
<img width="1894" height="984" alt="image" src="https://github.com/user-attachments/assets/a618a7bf-fb24-4e3b-b307-d79138d410c8" />

## 🖼 Demo Overview

Upload an ultrasound image to receive:
- A scan quality assessment  
- A confidence level  
- A recommended next action  
- **An explanation of why the scan may be low quality**

Example outputs:
- “Clear scan. Proceed with interpretation.”  
- “Unclear scan. Retake the scan.”  
- “Uncertain scan quality. Consider retaking for confirmation.”  
- “Likely issues: depth, gain, alignment, missing structures, artifacts.”

## ⚡ Features  
- Web-based ultrasound image upload  
- Automatic grayscale preprocessing and normalization  
- Convolutional Neural Network for scan quality assessment  
- Conservative decision thresholds to reduce overconfident predictions  
- Binary entropy used to estimate prediction uncertainty  
- **RAG-based explanation layer that tells you *why* a scan is low quality**  
- Explanations grounded in published ultrasound quality criteria  
- Public demo deployed on Hugging Face Spaces  

## 🧠 Model Details
- TensorFlow / Keras Convolutional Neural Network (CNN)
- Input: grayscale ultrasound images (128 × 128)
- Output: probability that a scan is unclear
- Binary entropy computed from predicted probabilities to estimate uncertainty
- Threshold-based mapping to:
  - Clear
  - Uncertain
  - Unclear
- Optimized for real-time scan quality feedback at image capture

### RAG Explanation Module
- Uses **Retrieval-Augmented Generation (RAG)** with a local vector store
- Clinical scan quality guidelines sourced from the **Journal of Hospital Medicine**
- Guidelines are chunked and embedded locally
- For unclear or uncertain scans:
  - Relevant guideline chunks are retrieved
  - Content is compressed into high-level quality factors:
    - Depth
    - Gain
    - Alignment
    - Coverage
    - Artifacts
- Returns short, actionable explanations instead of raw guideline text

## 📊 Performance Summary
- Best validation accuracy: **0.68**
- Best validation AUC: **~0.79**
- Performance stabilized after ~6–8 epochs
- Model optimized for conservative uncertainty handling rather than peak accuracy

## 🛠 Tech Stack  
- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow  
- Gradio  
- Transformers + PyTorch (local embeddings)  
- Local JSON-based vector store (chunk + embed + cosine similarity)

## 📝 What I Learned  
- Building CNNs for ultrasound image quality assessment  
- Using uncertainty-aware thresholds in medical ML  
- Implementing RAG for guideline-grounded explanations  
- Translating clinical standards into user-facing feedback  
- Deploying interactive ML applications on Hugging Face Spaces  

## 📄 License  
MIT License
