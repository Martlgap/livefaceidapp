# 🎭 Live Face Recognition App

A **real-time face recognition application** built with Gradio that allows you to recognize faces in live webcam video streams directly in your browser. Upload reference images to build a face gallery, then watch as the system identifies matching faces in real-time with confidence scores and visual annotations.

> **✨ Key Features:** Real-time processing • Web-based interface • Multi-face detection • Adjustable similarity thresholds • Live confidence scoring

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange.svg)](https://gradio.app/)
[![uv](https://img.shields.io/badge/uv-dependency%20management-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/martlgap/livefaceidapp)](https://github.com/martlgap/livefaceidapp)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📑 Table of Contents

- [👀 Example](#-example)
- [🚀 Quick Start](#-quick-start)
  - [💻 System Requirements](#-system-requirements)
  - [Installation & Usage](#installation--usage)
  - [🌐 Deployment Options](#-deployment-options)
- [📖 How to Use](#-how-to-use)
- [🎛️ Features](#️-features)
- [🧠 Machine Learning Models](#-machine-learning-models)
- [🔧 Troubleshooting](#-troubleshooting)
- [🐛 Development Status & Roadmap](#-development-status--roadmap)
- [📖 About This Project](#-about-this-project)
- [📚 Resources & References](#-resources--references)

# 👀 Example

As an example, instead of using a webcam I fed a snippet of BigBangTheory into the app and it recognized the characters in real-time:

![Demo](example.gif)

# 🚀 Quick Start

## 💻 System Requirements

### Minimum Requirements
- **Python 3.12** (exactly - not 3.11 or 3.13)
- **4GB RAM** (8GB recommended)
- **Webcam** (built-in or USB)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### Recommended Setup
- **8GB+ RAM** for optimal performance
- **Multi-core CPU** (Intel i5/AMD Ryzen 5 or better)
- **Good lighting conditions** for best recognition accuracy
- **[uv](https://github.com/astral-sh/uv)** for fastest dependency management

### Supported Platforms
- ✅ **macOS** (Intel & Apple Silicon)
- ✅ **Linux** (Ubuntu 20.04+, CentOS 8+)
- ✅ **Windows 10/11** (with proper Python setup)

### Browser Compatibility
- ✅ Chrome/Chromium 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Installation & Usage

### 🟢 Method 1: Using uv (Recommended)

The fastest and most reliable way to get started:

```bash
# Clone the repository
git clone https://github.com/Martlgap/livefaceidapp.git
cd livefaceidapp

# Install dependencies and run
uv sync
uv run python main.py
```

Or use the convenience script:

```bash
chmod +x run.sh
./run.sh
```

### ⚪ Method 2: Using pip (Alternative)

If you prefer traditional Python package management:

```bash
# Clone and setup virtual environment
git clone https://github.com/Martlgap/livefaceidapp.git
cd livefaceidapp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install gradio scikit-image scikit-learn opencv-python-headless onnxruntime mediapipe numpy pillow watchdog memory-profiler

# Run the application
python main.py
```

> **💡 Pro Tip:** The uv approach is significantly faster for dependency resolution and installation, especially on fresh setups.

## 🌐 Deployment Options

### Local Development
The app runs on `http://localhost:7860` by default with `share=True` enabled, automatically generating a public shareable link for easy testing and demonstration.

### Server Deployment with HTTPS

For production deployment on a server with HTTPS support:

```bash
# Download and setup ssl-proxy
wget https://github.com/suyashkumar/ssl-proxy/releases/download/v0.2.7/ssl-proxy-linux-amd64.tar.gz
tar -xzf ssl-proxy-linux-amd64.tar.gz

# Run with HTTPS proxy
./ssl-proxy-linux-amd64 -from 0.0.0.0:8502 -to 0.0.0.0:7860
```

Then access your app securely at `https://your-server-ip:8502`

### 🔗 Alternative: Using Gradio's Built-in Sharing
The app automatically provides a public shareable link when launched, perfect for:
- Remote demonstrations
- Cross-device testing  
- Sharing with team members
- Mobile device access

## ✨ Why Gradio?

This implementation leverages Gradio's strengths for computer vision applications:

- **🎥 Native Webcam Support**: Seamless browser-based camera integration
- **⚡ Real-time Streaming**: Optimized for live video processing
- **🌍 Zero Configuration**: Works across platforms without WebRTC setup
- **📱 Mobile Friendly**: Responsive design for various devices
- **🔧 Simple Deployment**: No complex server configurations needed

# 📖 How to Use

## Step-by-Step Guide

### 1. 🖼️ Build Your Face Gallery
- Click **"Upload Reference Images"** in the left panel
- Select one or more photos containing faces you want to recognize
- The system automatically detects faces and adds them to your gallery
- Each image filename becomes the person's name label

### 2. 🎥 Start Live Recognition
- Allow browser access to your webcam when prompted
- Position faces in front of the camera
- Watch as the system detects and recognizes faces in real-time

### 3. 🎛️ Fine-tune Performance
- Adjust the **"Similarity Threshold"** slider:
  - **Lower values** (0.3-0.7): Stricter matching, fewer false positives
  - **Higher values** (0.8-1.2): More lenient matching, catches more matches
- Monitor **live match scores** and **confidence percentages** in the right panel

### 4. 📊 Understand the Results
- **Green boxes**: Successfully matched faces
- **Blue boxes**: Detected but unmatched faces
- **Distance scores**: Lower = better match (typically < 1.0 for good matches)
- **Confidence %**: Higher = more certain match

## ⚙️ Technical Architecture

The application processes video streams through three optimized stages:

1. **🔍 Face Detection**: MediaPipe's FaceMesh identifies up to 7 faces per frame with precise landmark detection
2. **🧠 Face Recognition**: MobileNetV2-based neural network extracts 512-dimensional face embeddings via ONNX Runtime
3. **🎯 Face Matching**: Cosine distance similarity matching against your gallery with configurable thresholds

The entire pipeline is optimized for real-time performance, processing frames at up to 10 FPS depending on hardware capabilities.

A detailed description of the implementation can be found here:

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://martlgap.medium.com/529fc686b475)

# 🧠 Machine Learning Models

The app uses the following machine learning models:

- **Face Detection**: FaceMesh ([MediaPipe](https://google.github.io/mediapipe/solutions/face_mesh.html))
- **Face Recognition**: [MobileNetV2](https://arxiv.org/abs/1801.04381) Architecture, trained with [MS1M](https://arxiv.org/abs/1607.08221) dataset and [ArcFace](https://arxiv.org/abs/1801.07698) Loss using [Tensorflow](https://tensorflow.org), converted to [ONNX](https://onnxruntime.ai)

# 🎛️ Features

## Core Functionality
- **🎥 Real-time Face Recognition**: Process live webcam feeds with minimal latency
- **📁 Dynamic Gallery Management**: Upload multiple reference images to build your face database
- **🎯 Adjustable Similarity Threshold**: Fine-tune recognition sensitivity from strict to lenient
- **👥 Multi-face Detection**: Simultaneously detect and recognize up to 7 faces per frame
- **📊 Live Confidence Scoring**: Real-time distance metrics and confidence percentages
- **🎨 Visual Annotations**: Dynamic bounding boxes, name labels, and match indicators

## Technical Features
- **🌐 Browser-based Interface**: Zero installation for end users - runs in any modern browser
- **📱 Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **⚡ High Performance**: ONNX Runtime optimization for fast inference
- **🔄 Streaming Architecture**: Efficient frame processing with Gradio's streaming capabilities
- **🎛️ Interactive Controls**: Real-time threshold adjustment and gallery updates

## User Experience
- **🚀 One-click Setup**: Simple script execution to get started
- **📤 Drag & Drop Upload**: Intuitive image gallery management
- **📈 Live Feedback**: Instant visual and numerical match results
- **🔗 Share Ready**: Built-in public link generation for demonstrations

# 📖 About This Project

## 🎓 Background
This application was developed during research at the **[Chair of Human-Machine Communication](https://www.ce.cit.tum.de/en/mmk/home/)** at TUM (Technical University of Munich). The primary goal was to evaluate and compare live face recognition systems across different platforms and frameworks.

## 🔬 Research Context
The project explores the practical implementation of real-time face recognition in web-based environments, focusing on:
- **Performance benchmarking** of different ML frameworks
- **User experience optimization** for browser-based computer vision
- **Accessibility** of advanced AI technologies through simple interfaces
- **Privacy considerations** in live video processing applications

## 🏗️ Architecture Evolution
This implementation represents a significant evolution from earlier versions:
- **V1**: Original WebRTC-based streaming (complex setup, platform limitations)
- **V2**: Streamlit implementation (good but limited real-time capabilities)  
- **V3**: Current Gradio-based solution (optimal balance of performance and usability)

## 🤝 Contributing
This project serves as both a research tool and an educational resource. Contributions, suggestions, and improvements are welcome! Whether you're interested in:
- Performance optimizations
- Feature enhancements  
- Platform compatibility
- Documentation improvements

Feel free to open issues or submit pull requests.

## 📄 License & Usage
Released under MIT License - feel free to use, modify, and distribute for both academic and commercial purposes. If you use this work in research, please consider citing the associated publications.

# � Troubleshooting

## Common Issues & Solutions

### 🎥 Webcam Problems
- **Camera not detected**: Ensure browser permissions are granted and no other apps are using the webcam
- **Poor video quality**: Check lighting conditions and camera resolution settings
- **Laggy performance**: Close other browser tabs/applications using camera resources

### 🔍 Recognition Issues
- **No faces detected**: Ensure adequate lighting and face visibility
- **False matches**: Decrease similarity threshold (try 0.5-0.7)
- **Missing matches**: Increase similarity threshold (try 0.9-1.2)
- **Low confidence scores**: Add more/better quality reference images

### 💻 Performance Issues
- **Slow processing**: Check system resources, close unnecessary applications
- **High CPU usage**: Normal behavior - face recognition is computationally intensive
- **Memory warnings**: Restart the application if processing many large images

### 🚀 Installation Problems
- **Python version**: Ensure Python 3.12 is installed (not 3.11 or 3.13)
- **uv not found**: Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Dependencies fail**: Try the pip installation method as fallback

## 📊 Performance Tips
- **Optimal conditions**: Good lighting, clear face visibility, minimal background motion
- **Image quality**: Use high-resolution reference images with clear, front-facing photos
- **Hardware**: Better performance on systems with dedicated GPUs (future CUDA support planned)

# 📚 Resources & References

## Core Technologies
- **[Gradio](https://gradio.app/)** - Modern ML app framework powering the interface
- **[MediaPipe](https://mediapipe-studio.webapps.google.com/home)** - Production-ready face detection
- **[ONNX Runtime](https://onnxruntime.ai/)** - Cross-platform ML inference optimization
- **[OpenCV](https://opencv.org/)** - Computer vision processing library
- **[uv](https://github.com/astral-sh/uv)** - Fast Python package management

## Documentation & Guides
- **[Gradio Documentation](https://gradio.app/docs/)** - Complete framework reference
- **[MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)** - Face detection details
- **[ONNX Model Zoo](https://onnxruntime.ai/docs/get-started/)** - ML model optimization guides
- **[ssl-proxy](https://github.com/suyashkumar/ssl-proxy)** - HTTPS deployment tool

## Research Papers
- **[MobileNetV2](https://arxiv.org/abs/1801.04381)** - Efficient neural architecture
- **[ArcFace Loss](https://arxiv.org/abs/1801.07698)** - Face recognition training methodology  
- **[MS1M Dataset](https://arxiv.org/abs/1607.08221)** - Large-scale face recognition dataset

# 🐛 Development Status & Roadmap

## ✅ Completed Features
- [X] ~~Severe lag/delay in video stream~~ → **Fixed by migrating to Gradio**
- [X] ~~WebRTC streaming issues~~ → **Resolved with native webcam support**
- [X] ~~Mobile device optimization~~ → **Responsive design implemented**
- [X] ~~Confidence scores display~~ → **Live distance metrics & confidence percentages**
- [X] ~~Multi-face recognition~~ → **Up to 7 simultaneous faces supported**
- [X] ~~Real-time threshold adjustment~~ → **Interactive slider control**

## 🚧 In Progress & Planned Features
- [ ] **Face Embedding Visualization**: t-SNE/UMAP plots of face embeddings
- [ ] **Advanced Gallery Management**: Delete, edit, and organize face entries
- [ ] **Export Functionality**: Save recognition results and matched faces
- [ ] **Performance Optimization**: Enhanced processing for lower-end devices
- [ ] **Batch Processing Mode**: Upload videos for offline face recognition
- [ ] **Recognition History**: Track and review past recognition sessions
- [ ] **Custom Model Support**: Load your own trained face recognition models

## 🔧 Technical Improvements
- [ ] **GPU Acceleration**: CUDA support for faster inference
- [ ] **Model Quantization**: Optimize model size for edge deployment
- [ ] **Database Integration**: Persistent face gallery storage
- [ ] **API Endpoints**: REST API for programmatic access
- [ ] **Docker Support**: Containerized deployment options

## 💡 Enhancement Ideas
- [ ] **Age/Gender Detection**: Additional face attribute analysis
- [ ] **Liveness Detection**: Prevent spoofing attempts
- [ ] **Face Clustering**: Automatic grouping of similar faces
- [ ] **Privacy Mode**: Local-only processing without sharing
