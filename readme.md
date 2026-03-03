# Image Processing Web App (C++ + CUDA + REST API)

A web-deployable image processing system built with:

- C++
- OpenCV
- CUDA (optional GPU acceleration)
- Crow (C++ REST framework)
- HTML + JavaScript frontend

This project demonstrates a clean architecture where the processing engine is written in C++ and can run either on CPU or GPU (CUDA), while being accessible through a web interface.

---

## 🚀 Features

- Upload image from browser
- Apply processing operations:
  - Grayscale
  - Gaussian Blur
  - Edge Detection
- Choose execution device:
  - CPU
  - GPU (CUDA)
- Automatic fallback to CPU if GPU is unavailable
- Clean backend abstraction (CPUProcessor / CUDAProcessor)

---

## 🏗 Architecture
```
Frontend (HTML + JS)
↓
REST API (Crow)
↓
ImageProcessor Interface
↓
┌───────────────┐
│ CPUProcessor  │
│ CUDAProcessor │
└───────────────┘
```


The frontend communicates with the backend via HTTP POST.
The backend selects CPU or GPU processing dynamically.

---

## 📁 Project Structure
```
Image Processing Tool
├── backend
│   ├── CMakeLists.txt
│   ├── build
│   │   ├── CMakeCache.txt
│   │   ├── CMakeFiles
│   │   ├── Makefile
│   │   ├── cmake_install.cmake
│   │   └── server
│   ├── include
│   │   ├── CPUProcessor.h
│   │   ├── CUDAProcessor.h
│   │   └── ImageProcessor.h
│   ├── src
│   │   ├── CPUProcessor.cpp
│   │   ├── CUDAProcessor.cu
│   │   └── main.cpp
│   └── static
│       ├── app.js
│       ├── index.html
│       └── style.css
└── readme.md
---

## 🧠 How It Works

1. User uploads an image in browser.
2. Image is sent to backend via POST request.
3. Backend decodes image using OpenCV.
4. Based on `device` parameter:
   - CPU → uses CPUProcessor
   - GPU → uses CUDAProcessor (if available)
5. Processed image is returned as PNG.
6. Browser displays result.

---

## ⚙️ Requirements

### System Requirements

- C++17 compatible compiler
- CMake (>= 3.10)
- OpenCV (C++ version)
- CUDA Toolkit (if using GPU)
- NVIDIA GPU (for CUDA execution)

### Libraries

- OpenCV
- CUDA Toolkit
- Crow (header-only)

---

## 🔧 Build Instructions

### 1️⃣ Install Dependencies

- Install OpenCV (C++ version)
- Install CUDA Toolkit
- Download Crow header

---

### 2️⃣ Build Project

```bash
mkdir build
cd build
cmake ..
make
```
### 3️⃣ Run Server
```
./server

Server runs on:

http://localhost:18080
```

### main.cpp
```
Uses Crow C++ web framework.

Responsibilities: - Start HTTP server - Define API routes - Serve static files - Receive image uploads - Select CPU or CUDA processor - Return processed image

Acts as the traffic controller between frontend and processing modules.
```

## Extending the flow
```
When you add a new filter, it must exist in:

🔹 Backend processing logic

🔹 API operation list

🔹 Frontend dropdown

🔹 Processor interface

Because your flow is:

Frontend Dropdown
      ↓
/operations API
      ↓
processImage() → sends operation
      ↓
main.cpp → selects function
      ↓
CPUProcessor / CUDAProcessor
      ↓
Image returned

So we must extend this pipeline properly.
```

### To Clone
```
git clone --recurse-submodules <your_repo>
```