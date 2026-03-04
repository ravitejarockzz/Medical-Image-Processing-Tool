# **Image Processing Web App (C++ + OpenCV + Crow + CUDA)**

A modular, extensible medical image processing backend built with:

- **C++17**
- **OpenCV**
- **Crow (C++ Web Framework)**
- **Optional CUDA backend**
- **Dynamic filter registration (Factory Pattern)**
- **Frontend served directly from backend**

---

# **✨ Features**

- Gaussian Blur (CPU / CUDA)
- Canny Edge Detection (CPU / CUDA)
- Grayscale Conversion (CPU / CUDA)
- Dynamic filter auto-registration
- Runtime CPU/GPU switching
- Clean frontend UI with sliders
- No hardcoded switch-case logic
- Extensible plugin-style architecture

---

# **🏗 Architecture Overview**

This project follows a clean, extensible architecture.

---

## **1️⃣ IFilter Interface**

All filters inherit from:

```cpp
class IFilter {
public:
    virtual cv::Mat apply(
        const cv::Mat& input,
        const std::unordered_map<std::string, double>& params
    ) = 0;

    virtual ~IFilter() = default;
};
```

---

## **2️⃣ Self-Registering Filters**

Filters register themselves using macros:

```cpp
REGISTER_CPU_FILTER(CannyCPU, "canny");
REGISTER_CUDA_FILTER(CannyCUDA, "canny");
```

This uses **static auto-registration**, meaning:

- Filters register before `main()` runs
- No central modification required
- Open/Closed Principle is respected

---

## **3️⃣ Factory Pattern**

`FilterFactory` maintains maps:

```cpp
std::unordered_map<std::string, FilterCreator> cpuFilters;
std::unordered_map<std::string, FilterCreator> cudaFilters;
```

The processor simply asks:

```cpp
factory.createCPU("canny");
```

No switch-case logic anywhere.

---

## **4️⃣ Processor Abstraction**

- `CPUProcessor`
- `CUDAProcessor`

Both implement:

```cpp
process(input, operationName, parameters);
```

The frontend selects `"cpu"` or `"cuda"` at runtime.

---

# **📁 Project Structure**

```
backend/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── include/
│   ├── core/
│   │   └── IFilter.h
│   ├── factory/
│   │   └── FilterFactory.h
│   ├── processor/
│   │   ├── ImageProcessor.h
│   │   ├── CPUProcessor.h
│   │   └── CUDAProcessor.h
│   ├── cpu/
│   └── cuda/
│
├── src/
│   ├── cpu/
│   ├── cuda/
│   ├── processor/
│   ├── factory/
│   └── main.cpp
│
└── CMakeLists.txt
```

---

# **🚀 How To Clone (With Crow Submodule)**

This project uses **Crow** as a git submodule.

### Clone with submodules:

```bash
git clone --recurse-submodules <your-repo-url>
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

# **🔧 Requirements (Linux / WSL)**
- C++17 compatible compiler
- CMake (>= 3.10)
- OpenCV (C++ version)
- CUDA Toolkit (if using GPU)
- NVIDIA GPU (for CUDA execution)

Install dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libopencv-dev
```

If using CUDA:

- Install CUDA Toolkit
- Ensure `nvcc` is available

---

# **🛠 Build Instructions**

From the `backend` directory:

```bash
mkdir build
cd build
cmake ..
make -j $(nproc)
```

Run server:

```bash
./server
```

Open in browser:

```
http://localhost:18080
```

---

# **⚙️ CUDA Support**

CMake option:

```cmake
option(USE_CUDA "Enable CUDA backend" ON)
```

If CUDA toolkit is not found:

- GPU mode is disabled automatically
- CPU mode continues to work

Graceful fallback is built-in.

---

# **➕ How To Add a New Filter**

Example: Adding `grayscale`

### 1️⃣ Create CPU file

`src/cpu/GrayscaleCPU.cpp`

```cpp
class GrayscaleCPU : public IFilter {
public:
    cv::Mat apply(const cv::Mat& input,
                  const std::unordered_map<std::string,double>&) override {
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        cv::Mat output;
        cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
        return output;
    }
};

REGISTER_CPU_FILTER(GrayscaleCPU, "grayscale");
```

### 2️⃣ Create CUDA file

`src/cuda/GrayscaleCUDA.cu`

```cpp
REGISTER_CUDA_FILTER(GrayscaleCUDA, "grayscale");
```

### 3️⃣ Rebuild

```bash
cmake ..
make
```

No changes required in:

- `main.cpp`
- Processor classes
- Factory logic

---

# **🌐 API Endpoints**

## **GET /operations**

Returns:

```json
{
  "available_operations": ["gaussian_blur", "canny", "grayscale"],
  "gpu_available": true
}
```

---

## **POST /process**

Multipart form:

- `file` → image
- `config` → JSON string

Returns processed PNG image.

---

# **🎯 Design Principles Used**

- Factory Pattern
- Static Auto-Registration
- Open/Closed Principle
- Separation of Concerns
- Runtime Polymorphism
- CPU/GPU Abstraction

---

# **📌 Notes**

- Grayscale converts back to BGR for frontend compatibility.
- CUDA filters currently fallback to CPU logic.
- `GLOB_RECURSE` in CMake auto-detects new source files.

---

# **🔮 Future Improvements**

- Real CUDA kernels
- CPU vs GPU benchmarking
- Plugin-based dynamic loading
- Histogram equalization
- DICOM support
- Docker deployment
- Authentication layer

---

