const API = "";

let lastOperation = "";
let lastInputName = "";
let processedReady = false;


// ----------------------
// Load operations
// ----------------------

let cpuOperations = [];
let gpuOperations = [];

fetch("/operations")
  .then(res => res.json())
  .then(data => {

    cpuOperations = data.cpu_operations;
    gpuOperations = data.gpu_operations;

    const modeSelect = document.getElementById("processorMode");

    const cpuOption = document.createElement("option");
    cpuOption.value = "cpu";
    cpuOption.text = "CPU";
    modeSelect.appendChild(cpuOption);

    if (data.gpu_available) {
        const gpuOption = document.createElement("option");
        gpuOption.value = "cuda";
        gpuOption.text = "GPU";
        modeSelect.appendChild(gpuOption);
    }

    updateOperations();

    modeSelect.addEventListener("change", updateOperations);
});

function updateOperations() {

    const opSelect = document.getElementById("operation");
    opSelect.innerHTML = "";

    const mode = document.getElementById("processorMode").value;

    const ops = mode === "cuda" ? gpuOperations : cpuOperations;

    ops.forEach(op => {

        const option = document.createElement("option");
        option.value = op;
        option.text = op;

        opSelect.appendChild(option);
    });

    toggleControls();
}


// ----------------------
// Slider labels
// ----------------------

// Gaussian blur slider
kernel.oninput = () => kernelVal.innerText = kernel.value;

// Canny sliders
low.oninput = () => lowVal.innerText = low.value;
high.oninput = () => highVal.innerText = high.value;

// Harris sliders
blockSize.oninput = () => blockSizeVal.innerText = blockSize.value;
kSize.oninput = () => kSizeVal.innerText = kSize.value;
kValue.oninput = () => kValueVal.innerText = kValue.value;
harrisThreshold.oninput = () => harrisThresholdVal.innerText = harrisThreshold.value;

// Median blur slider
medianKSize.oninput = () => medianKSizeVal.innerText = medianKSize.value;

// Bilateral sliders
biDiameter.oninput = () => biDiameterVal.innerText = biDiameter.value;
biSigmaColor.oninput = () => biSigmaColorVal.innerText = biSigmaColor.value;
biSigmaSpace.oninput = () => biSigmaSpaceVal.innerText = biSigmaSpace.value;

// Contrast Limited Adaptive Histogram Equalization (CLAHE) sliders
claheClipLimit.oninput = () => claheClipLimitVal.innerText = claheClipLimit.value;
claheTileSize.oninput = () => claheTileSizeVal.innerText = claheTileSize.value;

// Adaptive threshold sliders
adaptiveBlockSize.oninput = () => adaptiveBlockSizeVal.innerText = adaptiveBlockSize.value;
adaptiveC.oninput = () => adaptiveCVal.innerText = adaptiveC.value;

// SIFT sliders
siftFeatures.oninput = () => siftFeaturesVal.innerText = siftFeatures.value;
siftContrast.oninput = () => siftContrastVal.innerText = siftContrast.value;
siftEdge.oninput = () => siftEdgeVal.innerText = siftEdge.value;
siftSigma.oninput = () => siftSigmaVal.innerText = siftSigma.value;

// Watershed sliders
waterKSize.oninput = () => waterKSizeVal.innerText = waterKSize.value;
waterDistRatio.oninput = () => waterDistRatioVal.innerText = waterDistRatio.value;

// Erosion sliders
erosionKSize.oninput = () => erosionKSizeVal.innerText = erosionKSize.value;
erosionIterations.oninput = () => erosionIterationsVal.innerText = erosionIterations.value;

// Dilation sliders
dilationKSize.oninput = () => dilationKSizeVal.innerText = dilationKSize.value;
dilationIterations.oninput = () => dilationIterationsVal.innerText = dilationIterations.value;

// ----------------------
// Preview on select (NEW)
// ----------------------

function previewImage() {
    const file = imageInput.files[0];
    if (!file) return;

    original.src = URL.createObjectURL(file);
    lastInputName = file.name.split(".")[0];

    clearProcessed();   // reset processed on new image
}


// ----------------------
// Clear processed image (UTILITY)
// ----------------------

function clearProcessed() {
    processed.src = "";
    processedReady = false;
    downloadBtn.disabled = true;
    viewerBtn.disabled = true;   // ✅ disable viewer
}


// ----------------------
// Control visibility + reset processed on op change
// ----------------------

function toggleControls() {
    blurControls.style.display =
        operation.value === "gaussian_blur" ? "block" : "none";

    cannyControls.style.display =
        operation.value === "canny" ? "block" : "none";

    harrisControls.style.display = 
        operation.value === "harris_corner" ? "block" : "none";

    medianBlurControls.style.display = 
        operation.value === "median_blur" ? "block" : "none";

    bilateralControls.style.display =
        operation.value === "bilateral_filter" ? "block" : "none";

    histEqControls.style.display = 
            operation.value === "histogram_equalization" ? "block" : "none";    
    
    claheControls.style.display = 
        operation.value === "clahe" ? "block" : "none";

    otsuControls.style.display = 
        operation.value === "otsu_threshold" ? "block" : "none";

    adaptiveControls.style.display = 
        operation.value === "adaptive_threshold" ? "block" : "none";

    siftControls.style.display = 
        operation.value === "sift" ? "block" : "none";

    watershedControls.style.display = 
        operation.value === "watershed" ? "block" : "none";

    erosionControls.style.display = 
        operation.value === "erosion" ? "block" : "none";

    dilationControls.style.display =
        operation.value === "dilation" ? "block" : "none";
        
    clearProcessed();   // ✅ reset processed when operation changes
}


// ----------------------
// Processing call
// ----------------------

function processImage() {

    const file = imageInput.files[0];
    if (!file) {
        alert("Select an image first");
        return;
    }

    lastOperation = operation.value;

    const config = {
        operation: lastOperation,
        mode: document.getElementById("processorMode").value,
        parameters: {}
    };

    if (lastOperation === "gaussian_blur") {
        config.parameters.kernel = parseInt(kernel.value);
    }

    if (lastOperation === "canny") {
        config.parameters.low = parseInt(low.value);
        config.parameters.high = parseInt(high.value);
    }

    if (lastOperation === "harris_corner") {
        config.parameters.block_size = parseInt(blockSize.value);
        config.parameters.k_size = parseInt(kSize.value);
        config.parameters.k = parseFloat(kValue.value); // Use parseFloat for decimals!
        config.parameters.threshold = parseInt(harrisThreshold.value);
    }

    if (lastOperation === "median_blur") {
        config.parameters.ksize = parseInt(medianKSize.value);
    }

    if (lastOperation === "bilateral_filter") {
        config.parameters.d = parseInt(biDiameter.value);
        config.parameters.sigma_color = parseFloat(biSigmaColor.value);
        config.parameters.sigma_space = parseFloat(biSigmaSpace.value);
    }

    if (lastOperation === "clahe") {
        config.parameters.clip_limit = parseFloat(claheClipLimit.value);
        config.parameters.tile_grid_size = parseInt(claheTileSize.value);
    }

    if (lastOperation === "adaptive_threshold") {
        config.parameters.block_size = parseInt(adaptiveBlockSize.value);
        config.parameters.c = parseFloat(adaptiveC.value); 
    }

    if (lastOperation === "sift") {
        config.parameters.nfeatures = parseInt(siftFeatures.value);
        config.parameters.contrast_threshold = parseFloat(siftContrast.value);
        config.parameters.edge_threshold = parseFloat(siftEdge.value);
        config.parameters.sigma = parseFloat(siftSigma.value);
    }

    if (lastOperation === "watershed") {
        config.parameters.kernel_size = parseInt(waterKSize.value);
        config.parameters.distance_ratio = parseFloat(waterDistRatio.value);
    }

    if (lastOperation === "erosion") {
        config.parameters.kernel_size = parseInt(erosionKSize.value);
        config.parameters.iterations = parseInt(erosionIterations.value);
    }

    if (lastOperation === "dilation") { 
        config.parameters.kernel_size = parseInt(dilationKSize.value);
        config.parameters.iterations = parseInt(dilationIterations.value);
    }

    const form = new FormData();
    form.append("file", file);
    form.append("config", JSON.stringify(config));

    downloadBtn.disabled = true;   // disable during processing

    fetch(API + "/process", {
        method: "POST",
        body: form
    })
    .then(r => r.blob())
    .then(blob => {
    processed.src = URL.createObjectURL(blob);
    processedReady = true;

    downloadBtn.disabled = false;
    viewerBtn.disabled = false;   // ✅ enable viewer   
    });
}


// ----------------------
// Reset
// ----------------------

function resetImages() {
    original.src = "";
    imageInput.value = "";
    clearProcessed();
}


// ----------------------
// Download with naming
// ----------------------

function downloadImage() {
    if (!processedReady) return;

    const name = `${lastInputName}_${lastOperation}.png`;

    const a = document.createElement("a");
    a.href = processed.src;
    a.download = name;
    a.click();
}

// ==========================
// CLEAN PAN + ZOOM VIEWER
// ==========================

let scale = 1;
let translateX = 0;
let translateY = 0;
let dragging = false;
let startX = 0;
let startY = 0;

const zoomLevel = document.getElementById("zoomLevel");
const viewerModal = document.getElementById("viewerModal");
const viewerImage = document.getElementById("viewerImage");
const viewerCanvasWrap = document.getElementById("viewerCanvasWrap");

function openViewer() {
    if (!processed.src) return;

    viewerImage.src = processed.src;
    viewerModal.style.display = "flex";

    scale = 1;
    translateX = 0;
    translateY = 0;

    updateTransform();
}

function closeViewer() {
    viewerModal.style.display = "none";
}

function updateTransform() {
    viewerImage.style.transform =
        `translate(${translateX}px, ${translateY}px) scale(${scale})`;

    if (zoomLevel) {
        zoomLevel.innerText = `Zoom: ${Math.round(scale * 100)}%`;
    }
}

function resetZoom() {
    scale = 1;
    translateX = 0;
    translateY = 0;

    updateTransform();
}

// --------------------
// Zoom (mouse wheel)
// --------------------

viewerCanvasWrap.addEventListener("wheel", (e) => {
    e.preventDefault();

    const zoomAmount = 0.1;

    if (e.deltaY < 0) {
        scale += zoomAmount;
    } else {
        scale -= zoomAmount;
    }

    scale = Math.max(0.2, Math.min(scale, 6));

    updateTransform();
});

// --------------------
// Drag to pan
// --------------------

viewerCanvasWrap.addEventListener("mousedown", (e) => {
    dragging = true;
    startX = e.clientX - translateX;
    startY = e.clientY - translateY;
    viewerCanvasWrap.style.cursor = "grabbing";
});

window.addEventListener("mouseup", () => {
    dragging = false;
    viewerCanvasWrap.style.cursor = "grab";
});

window.addEventListener("mousemove", (e) => {
    if (!dragging) return;

    translateX = e.clientX - startX;
    translateY = e.clientY - startY;

    updateTransform();
});


// --------------------
// Fullscreen
// --------------------

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        viewerModal.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}