const API = "";

let lastOperation = "";
let lastInputName = "";
let processedReady = false;


// ----------------------
// Load operations
// ----------------------

fetch(API + "/operations")
.then(r => r.json())
.then(data => {
    data.available_operations.forEach(op => {
        const o = document.createElement("option");
        o.value = op;
        o.text = op.replace("_"," ");
        operation.appendChild(o);
    });
    toggleControls();
});


// ----------------------
// Slider labels
// ----------------------

kernel.oninput = () => kernelVal.innerText = kernel.value;
low.oninput = () => lowVal.innerText = low.value;
high.oninput = () => highVal.innerText = high.value;


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
        parameters: {}
    };

    if (lastOperation === "gaussian_blur") {
        config.parameters.kernel = parseInt(kernel.value);
    }

    if (lastOperation === "canny") {
        config.parameters.low = parseInt(low.value);
        config.parameters.high = parseInt(high.value);
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