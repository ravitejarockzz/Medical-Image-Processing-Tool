import cv2

def grayscale(image, params):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, params):
    k = params.get("kernel", 5)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)


def canny(image, params):
    low = params.get("low", 50)
    high = params.get("high", 150)
    return cv2.Canny(image, low, high)


# Plugin registry
FILTERS = {
    "grayscale": grayscale,
    "gaussian_blur": gaussian_blur,
    "canny": canny,
}
