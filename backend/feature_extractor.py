import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog
from scipy.stats import entropy


def circular_mean_std(h):
    """
    Calculate circular mean and circular standard deviation for HSV Hue channel.
    """
    h = h.astype(np.float32)
    angles = h * 2 * np.pi / 180.0

    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))

    mean = np.arctan2(sin_mean, cos_mean) * 180.0 / np.pi
    
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    r_clamped = np.clip(r, 1e-10, 1.0)
    std = np.sqrt(-2.0 * np.log(r_clamped))

    return float(mean), float(std)


def extract_features_from_image(image):
    """
    Extract 651-dimensional feature vector from a BGR OpenCV image numpy array.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input provided.")

    # Resize to 128x128 as trained in ffd.ipynb
    image = cv2.resize(image, (128, 128))

    # 1. RGB Features
    R, G, B = cv2.split(image)
    R_mean, G_mean, B_mean = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))
    R_std, G_std, B_std = float(np.std(R)), float(np.std(G)), float(np.std(B))

    # 2. HSV Features
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    H_mean, H_std = circular_mean_std(H)
    S_mean, V_mean = float(np.mean(S)), float(np.mean(V))
    S_std, V_std = float(np.std(S)), float(np.std(V))

    # 3. LAB Features
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L_mean, L_std = float(np.mean(L)), float(np.std(L))
    a_mean, a_std = float(np.mean(a)), float(np.std(a))
    b_mean, b_std = float(np.mean(b)), float(np.std(b))

    # 4. Grayscale & Texture Features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    glcm = graycomatrix(
        gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    glcm_contrast = float(graycoprops(glcm, "contrast")[0, 0])
    glcm_energy = float(graycoprops(glcm, "energy")[0, 0])
    glcm_homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-10)
    grayscale_entropy = float(entropy(hist.flatten()))

    # 5. Shape Features
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))
        circularity = float((4 * np.pi * contour_area) / (perimeter**2 + 1e-5))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(contour_area / (hull_area + 1e-5))
        _, _, w, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w / (h_box + 1e-5))
        extent = float(contour_area / (w * h_box + 1e-5))
    else:
        contour_area = 0.0
        perimeter = 0.0
        circularity = 0.0
        solidity = 0.0
        aspect_ratio = 0.0
        extent = 0.0

    # 6. Decay Feature
    dark_pixel_ratio = float(np.sum(gray < 40) / gray.size)

    # 7. HOG Features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )

    # Combine vector
    scalar_features = np.array(
        [
            R_mean,
            G_mean,
            B_mean,
            R_std,
            G_std,
            B_std,
            H_mean,
            S_mean,
            V_mean,
            H_std,
            S_std,
            V_std,
            L_mean,
            L_std,
            a_mean,
            a_std,
            b_mean,
            b_std,
            laplacian_variance,
            glcm_contrast,
            glcm_energy,
            glcm_homogeneity,
            grayscale_entropy,
            contour_area,
            perimeter,
            circularity,
            solidity,
            aspect_ratio,
            extent,
            dark_pixel_ratio,
        ],
        dtype=np.float32,
    )

    full_vector = np.hstack([scalar_features, hog_features])

    return full_vector, {
        "color_stats": {
            "R_mean": round(R_mean, 2),
            "G_mean": round(G_mean, 2),
            "B_mean": round(B_mean, 2),
            "H_mean": round(H_mean, 2),
            "S_mean": round(S_mean, 2),
            "V_mean": round(V_mean, 2),
        },
        "texture_stats": {
            "laplacian_variance": round(laplacian_variance, 2),
            "glcm_contrast": round(glcm_contrast, 4),
            "glcm_energy": round(glcm_energy, 4),
            "glcm_homogeneity": round(glcm_homogeneity, 4),
            "grayscale_entropy": round(grayscale_entropy, 4),
        },
        "shape_stats": {
            "contour_area": round(contour_area, 2),
            "perimeter": round(perimeter, 2),
            "circularity": round(circularity, 4),
            "solidity": round(solidity, 4),
            "dark_pixel_ratio": round(dark_pixel_ratio, 4),
        },
    }


def extract_features_from_bytes(img_bytes):
    """
    Decodes image byte stream and returns feature vector.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from bytes.")
    return extract_features_from_image(image)


def extract_features_from_path(image_path):
    """
    Reads image file from path and returns feature vector.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    return extract_features_from_image(image)
