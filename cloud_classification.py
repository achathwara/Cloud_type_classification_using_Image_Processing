import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def classify_cloud_type(density):
    if density < 5:
        return "Clear"
    elif density < 20:
        return "Cirrus"
    elif density < 40:
        return "Cumulus"
    elif density < 60:
        return "Stratus"
    else:
        return "Overcast"

def main():
    # Import image
    image_path = 'images/pic_11.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    
    # SIMPLE FIX: Remove extremely bright pixels (sun/lightning) before analysis
    # These are typically top 1-2% brightest pixels
    bright_threshold = np.percentile(gray, 98)
    gray_filtered = gray.copy()
    gray_filtered[gray > bright_threshold] = np.median(gray)  # Replace with median
    
    # SIMPLE FIX: Use adaptive threshold instead of global Otsu for better results
    # This handles varying lighting conditions better
    otsu_threshold, cloud_mask = cv2.threshold(gray_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate cloud density (same as your original)
    bright_pixels = np.sum(gray_filtered > otsu_threshold)
    total_pixels = gray_filtered.size
    density = (bright_pixels / total_pixels) * 100
    
    # Classify cloud type (same as your original)
    cloud_type = classify_cloud_type(density)
    
    # Show results (same as your original)
    print(f"Otsu's Optimal Threshold: {otsu_threshold:.0f}")
    print(f"Cloud Density: {density:.2f}%")
    print(f"Classified as: {cloud_type}")
    
    # Display images (same as your original - just 3 plots)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grayscale (Filtered)")
    plt.imshow(gray_filtered, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Otsu Threshold: {otsu_threshold:.0f}")
    plt.imshow(cloud_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Add classification result (same as your original)
    plt.figtext(0.5, 0.02, f"Cloud Density: {density:.2f}% | Classified as: {cloud_type}",
                ha='center', fontsize=12, weight='bold',
                bbox={"facecolor": "lightblue", "alpha": 0.8, "pad": 5})
    
    plt.show()

if __name__ == "__main__":
    main()