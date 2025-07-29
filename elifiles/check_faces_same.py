import cv2
from mtcnn import MTCNN
from PIL import Image
import sys
sys.path.append('./hashes')
from dinohash import DINOHash

def compare_face_images(img1_path, img2_path):
    # Initialize MTCNN and DINOHash
    detector = MTCNN()
    dinohash = DINOHash(pca_dims=96, model="vits14_reg", prod_mode=True)
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces1 = detector.detect_faces(img1_rgb)
    faces2 = detector.detect_faces(img2_rgb)
    
    if not faces1 or not faces2:
        return False, "No face detected in one or both images"
    
    # Extract face regions (use the first/largest face)
    box1 = faces1[0]['box']
    box2 = faces2[0]['box']
    
    face1 = img1_rgb[box1[1]:box1[1]+box1[3], box1[0]:box1[0]+box1[2]]
    face2 = img2_rgb[box2[1]:box2[1]+box2[3], box2[0]:box2[0]+box2[2]]
    
    # Convert to PIL Images
    face1_pil = Image.fromarray(face1)
    face2_pil = Image.fromarray(face2)
    
    # Generate hashes
    hash1 = dinohash.hash([face1_pil])[0]
    hash2 = dinohash.hash([face2_pil])[0]
    
    # Compare hashes (you can adjust the threshold)
    hamming_distance = bin(int(hash1.hex, 16) ^ int(hash2.hex, 16)).count('1')
    similarity = 1 - (hamming_distance / len(hash1.hex) / 4)  # rough similarity
    
    return similarity > 0.85, f"Similarity: {similarity:.3f}"

# Usage
result, message = compare_face_images("./images/ronnychieng/download (8).jpeg", "./images/ronnychieng/download (10).jpeg")
print(f"Same image: {result} - {message}")