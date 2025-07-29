import subprocess
import os
import cv2
from mtcnn import MTCNN
import tempfile
import numpy as np
from PIL import Image
import math

def get_dinohash(image_path):
    """
    Get dinohash for an image using the command line version
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: The hash string, or None if failed
    """
    try:
        # Run the command line version
        result = subprocess.run([
            'python3', 'hashes/dinohash.py', image_path
        ], 
        capture_output=True, 
        text=True, 
        cwd='/home/ssm-user/dinohash-perceptual-hash'
        )
        
        if result.returncode == 0:
            # Return the hash (strip whitespace)
            return result.stdout.strip()
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Failed to run dinohash command: {e}")
        return None

def align_face_with_landmarks(image, keypoints):
    """
    Align face using eye landmarks to make eyes horizontal
    
    Args:
        image (numpy.ndarray): Input image
        keypoints (dict): MTCNN keypoints with 'left_eye' and 'right_eye'
        
    Returns:
        numpy.ndarray: Aligned face image
    """
    # Get eye coordinates
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    
    # Calculate the angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
    print(f"Face rotation angle: {angle:.2f} degrees")
    
    # Calculate center point between eyes
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2
    center = (center_x, center_y)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return aligned, angle

def normalize_face_crop(face_image, target_size=(160, 160)):
    """
    Normalize face crop to standard size and enhance contrast
    
    Args:
        face_image (numpy.ndarray): Face image
        target_size (tuple): Target size for normalization
        
    Returns:
        numpy.ndarray: Normalized face image
    """
    # Resize to standard size
    resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert to LAB color space for better illumination normalization
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized

def extract_and_align_face(image_path, output_path):
    """
    Extract face, align it using landmarks, and save it
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save the aligned face
        
    Returns:
        bool: True if face was found, aligned and saved, False otherwise
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Detect faces with landmarks
        faces = detector.detect_faces(img_rgb)
        
        if not faces:
            print(f"No face detected in {image_path}")
            return False
        
        # Get the face with highest confidence
        best_face = max(faces, key=lambda x: x['confidence'])
        confidence = best_face['confidence']
        print(f"Face detected with confidence: {confidence:.3f}")
        
        # Check if we have good landmarks
        if 'keypoints' not in best_face:
            print("No facial landmarks detected")
            return False
        
        keypoints = best_face['keypoints']
        
        # Verify we have the required landmarks
        if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
            print("Missing eye landmarks for alignment")
            return False
        
        # Extract face bounding box with generous padding
        x, y, w, h = best_face['box']
        
        # Calculate padding based on face size
        padding = max(w, h) // 4  # 25% padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_rgb.shape[1] - x, w + 2*padding)
        h = min(img_rgb.shape[0] - y, h + 2*padding)
        
        # Extract face region
        face_region = img_rgb[y:y+h, x:x+w]
        
        # Adjust keypoints relative to cropped face
        adjusted_keypoints = {}
        for key, point in keypoints.items():
            adjusted_keypoints[key] = (point[0] - x, point[1] - y)
        
        print(f"Original landmarks - Left eye: {keypoints['left_eye']}, Right eye: {keypoints['right_eye']}")
        print(f"Adjusted landmarks - Left eye: {adjusted_keypoints['left_eye']}, Right eye: {adjusted_keypoints['right_eye']}")
        
        # Align the face using eye landmarks
        aligned_face, rotation_angle = align_face_with_landmarks(face_region, adjusted_keypoints)
        
        # Normalize the face (resize, enhance contrast)
        normalized_face = normalize_face_crop(aligned_face, target_size=(200, 200))
        
        # Convert back to BGR for OpenCV saving
        face_bgr = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2BGR)
        
        # Save the aligned face
        cv2.imwrite(output_path, face_bgr)
        print(f"Aligned face extracted and saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error extracting and aligning face from {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_faces_with_threshold(img1_path, img2_path, threshold=0.80, save_faces=False):
    """
    Compare faces in two images using MTCNN alignment + dinohash with similarity threshold
    
    Args:
        img1_path (str): Path to first image
        img2_path (str): Path to second image
        threshold (float): Similarity threshold (0.0-1.0)
        save_faces (bool): Whether to save extracted faces for inspection
        
    Returns:
        tuple: (is_same_person, similarity_score)
    """
    print(f"Comparing faces with alignment:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print(f"  Similarity threshold: {threshold}")
    print("-" * 60)
    
    # Create temporary files for extracted faces
    with tempfile.NamedTemporaryFile(suffix='_aligned_face1.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='_aligned_face2.jpg', delete=False) as tmp2:
        
        face1_path = tmp1.name
        face2_path = tmp2.name
    
    try:
        # Extract and align faces
        print("Extracting and aligning face from image 1...")
        face1_extracted = extract_and_align_face(img1_path, face1_path)
        
        print("\nExtracting and aligning face from image 2...")
        face2_extracted = extract_and_align_face(img2_path, face2_path)
        
        if not face1_extracted or not face2_extracted:
            print("❌ Could not extract and align faces from both images")
            return False, 0.0
        
        # Save faces permanently if requested
        if save_faces:
            permanent_face1 = f"aligned_face1_{os.path.basename(img1_path)}"
            permanent_face2 = f"aligned_face2_{os.path.basename(img2_path)}"
            
            import shutil
            shutil.copy2(face1_path, permanent_face1)
            shutil.copy2(face2_path, permanent_face2)
            print(f"\nSaved aligned faces: {permanent_face1}, {permanent_face2}")
        
        # Get dinohashes for the aligned faces
        print("\nGenerating hash for aligned face 1...")
        hash1 = get_dinohash(face1_path)
        
        print("Generating hash for aligned face 2...")
        hash2 = get_dinohash(face2_path)
        
        if hash1 and hash2:
            print(f"\nFace 1 hash: {hash1}")
            print(f"Face 2 hash: {hash2}")
            
            # Calculate Hamming distance and similarity
            try:
                hash1_int = int(hash1, 16)
                hash2_int = int(hash2, 16)
                hamming_distance = bin(hash1_int ^ hash2_int).count('1')
                
                # Calculate similarity
                total_bits = len(hash1) * 4  # 4 bits per hex digit
                similarity = 1 - (hamming_distance / total_bits)
                
                print(f"\nSimilarity Analysis:")
                print(f"  Hamming distance: {hamming_distance} bits different")
                print(f"  Total bits: {total_bits}")
                print(f"  Similarity score: {similarity:.4f}")
                
                # Check against threshold
                is_same = similarity >= threshold
                
                if is_same:
                    print(f"✅ SAME PERSON: Similarity {similarity:.3f} >= threshold {threshold}")
                else:
                    print(f"❌ DIFFERENT PEOPLE: Similarity {similarity:.3f} < threshold {threshold}")
                
                return is_same, similarity
                
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                return False, 0.0
        else:
            print("❌ Could not generate hashes for aligned faces")
            return False, 0.0
            
    finally:
        # Clean up temporary files
        try:
            os.unlink(face1_path)
            os.unlink(face2_path)
        except:
            pass

# Example usage
if __name__ == "__main__":
    # Compare faces in two images
    img1 = "./elifiles/images/ronnychieng/download (8).jpeg"
    img2 = "./elifiles/images/ronnychieng/download (10).jpeg"
    
    if os.path.exists(img1) and os.path.exists(img2):
        # Try different thresholds to find what works best
        thresholds = [0.95, 0.90, 0.85, 0.80, 0.75]
        
        print("Testing different similarity thresholds:")
        print("="*60)
        
        for thresh in thresholds:
            print(f"\n--- Testing threshold: {thresh} ---")
            is_same, similarity = compare_faces_with_threshold(img1, img2, threshold=thresh, save_faces=(thresh==0.80))
            print(f"Result: {'SAME PERSON' if is_same else 'DIFFERENT PEOPLE'}")
            
        print("\n" + "="*60)
        print("Recommendation: Use the threshold that gives the most reasonable results")
        print("For same person: similarity should be > 0.75-0.85")
        print("For different people: similarity should be < 0.70")
        
    else:
        print("One or both image files not found")
        print(f"Image 1 exists: {os.path.exists(img1)}")
        print(f"Image 2 exists: {os.path.exists(img2)}")