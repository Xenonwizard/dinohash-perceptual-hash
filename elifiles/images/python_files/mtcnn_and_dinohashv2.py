import subprocess
import os
import cv2
from mtcnn import MTCNN
import tempfile
import numpy as np
from PIL import Image

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

def extract_face_and_save(image_path, output_path):
    """
    Extract the largest face from an image using MTCNN and save it
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save the extracted face
        
    Returns:
        bool: True if face was found and saved, False otherwise
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
        
        # Detect faces
        faces = detector.detect_faces(img_rgb)
        
        if not faces:
            print(f"No face detected in {image_path}")
            return False
        
        # Get the face with highest confidence
        best_face = max(faces, key=lambda x: x['confidence'])
        print(f"Face detected with confidence: {best_face['confidence']:.3f}")
        
        # Extract face bounding box
        x, y, w, h = best_face['box']
        
        # Add some padding around the face (optional)
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_rgb.shape[1] - x, w + 2*padding)
        h = min(img_rgb.shape[0] - y, h + 2*padding)
        
        # Extract face region
        face_region = img_rgb[y:y+h, x:x+w]
        
        # Convert back to BGR for OpenCV saving
        face_bgr = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
        
        # Save the face
        cv2.imwrite(output_path, face_bgr)
        print(f"Face extracted and saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error extracting face from {image_path}: {e}")
        return False

def compare_faces(img1_path, img2_path, save_faces=False):
    """
    Compare faces in two images using MTCNN + dinohash
    
    Args:
        img1_path (str): Path to first image
        img2_path (str): Path to second image
        save_faces (bool): Whether to save extracted faces for inspection
        
    Returns:
        bool: True if faces have same hash, False otherwise
    """
    print(f"Comparing faces in:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print("-" * 50)
    
    # Create temporary files for extracted faces
    with tempfile.NamedTemporaryFile(suffix='_face1.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='_face2.jpg', delete=False) as tmp2:
        
        face1_path = tmp1.name
        face2_path = tmp2.name
    
    try:
        # Extract faces
        print("Extracting face from image 1...")
        face1_extracted = extract_face_and_save(img1_path, face1_path)
        
        print("Extracting face from image 2...")
        face2_extracted = extract_face_and_save(img2_path, face2_path)
        
        if not face1_extracted or not face2_extracted:
            print("❌ Could not extract faces from both images")
            return False
        
        # Save faces permanently if requested
        if save_faces:
            permanent_face1 = f"extracted_face1_{os.path.basename(img1_path)}"
            permanent_face2 = f"extracted_face2_{os.path.basename(img2_path)}"
            
            import shutil
            shutil.copy2(face1_path, permanent_face1)
            shutil.copy2(face2_path, permanent_face2)
            print(f"Saved extracted faces: {permanent_face1}, {permanent_face2}")
        
        # Get dinohashes for the extracted faces
        print("Generating hash for face 1...")
        hash1 = get_dinohash(face1_path)
        
        print("Generating hash for face 2...")
        hash2 = get_dinohash(face2_path)
        
        if hash1 and hash2:
            print(f"Face 1 hash: {hash1}")
            print(f"Face 2 hash: {hash2}")
            
            if hash1 == hash2:
                print("✅ SAME PERSON: Faces have identical hashes")
                return True
            else:
                print("❌ DIFFERENT PEOPLE: Faces have different hashes")
                
                # Calculate similarity (rough estimate)
                try:
                    # Convert hex to binary and calculate hamming distance
                    hash1_int = int(hash1, 16)
                    hash2_int = int(hash2, 16)
                    hamming_distance = bin(hash1_int ^ hash2_int).count('1')
                    
                    # Rough similarity calculation
                    total_bits = len(hash1) * 4  # 4 bits per hex digit
                    similarity = 1 - (hamming_distance / total_bits)
                    print(f"Similarity: {similarity:.3f} ({hamming_distance} different bits out of {total_bits})")
                    
                except:
                    print("Could not calculate similarity")
                
                return False
        else:
            print("❌ Could not generate hashes for extracted faces")
            return False
            
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
        # Set save_faces=True to save the extracted faces for inspection
        are_same_person = compare_faces(img1, img2, save_faces=True)
        print("\n" + "="*50)
        print(f"RESULT: {'SAME PERSON' if are_same_person else 'DIFFERENT PEOPLE'}")
    else:
        print("One or both image files not found")
        print(f"Image 1 exists: {os.path.exists(img1)}")
        print(f"Image 2 exists: {os.path.exists(img2)}")