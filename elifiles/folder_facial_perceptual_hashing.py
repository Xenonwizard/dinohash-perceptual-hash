import subprocess
import os
import cv2
import glob
from mtcnn import MTCNN
import tempfile
import numpy as np
from PIL import Image
import imagehash
import math

def get_dinohash(image_path):
    """Get dinohash for an image using the command line version"""
    try:
        result = subprocess.run([
            'python3', 'hashes/dinohash.py', image_path
        ], 
        capture_output=True, 
        text=True, 
        cwd='/home/ssm-user/dinohash-perceptual-hash'
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Failed to run dinohash command: {e}")
        return None

def face_specific_phash(face_image_path, hash_size=16):
    """Generate face-specific perceptual hash using multiple methods"""
    try:
        # Load image
        img = cv2.imread(face_image_path)
        pil_img = Image.open(face_image_path)
        
        # Method 1: Standard pHash with larger hash size for faces
        gray_pil = pil_img.convert('L').resize((128, 128))  # Larger for faces
        phash = imagehash.phash(gray_pil, hash_size=hash_size)
        
        # Method 2: Average hash (more robust to lighting)
        ahash = imagehash.average_hash(gray_pil, hash_size=hash_size)
        
        # Method 3: Difference hash (good for facial structure)
        dhash = imagehash.dhash(gray_pil, hash_size=hash_size)
        
        # Method 4: Wavelet hash (captures textures)
        whash = imagehash.whash(gray_pil, hash_size=hash_size)
        
        return {
            'phash': str(phash),
            'ahash': str(ahash), 
            'dhash': str(dhash),
            'whash': str(whash),
            'dinohash': get_dinohash(face_image_path)
        }
        
    except Exception as e:
        print(f"Error generating face hashes: {e}")
        return None

def facial_region_hash(face_image_path):
    """Generate hash based on facial regions"""
    try:
        img = cv2.imread(face_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Define facial regions
        regions = {
            'forehead': gray[0:int(height*0.3), int(width*0.2):int(width*0.8)],
            'eyes': gray[int(height*0.2):int(height*0.5), :],
            'nose': gray[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)],
            'mouth': gray[int(height*0.6):int(height*0.9), int(width*0.2):int(width*0.8)],
            'chin': gray[int(height*0.7):, int(width*0.3):int(width*0.7)]
        }
        
        region_hashes = {}
        for region_name, region in regions.items():
            if region.size > 0:
                # Resize region to standard size
                resized = cv2.resize(region, (32, 32))
                
                # Simple DCT-based hash
                dct = cv2.dct(np.float32(resized))
                dct_low = dct[:8, :8]
                median = np.median(dct_low)
                hash_bits = (dct_low > median).flatten()
                region_hashes[region_name] = ''.join(['1' if bit else '0' for bit in hash_bits])
        
        return region_hashes
        
    except Exception as e:
        print(f"Error generating region hashes: {e}")
        return None

def compare_face_hashes(hashes1, hashes2):
    """Compare face hashes using multiple methods"""
    if not hashes1 or not hashes2:
        return {}
    
    results = {}
    
    # Compare each hash type
    for hash_type in ['phash', 'ahash', 'dhash', 'whash']:
        if hash_type in hashes1 and hash_type in hashes2:
            # Calculate Hamming distance for imagehash
            try:
                h1 = imagehash.hex_to_hash(hashes1[hash_type])
                h2 = imagehash.hex_to_hash(hashes2[hash_type])
                hamming_dist = h1 - h2
                similarity = 1 - (hamming_dist / len(str(h1)))
                results[hash_type] = {
                    'similarity': max(0, similarity),
                    'hamming_distance': hamming_dist
                }
            except:
                results[hash_type] = {'similarity': 0, 'hamming_distance': float('inf')}
    
    # Compare dinohash
    if 'dinohash' in hashes1 and 'dinohash' in hashes2 and hashes1['dinohash'] and hashes2['dinohash']:
        try:
            hash1_int = int(hashes1['dinohash'], 16)
            hash2_int = int(hashes2['dinohash'], 16)
            hamming_distance = bin(hash1_int ^ hash2_int).count('1')
            total_bits = len(hashes1['dinohash']) * 4
            similarity = 1 - (hamming_distance / total_bits)
            results['dinohash'] = {
                'similarity': similarity,
                'hamming_distance': hamming_distance
            }
        except:
            results['dinohash'] = {'similarity': 0, 'hamming_distance': float('inf')}
    
    return results

def compare_region_hashes(regions1, regions2):
    """Compare facial region hashes"""
    if not regions1 or not regions2:
        return {}
    
    region_similarities = {}
    for region in regions1.keys():
        if region in regions2:
            h1, h2 = regions1[region], regions2[region]
            diff = sum(c1 != c2 for c1, c2 in zip(h1, h2))
            similarity = 1 - (diff / len(h1))
            region_similarities[region] = similarity
    
    return region_similarities

def align_face_with_landmarks(image, keypoints):
    """Align face using eye landmarks"""
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2
    center = (center_x, center_y)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return aligned

def extract_and_align_face(image_path, output_path):
    """Extract and align face for hashing"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(img_rgb)
        
        if not faces:
            print(f"No face detected in {image_path}")
            return False
        
        best_face = max(faces, key=lambda x: x['confidence'])
        print(f"Face confidence: {best_face['confidence']:.3f}")
        
        if 'keypoints' not in best_face:
            return False
        
        keypoints = best_face['keypoints']
        x, y, w, h = best_face['box']
        
        # Extract with padding
        padding = max(w, h) // 4
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_rgb.shape[1] - x, w + 2*padding)
        h = min(img_rgb.shape[0] - y, h + 2*padding)
        
        face_region = img_rgb[y:y+h, x:x+w]
        
        # Adjust keypoints
        adjusted_keypoints = {}
        for key, point in keypoints.items():
            adjusted_keypoints[key] = (point[0] - x, point[1] - y)
        
        # Align face
        aligned_face = align_face_with_landmarks(face_region, adjusted_keypoints)
        
        # Normalize size
        normalized = cv2.resize(aligned_face, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        # Save
        face_bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, face_bgr)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def comprehensive_face_comparison(img1_path, img2_path, save_faces=True):
    """Comprehensive face comparison using multiple perceptual hashing methods"""
    print(f"Comprehensive face comparison:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print("-" * 60)
    
    with tempfile.NamedTemporaryFile(suffix='_face1.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='_face2.jpg', delete=False) as tmp2:
        
        face1_path = tmp1.name
        face2_path = tmp2.name
    
    try:
        # Extract and align faces
        print("Extracting faces...")
        if not extract_and_align_face(img1_path, face1_path):
            return False, {}
        if not extract_and_align_face(img2_path, face2_path):
            return False, {}
        
        if save_faces:
            import shutil
            shutil.copy2(face1_path, f"face1_{os.path.basename(img1_path)}")
            shutil.copy2(face2_path, f"face2_{os.path.basename(img2_path)}")
        
        # Generate multiple types of hashes
        print("Generating face-specific hashes...")
        hashes1 = face_specific_phash(face1_path)
        hashes2 = face_specific_phash(face2_path)
        
        print("Generating region-based hashes...")
        regions1 = facial_region_hash(face1_path)
        regions2 = facial_region_hash(face2_path)
        
        # Compare hashes
        hash_results = compare_face_hashes(hashes1, hashes2)
        region_results = compare_region_hashes(regions1, regions2)
        
        # Print detailed results
        print("\n" + "="*50)
        print("HASH COMPARISON RESULTS:")
        print("="*50)
        
        overall_similarities = []
        
        for hash_type, result in hash_results.items():
            similarity = result['similarity']
            overall_similarities.append(similarity)
            print(f"{hash_type.upper():>10}: {similarity:.4f} ({result['hamming_distance']} bits different)")
        
        print("\nREGION COMPARISON RESULTS:")
        print("-"*30)
        for region, similarity in region_results.items():
            overall_similarities.append(similarity)
            print(f"{region.capitalize():>10}: {similarity:.4f}")
        
        # Calculate overall similarity
        avg_similarity = np.mean(overall_similarities) if overall_similarities else 0
        weighted_similarity = (
            hash_results.get('dinohash', {}).get('similarity', 0) * 0.3 +
            hash_results.get('phash', {}).get('similarity', 0) * 0.3 +
            hash_results.get('dhash', {}).get('similarity', 0) * 0.2 +
            np.mean(list(region_results.values())) * 0.2 if region_results else 0
        )
        
        print(f"\nOVERALL SIMILARITY:")
        print(f"  Average: {avg_similarity:.4f}")
        print(f"  Weighted: {weighted_similarity:.4f}")
        
        # Decision thresholds
        thresholds = {
            'very_high': 0.85,
            'high': 0.75,
            'medium': 0.65,
            'low': 0.55
        }
        
        decision = "DIFFERENT PEOPLE"
        if weighted_similarity >= thresholds['very_high']:
            decision = "SAME PERSON (Very High Confidence)"
        elif weighted_similarity >= thresholds['high']:
            decision = "SAME PERSON (High Confidence)"
        elif weighted_similarity >= thresholds['medium']:
            decision = "POSSIBLY SAME PERSON (Medium Confidence)"
        elif weighted_similarity >= thresholds['low']:
            decision = "POSSIBLY DIFFERENT PEOPLE (Low Confidence)"
        
        print(f"\nFINAL DECISION: {decision}")
        
        return weighted_similarity >= thresholds['medium'], {
            'weighted_similarity': weighted_similarity,
            'average_similarity': avg_similarity,
            'hash_results': hash_results,
            'region_results': region_results,
            'decision': decision
        }
        
    finally:
        try:
            os.unlink(face1_path)
            os.unlink(face2_path)
        except:
            pass

# Example usage
if __name__ == "__main__":
    # Test folders
    ronnychieng_folder = "./elifiles/images/ronnychieng/"
    test_folder = "./elifiles/images/ronnychieng_test/"  # Adjust this path as needed
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    ronny_images = []
    test_images = []
    
    for ext in image_extensions:
        ronny_images.extend(glob.glob(os.path.join(ronnychieng_folder, ext)))
        test_images.extend(glob.glob(os.path.join(test_folder, ext)))
    
    print(f"Found {len(ronny_images)} images in ronnychieng folder")
    print(f"Found {len(test_images)} images in test folder")
    
    # Test within ronnychieng folder (should be same person)
    print("\n" + "="*60)
    print("TESTING RONNY CHIENG IMAGES (Should be SAME PERSON)")
    print("="*60)
    
    for i, (img1, img2) in enumerate(combinations(ronny_images, 2)):
        print(f"\n--- Test {i+1}: Ronny vs Ronny ---")
        is_same, results = comprehensive_face_comparison_pil(img1, img2, save_faces=False)
        similarity = results.get('weighted_similarity', 0) if results else 0
        print(f"Result: {'✅ SAME' if is_same else '❌ DIFFERENT'} (Score: {similarity:.3f})")
    
    # Test ronnychieng vs test folder (should be different people)
    print("\n" + "="*60)
    print("TESTING RONNY VS TEST IMAGES (Should be DIFFERENT PEOPLE)")
    print("="*60)
    
    test_count = 0
    for ronny_img in ronny_images[:3]:  # Test first 3 ronny images
        for test_img in test_images[:3]:  # Against first 3 test images
            test_count += 1
            print(f"\n--- Test {test_count}: Ronny vs Test ---")
            print(f"Ronny: {os.path.basename(ronny_img)}")
            print(f"Test: {os.path.basename(test_img)}")
            is_same, results = comprehensive_face_comparison_pil(ronny_img, test_img, save_faces=False)
            similarity = results.get('weighted_similarity', 0) if results else 0
            print(f"Result: {'✅ SAME' if is_same else '❌ DIFFERENT'} (Score: {similarity:.3f})")