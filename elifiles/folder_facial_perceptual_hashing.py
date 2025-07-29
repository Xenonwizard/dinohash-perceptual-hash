import subprocess
import os
from mtcnn import MTCNN
import tempfile
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
import math
import glob
from itertools import combinations

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

def align_face_pil(image, left_eye, right_eye):
    """Align face using PIL instead of OpenCV"""
    # Calculate angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
    # Rotate image
    rotated = image.rotate(-angle, expand=False, center=((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2))
    
    return rotated

def enhance_face_image(pil_image):
    """Enhance face image for better hashing"""
    # Normalize size
    normalized = pil_image.resize((200, 200), Image.Resampling.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(normalized)
    enhanced = enhancer.enhance(1.2)
    
    # Enhance sharpness slightly
    sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
    final = sharpness_enhancer.enhance(1.1)
    
    return final

def extract_and_align_face_pil(image_path, output_path):
    """Extract and align face using PIL and MTCNN"""
    try:
        # Load image with PIL
        pil_img = Image.open(image_path)
        img_array = np.array(pil_img)
        
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Detect faces
        faces = detector.detect_faces(img_array)
        
        if not faces:
            return False, "No face detected"
        
        # Get best face
        best_face = max(faces, key=lambda x: x['confidence'])
        
        if best_face['confidence'] < 0.9:
            return False, f"Low confidence: {best_face['confidence']:.3f}"
        
        if 'keypoints' not in best_face:
            return False, "No landmarks detected"
        
        keypoints = best_face['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Extract face region with padding
        x, y, w, h = best_face['box']
        padding = max(w, h) // 4
        
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(pil_img.width - x, w + 2*padding)
        h = min(pil_img.height - y, h + 2*padding)
        
        # Crop face
        face_crop = pil_img.crop((x, y, x + w, y + h))
        
        # Adjust eye coordinates relative to crop
        adjusted_left_eye = (left_eye[0] - x, left_eye[1] - y)
        adjusted_right_eye = (right_eye[0] - x, right_eye[1] - y)
        
        # Align face
        aligned_face = align_face_pil(face_crop, adjusted_left_eye, adjusted_right_eye)
        
        # Enhance the face
        enhanced_face = enhance_face_image(aligned_face)
        
        # Save
        enhanced_face.save(output_path, 'JPEG', quality=95)
        
        return True, f"Success (confidence: {best_face['confidence']:.3f})"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def generate_face_hashes(face_image_path):
    """Generate multiple face-specific hashes"""
    try:
        print(f"    Loading image: {face_image_path}")
        pil_img = Image.open(face_image_path)
        
        # Convert to grayscale for hashing
        print("    Converting to grayscale...")
        gray = pil_img.convert('L')
        
        hashes = {}
        
        # Try each hash type individually to isolate errors
        try:
            print("    Generating phash_8...")
            hashes['phash_8'] = str(imagehash.phash(gray, hash_size=8))
        except Exception as e:
            print(f"    ❌ phash_8 failed: {e}")
            hashes['phash_8'] = None
            
        try:
            print("    Generating ahash_8...")
            hashes['ahash_8'] = str(imagehash.average_hash(gray, hash_size=8))
        except Exception as e:
            print(f"    ❌ ahash_8 failed: {e}")
            hashes['ahash_8'] = None
            
        try:
            print("    Generating dhash_8...")
            hashes['dhash_8'] = str(imagehash.dhash(gray, hash_size=8))
        except Exception as e:
            print(f"    ❌ dhash_8 failed: {e}")
            hashes['dhash_8'] = None
            
        try:
            print("    Generating dinohash...")
            hashes['dinohash'] = get_dinohash(face_image_path)
        except Exception as e:
            print(f"    ❌ dinohash failed: {e}")
            hashes['dinohash'] = None
        
        print(f"    Generated {sum(1 for v in hashes.values() if v is not None)} out of {len(hashes)} hashes")
        
        # Return hashes if at least one succeeded
        if any(v is not None for v in hashes.values()):
            return hashes
        else:
            print("    ❌ All hash generation methods failed")
            return None
        
    except Exception as e:
        print(f"Error in generate_face_hashes: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_hash_similarity(hash1, hash2, hash_type):
    """Calculate similarity between two hashes"""
    if not hash1 or not hash2:
        return 0
    
    try:
        if hash_type == 'dinohash':
            # Handle hex dinohash
            h1_int = int(hash1, 16)
            h2_int = int(hash2, 16)
            hamming_dist = bin(h1_int ^ h2_int).count('1')
            total_bits = len(hash1) * 4
            return 1 - (hamming_dist / total_bits)
        else:
            # Handle imagehash
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            hamming_dist = h1 - h2
            total_bits = len(str(h1))
            return max(0, 1 - (hamming_dist / total_bits))
    except:
        return 0

def comprehensive_face_comparison_pil(img1_path, img2_path, save_faces=False, verbose=True):
    """Compare faces using PIL-based processing"""
    if verbose:
        print(f"Comparing: {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='_face1.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='_face2.jpg', delete=False) as tmp2:
        
        face1_path = tmp1.name
        face2_path = tmp2.name
    
    try:
        # Extract and align faces
        success1, msg1 = extract_and_align_face_pil(img1_path, face1_path)
        if not success1:
            if verbose:
                print(f"  ❌ Image 1 failed: {msg1}")
            return False, {'error': f"Image 1: {msg1}"}
        
        success2, msg2 = extract_and_align_face_pil(img2_path, face2_path)
        if not success2:
            if verbose:
                print(f"  ❌ Image 2 failed: {msg2}")
            return False, {'error': f"Image 2: {msg2}"}
        
        # Save processed faces if requested
        if save_faces:
            import shutil
            shutil.copy2(face1_path, f"processed_face1_{os.path.basename(img1_path)}")
            shutil.copy2(face2_path, f"processed_face2_{os.path.basename(img2_path)}")
        
        # Generate hashes
        hashes1 = generate_face_hashes(face1_path)
        hashes2 = generate_face_hashes(face2_path)
        
        if not hashes1 or not hashes2:
            if verbose:
                print("  ❌ Failed to generate hashes")
            return False, {'error': "Hash generation failed"}
        
        # DEBUG: Print actual hash values
        if verbose:
            print("  🔍 DEBUG - Hash values:")
            for hash_type in hashes1.keys():
                h1 = hashes1.get(hash_type)
                h2 = hashes2.get(hash_type)
                print(f"    {hash_type}: {h1} vs {h2}")
        
        # Compare hashes
        similarities = []
        detailed_results = {}
        
        if verbose:
            print("  🔍 DEBUG - Individual similarities:")
        
        for hash_type in hashes1.keys():
            similarity = calculate_hash_similarity(hashes1[hash_type], hashes2[hash_type], hash_type)
            if hashes1[hash_type] is not None and hashes2[hash_type] is not None:
                similarities.append(similarity)
                detailed_results[hash_type] = similarity
                if verbose:
                    print(f"    {hash_type}: {similarity:.4f}")
            else:
                detailed_results[hash_type] = 0.0
                if verbose:
                    print(f"    {hash_type}: FAILED (None hash)")
        
        # Calculate overall similarity with weights - only use successful hashes
        weights = {
            'dinohash': 0.25,
            'phash_8': 0.20,
            'ahash_8': 0.20,
            'dhash_8': 0.20,
            'whash_8': 0.15
        }
        
        # Calculate weighted similarity only from successful hashes
        total_weight = 0
        weighted_sum = 0
        
        for hash_type, weight in weights.items():
            if hash_type in detailed_results and detailed_results[hash_type] > 0:
                weighted_sum += detailed_results[hash_type] * weight
                total_weight += weight
        
        # Normalize by actual total weight used
        weighted_similarity = weighted_sum / total_weight if total_weight > 0 else 0
        
        average_similarity = np.mean(similarities) if similarities else 0
        
        if verbose:
            print(f"  🔍 DEBUG - Calculations:")
            print(f"    Weighted sum: {weighted_sum:.4f}")
            print(f"    Total weight: {total_weight:.4f}")
            print(f"    Weighted similarity: {weighted_similarity:.4f}")
            print(f"    Average similarity: {average_similarity:.4f}")
        
        # More generous decision logic for face comparison
        if weighted_similarity >= 0.75:
            decision = "SAME PERSON (High Confidence)"
            is_same = True
        elif weighted_similarity >= 0.60:
            decision = "SAME PERSON (Medium Confidence)"
            is_same = True
        elif weighted_similarity >= 0.45:
            decision = "POSSIBLY SAME PERSON (Low Confidence)"
            is_same = True
        elif weighted_similarity >= 0.30:
            decision = "POSSIBLY DIFFERENT (Low Confidence)"
            is_same = False
        else:
            decision = "DIFFERENT PEOPLE (High Confidence)"
            is_same = False
        
        if verbose:
            print(f"  Final: {weighted_similarity:.3f} - {decision}")
        
        return is_same, {
            'weighted_similarity': weighted_similarity,
            'average_similarity': average_similarity,
            'detailed_results': detailed_results,
            'decision': decision
        }
        
    finally:
        # Cleanup
        try:
            os.unlink(face1_path)
            os.unlink(face2_path)
        except:
            pass

def batch_test_folders(ronnychieng_folder, test_folder):
    """Test face comparison across two folders"""
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    
    ronny_images = []
    test_images = []
    
    print("Scanning for images...")
    for ext in image_extensions:
        ronny_images.extend(glob.glob(os.path.join(ronnychieng_folder, ext)))
        test_images.extend(glob.glob(os.path.join(test_folder, ext)))
    
    print(f"Found {len(ronny_images)} images in ronnychieng folder")
    print(f"Found {len(test_images)} images in test folder")
    
    if len(ronny_images) == 0:
        print(f"❌ No images found in {ronnychieng_folder}")
        return
    
    if len(test_images) == 0:
        print(f"❌ No images found in {test_folder}")
        return
    
    # Test within ronnychieng folder (should be same person)
    print("\n" + "="*70)
    print("TESTING RONNY CHIENG IMAGES AGAINST EACH OTHER")
    print("(Should detect as SAME PERSON)")
    print("="*70)
    
    ronny_pairs = list(combinations(ronny_images, 2))
    same_person_correct = 0
    same_person_total = len(ronny_pairs)
    
    for i, (img1, img2) in enumerate(ronny_pairs):
        print(f"\n--- Ronny Test {i+1}/{same_person_total} ---")
        is_same, results = comprehensive_face_comparison_pil(img1, img2, save_faces=False, verbose=True)
        
        if results and 'error' not in results:
            similarity = results.get('weighted_similarity', 0)
            if is_same:
                same_person_correct += 1
                print(f"  ✅ CORRECT: Detected as same person (Score: {similarity:.3f})")
            else:
                print(f"  ❌ INCORRECT: Detected as different people (Score: {similarity:.3f})")
        else:
            print(f"  ⚠️  FAILED: {results.get('error', 'Unknown error')}")
            same_person_total -= 1  # Don't count failed comparisons
    
    # Test ronnychieng vs test folder (should be different people)
    print("\n" + "="*70)
    print("TESTING RONNY CHIENG VS TEST IMAGES")
    print("(Should detect as DIFFERENT PEOPLE)")
    print("="*70)
    
    different_person_correct = 0
    different_person_total = 0
    
    # Test first few images from each folder to avoid too many comparisons
    max_ronny = min(3, len(ronny_images))
    max_test = min(3, len(test_images))
    
    for i, ronny_img in enumerate(ronny_images[:max_ronny]):
        for j, test_img in enumerate(test_images[:max_test]):
            different_person_total += 1
            print(f"\n--- Different People Test {different_person_total} ---")
            print(f"  Ronny: {os.path.basename(ronny_img)}")
            print(f"  Test:  {os.path.basename(test_img)}")
            
            is_same, results = comprehensive_face_comparison_pil(ronny_img, test_img, save_faces=False, verbose=False)
            
            if results and 'error' not in results:
                similarity = results.get('weighted_similarity', 0)
                if not is_same:
                    different_person_correct += 1
                    print(f"  ✅ CORRECT: Detected as different people (Score: {similarity:.3f})")
                else:
                    print(f"  ❌ INCORRECT: Detected as same person (Score: {similarity:.3f})")
                print(f"     {results.get('decision', '')}")
            else:
                print(f"  ⚠️  FAILED: {results.get('error', 'Unknown error')}")
                different_person_total -= 1  # Don't count failed comparisons
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY RESULTS")
    print("="*70)
    
    if same_person_total > 0:
        same_accuracy = (same_person_correct / same_person_total) * 100
        print(f"Same Person Detection:     {same_person_correct}/{same_person_total} ({same_accuracy:.1f}% correct)")
    
    if different_person_total > 0:
        different_accuracy = (different_person_correct / different_person_total) * 100
        print(f"Different People Detection: {different_person_correct}/{different_person_total} ({different_accuracy:.1f}% correct)")
    
    if same_person_total > 0 and different_person_total > 0:
        overall_correct = same_person_correct + different_person_correct
        overall_total = same_person_total + different_person_total
        overall_accuracy = (overall_correct / overall_total) * 100
        print(f"Overall Accuracy:          {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)")
    
    print("\nRecommendations:")
    if same_person_total > 0 and (same_person_correct / same_person_total) < 0.7:
        print("- Same person detection is low. Consider lowering similarity threshold.")
    if different_person_total > 0 and (different_person_correct / different_person_total) < 0.7:
        print("- Different people detection is low. Consider raising similarity threshold.")
    if same_person_total > 0 and different_person_total > 0:
        total_acc = (same_person_correct + different_person_correct) / (same_person_total + different_person_total)
        if total_acc > 0.8:
            print("- ✅ Good performance! Face hashing is working well for your dataset.")
        elif total_acc > 0.6:
            print("- ⚠️  Moderate performance. Consider using dedicated face recognition instead.")
        else:
            print("- ❌ Poor performance. Strongly recommend switching to proper face recognition.")

# Example usage
if __name__ == "__main__":
    # Define your folder paths
    ronnychieng_folder = "./elifiles/images/ronnychieng/"
    test_folder = "./elifiles/images/test/"  # Update this path as needed
    
    # Check if folders exist
    if not os.path.exists(ronnychieng_folder):
        print(f"❌ Ronny Chieng folder not found: {ronnychieng_folder}")
        print("Please update the ronnychieng_folder path")
        exit(1)
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        print("Please update the test_folder path or create the folder")
        exit(1)
    
    # Run the batch test
    batch_test_folders(ronnychieng_folder, test_folder)