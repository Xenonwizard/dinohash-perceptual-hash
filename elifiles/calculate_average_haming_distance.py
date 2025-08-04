import subprocess
import os
import glob
from itertools import combinations

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

def hamming_distance(hash1, hash2):
    """
    Calculate hamming distance between two hash strings
    
    Args:
        hash1 (str): First hash string
        hash2 (str): Second hash string
        
    Returns:
        int: Number of differing positions, or -1 if lengths don't match
    """
    if len(hash1) != len(hash2):
        return -1
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def get_image_files(folder_path):
    """
    Get all image files from a folder that actually exist
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern_upper = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=False))
    
    # Filter out files that don't actually exist (just in case)
    existing_files = [f for f in image_files if os.path.isfile(f)]
    
    print(f"Found {len(existing_files)} existing image files out of {len(image_files)} detected")
    return sorted(list(set(existing_files)))

def debug_folder_contents(folder_path):
    """
    Debug function to see what's actually in the folder
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return
    
    print(f"\nDEBUG: Contents of {folder_path}:")
    all_files = os.listdir(folder_path)
    print(f"Total files/folders: {len(all_files)}")
    
    for file in sorted(all_files):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            print(f"  FILE: {file}")
        else:
            print(f"  DIR:  {file}")

def analyze_folders_for_threshold(folders=["./images/ronnychieng", "./images/ronnychieng_test"]):
    """
    Analyze hamming distances in multiple folders to determine optimal threshold
    """
    all_results = {}
    
    # First, debug what's in each folder
    for folder_path in folders:
        debug_folder_contents(folder_path)
    
    for folder_path in folders:
        print(f"\n{'='*60}")
        print(f"ANALYZING FOLDER: {folder_path}")
        print(f"{'='*60}")
        
        # Get all image files
        image_files = get_image_files(folder_path)
        print(f"Found {len(image_files)} images")
        
        if len(image_files) < 2:
            print("Need at least 2 images for comparison.")
            continue
        
        # Generate hashes for all images
        print("Generating hashes...")
        image_hashes = {}
        
        for img_path in image_files:
            # Double-check file exists before processing
            if not os.path.isfile(img_path):
                print(f"✗ File not found: {os.path.basename(img_path)}")
                continue
                
            hash_value = get_dinohash(img_path)
            if hash_value:
                image_hashes[img_path] = hash_value
                print(f"✓ {os.path.basename(img_path)}")
            else:
                print(f"✗ Failed to hash: {os.path.basename(img_path)}")
        
        successful_images = list(image_hashes.keys())
        print(f"\nSuccessfully hashed {len(successful_images)} images")
        
        if len(successful_images) < 2:
            print("Not enough images for comparison.")
            continue
        
        # Calculate all hamming distances
        print("\nCalculating hamming distances...")
        distances = []
        
        for img1, img2 in combinations(successful_images, 2):
            hash1 = image_hashes[img1]
            hash2 = image_hashes[img2]
            
            distance = hamming_distance(hash1, hash2)
            
            if distance >= 0:
                distances.append(distance)
                img1_name = os.path.basename(img1)
                img2_name = os.path.basename(img2)
                print(f"{img1_name} vs {img2_name}: {distance}")
        
        if not distances:
            print("No valid distances calculated.")
            continue
        
        # Calculate statistics
        hash_length = len(list(image_hashes.values())[0])
        average_distance = sum(distances) / len(distances)
        min_distance = min(distances)
        max_distance = max(distances)
        
        # Convert to similarity percentages
        avg_similarity = (1 - (average_distance / hash_length)) * 100
        min_similarity = (1 - (max_distance / hash_length)) * 100
        max_similarity = (1 - (min_distance / hash_length)) * 100
        
        print("\n" + "-"*40)
        print(f"RESULTS FOR {os.path.basename(folder_path)}")
        print("-"*40)
        print(f"Total pairs compared: {len(distances)}")
        print(f"Hash length: {hash_length} bits")
        print(f"Average hamming distance: {average_distance:.2f}")
        print(f"Min hamming distance: {min_distance}")
        print(f"Max hamming distance: {max_distance}")
        print(f"Average similarity: {avg_similarity:.1f}%")
        print(f"Best similarity: {max_similarity:.1f}%")
        print(f"Worst similarity: {min_similarity:.1f}%")
        
        # Store results
        all_results[folder_path] = {
            'distances': distances,
            'average_distance': average_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'hash_length': hash_length,
            'avg_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity
        }
    
    # Overall analysis and recommendations
    if all_results:
        print("\n" + "="*60)
        print("OVERALL THRESHOLD RECOMMENDATIONS")
        print("="*60)
        
        # Combine all distances from both folders
        all_distances = []
        for result in all_results.values():
            all_distances.extend(result['distances'])
        
        if all_distances:
            overall_avg = sum(all_distances) / len(all_distances)
            overall_max = max(all_distances)
            overall_min = min(all_distances)
            hash_length = list(all_results.values())[0]['hash_length']  # Should be same for all
            
            overall_avg_sim = (1 - (overall_avg / hash_length)) * 100
            
            print(f"Combined analysis from {len(all_results)} folders:")
            print(f"Total pairs: {len(all_distances)}")
            print(f"Overall average hamming distance: {overall_avg:.2f}")
            print(f"Overall average similarity: {overall_avg_sim:.1f}%")
            
            # Threshold recommendation
            if overall_avg_sim > 80:
                recommended_threshold = overall_avg + (overall_max - overall_avg) * 0.5
                print(f"\n✓ Images show good similarity (avg {overall_avg_sim:.1f}%)")
                print(f"✓ RECOMMENDED HAMMING DISTANCE THRESHOLD: {recommended_threshold:.0f}")
                print(f"✓ RECOMMENDED SIMILARITY THRESHOLD: {(1 - recommended_threshold/hash_length)*100:.1f}%")
                
                print(f"\nTo use in your code:")
                print(f"def are_similar_faces(img1_path, img2_path):")
                print(f"    hash1 = get_dinohash(img1_path)")
                print(f"    hash2 = get_dinohash(img2_path)")
                print(f"    if hash1 and hash2:")
                print(f"        distance = hamming_distance(hash1, hash2)")
                print(f"        return distance <= {recommended_threshold:.0f}")
                print(f"    return False")
            else:
                print(f"\n⚠ Images show low similarity (avg {overall_avg_sim:.1f}%)")
                print(f"⚠ Dinohash may not be ideal for face recognition")
                print(f"⚠ Consider using MTCNN + FaceNet instead")
    
    return all_results

# Example usage
if __name__ == "__main__":
    # Since you're running from elifiles directory, use the correct paths
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Check if images folder exists in current directory
    images_dir = "./images"
    if os.path.exists(images_dir):
        print(f"Found images directory: {os.path.abspath(images_dir)}")
        
        # List subdirectories in images
        subdirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        print(f"Subdirectories in images: {subdirs}")
        
        # Analyze both ronnychieng folders if they exist
        folders_to_analyze = []
        for folder_name in ["ronnychieng", "ronnychieng_test"]:
            folder_path = os.path.join(images_dir, folder_name)
            if os.path.exists(folder_path):
                folders_to_analyze.append(folder_path)
                print(f"✓ Will analyze: {folder_path}")
            else:
                print(f"✗ Folder not found: {folder_path}")
        
        if folders_to_analyze:
            results = analyze_folders_for_threshold(folders_to_analyze)
        else:
            print("No ronnychieng folders found to analyze!")
    else:
        print(f"Images directory not found at: {os.path.abspath(images_dir)}")
        print("Please check your current directory and folder structure.")