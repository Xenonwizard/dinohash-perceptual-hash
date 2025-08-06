import subprocess
import os

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

def compare_images(img1_path, img2_path):
    """
    Compare two images using dinohash
    
    Args:
        img1_path (str): Path to first image
        img2_path (str): Path to second image
        
    Returns:
        bool: True if images have same hash, False otherwise
    """
    hash1 = get_dinohash(img1_path)
    hash2 = get_dinohash(img2_path)
    
    if hash1 and hash2:
        return hash1 == hash2
    else:
        return False

# Example usage
if __name__ == "__main__":
    # Single image hash
    image_path = "./images/original.jpg"
    hash_value = get_dinohash(image_path)
    if hash_value:
        print(f"Hash for {image_path}: {hash_value}")
    
    # Compare two images
    img1 = "./images/ronnychieng/download (8).jpeg"
    img2 = "./images/ronnychieng/download (10).jpeg"
    
    if os.path.exists(img1) and os.path.exists(img2):
        are_same = compare_images(img1, img2)
        print(f"Images are {'the same' if are_same else 'different'}")
    else:
        print("One or both image files not found")
