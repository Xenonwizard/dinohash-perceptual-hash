#!/usr/bin/env python3
"""
Python 3.11+ Compatible Face Recognition Tester
Fixed for strict method ordering in newer Python versions
"""

import os
import glob
import tempfile
import subprocess
import numpy as np
import pandas as pd
from itertools import combinations
from PIL import Image, ImageEnhance
import imagehash
import math
from mtcnn import MTCNN
from sklearn.metrics import f1_score, accuracy_score
from scipy.spatial.distance import euclidean, cosine, chebyshev
try:
    from scipy.spatial.distance import manhattan
except ImportError:
    from scipy.spatial.distance import cityblock as manhattan

class SimpleFaceTester:
    """Python 3.11+ compatible face tester with proper method ordering"""
    
    def __init__(self):
        """Initialize with deferred method assignment for Python 3.11+ compatibility"""
        self.detector = MTCNN()
        self.results = []
        
        # Initialize algorithms after all methods are defined
        self._setup_algorithms_and_metrics()
    
    def _setup_algorithms_and_metrics(self):
        """Setup algorithms and metrics after all methods are defined"""
        # Core algorithms (most scientifically validated)
        self.algorithms = {
            'aHash': lambda img: str(imagehash.average_hash(img, 8)),
            'pHash': lambda img: str(imagehash.phash(img, 8)),        # Most cited
            'dHash': lambda img: str(imagehash.dhash(img, 8)),        # Most efficient
            'wHash': lambda img: str(imagehash.whash(img, 8)),        # Frequency domain
            'dinohash': self.compute_dinohash,                        # Deep learning
        }
        
        # Core distance metrics + best ones for DinoHash
        self.metrics = {
            'hamming': self.hamming_dist,      # Standard for hashing
            'euclidean': self.euclidean_dist,  # Good for embeddings
            'cosine': self.cosine_dist,        # Best for high-dimensional (DinoHash)
            'jaccard': self.jaccard_dist,      # Set-based similarity (binary features)
            'chebyshev': self.chebyshev_dist,  # Maximum difference metric
        }
    
    def hamming_dist(self, h1, h2):
        """Hamming distance (normalized) - use ImageHash built-in"""
        try:
            hash1 = imagehash.hex_to_hash(h1)
            hash2 = imagehash.hex_to_hash(h2)
            return (hash1 - hash2) / 64  # Normalized by 64 bits (8x8)
        except Exception as e:
            print(f"Hamming distance error: {e}")
            return 1.0
    
    def euclidean_dist(self, h1, h2):
        """Euclidean distance using proper L2 norm calculation"""
        try:
            # Convert hex hashes to bit arrays
            hash1 = imagehash.hex_to_hash(h1)
            hash2 = imagehash.hex_to_hash(h2)
            
            # Get the actual hash arrays (not just the difference count)
            bits1 = hash1.hash.flatten().astype(np.float64)
            bits2 = hash2.hash.flatten().astype(np.float64)
            
            # Calculate true Euclidean distance: sqrt(sum((x1-x2)^2))
            squared_diffs = (bits1 - bits2) ** 2
            euclidean_distance = np.sqrt(np.sum(squared_diffs))
            
            # Normalize by maximum possible distance (sqrt(64) for 8x8 hash)
            max_distance = np.sqrt(len(bits1))
            return euclidean_distance / max_distance
            
        except Exception as e:
            print(f"Euclidean distance error: {e}")
            return 1.0
    
    def chebyshev_dist(self, h1, h2):
        """Chebyshev distance (L∞ norm) - maximum difference in any dimension"""
        try:
            # Convert hex hashes to arrays
            bits1 = np.array([int(c, 16) for c in h1])
            bits2 = np.array([int(c, 16) for c in h2])
            
            # Calculate Chebyshev distance (max absolute difference)
            max_diff = np.max(np.abs(bits1 - bits2))
            return max_diff / 15  # Normalize by max hex digit value
        except Exception as e:
            print(f"Chebyshev distance error: {e}")
            return self.hamming_dist(h1, h2)
    
    def cosine_dist(self, h1, h2, algo_name=None):
        """Cosine distance - optimized for DinoHash"""
        try:
            if algo_name == 'dinohash' and h1 and h2:
                # DinoHash is hex string - convert to bit representation
                try:
                    bits1 = np.array([int(bit) for bit in bin(int(h1, 16))[2:].zfill(len(h1)*4)])
                    bits2 = np.array([int(bit) for bit in bin(int(h2, 16))[2:].zfill(len(h2)*4)])
                except ValueError:
                    # Fallback for invalid hex
                    return self.hamming_dist(h1, h2)
            else:
                # Traditional hashes - use hex digits as features
                bits1 = np.array([int(c, 16) for c in h1])
                bits2 = np.array([int(c, 16) for c in h2])
            
            # Calculate cosine distance
            dot_product = np.dot(bits1, bits2)
            norm1 = np.linalg.norm(bits1)
            norm2 = np.linalg.norm(bits2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return 1 - cosine_sim  # Convert to distance
        except Exception as e:
            print(f"Cosine distance error: {e}")
            # Fallback to hamming distance
            return self.hamming_dist(h1, h2)
    
    def jaccard_dist(self, h1, h2, algo_name=None):
        """Jaccard distance - good for sparse binary features"""
        try:
            if algo_name == 'dinohash' and h1 and h2:
                # DinoHash - convert hex to binary
                try:
                    bits1 = set([i for i, bit in enumerate(bin(int(h1, 16))[2:].zfill(len(h1)*4)) if bit == '1'])
                    bits2 = set([i for i, bit in enumerate(bin(int(h2, 16))[2:].zfill(len(h2)*4)) if bit == '1'])
                except ValueError:
                    return self.hamming_dist(h1, h2)
            else:
                # Traditional hashes - find positions of '1' bits
                hash_obj1 = imagehash.hex_to_hash(h1)
                hash_obj2 = imagehash.hex_to_hash(h2)
                bits1 = set([i for i, bit in enumerate(str(hash_obj1)) if bit == '1'])
                bits2 = set([i for i, bit in enumerate(str(hash_obj2)) if bit == '1'])
            
            intersection = len(bits1 & bits2)
            union = len(bits1 | bits2)
            
            if union == 0:
                return 0.0
            return 1 - (intersection / union)
        except Exception as e:
            print(f"Jaccard distance error: {e}")
            return self.hamming_dist(h1, h2)
    
    def compute_dinohash(self, img):
        """Compute DinoHash - save image temporarily and call external script"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                img.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            
            try:
                result = subprocess.run([
                    'python3', 'hashes/dinohash.py', tmp_path
                ], 
                capture_output=True, 
                text=True, 
                cwd='/home/ssm-user/dinohash-perceptual-hash',
                timeout=30  # 30 second timeout
                )
                
                if result.returncode == 0:
                    hash_result = result.stdout.strip()
                    if hash_result:
                        return hash_result
                    else:
                        return None
                else:
                    print(f"  ⚠️  DinoHash error: {result.stderr}")
                    return None
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"  ⚠️  DinoHash failed: {e}")
            return None
    
    def extract_face(self, img_path):
        """Extract and preprocess face"""
        try:
            img = Image.open(img_path)
            faces = self.detector.detect_faces(np.array(img))
            
            if not faces or faces[0]['confidence'] < 0.9:
                return None
            
            # Get best face
            face = max(faces, key=lambda x: x['confidence'])
            x, y, w, h = face['box']
            
            # Extract with padding
            padding = max(w, h) // 4
            x, y = max(0, x-padding), max(0, y-padding)
            w = min(img.width-x, w+2*padding)
            h = min(img.height-y, h+2*padding)
            
            face_crop = img.crop((x, y, x+w, y+h))
            
            # Align if keypoints available
            if 'keypoints' in face:
                left_eye = face['keypoints']['left_eye']
                right_eye = face['keypoints']['right_eye']
                
                # Calculate rotation angle
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = math.degrees(math.atan2(dy, dx))
                
                # Rotate face
                center = ((left_eye[0] + right_eye[0])/2 - x, (left_eye[1] + right_eye[1])/2 - y)
                face_crop = face_crop.rotate(-angle, center=center)
            
            # Normalize
            face_norm = face_crop.resize((128, 128)).convert('L')
            enhancer = ImageEnhance.Contrast(face_norm)
            return enhancer.enhance(1.2)
            
        except Exception as e:
            print(f"Face extraction failed for {img_path}: {e}")
            return None
    
    def compute_hashes(self, face_img):
        """Compute all hash types for a face image"""
        hashes = {}
        for name, func in self.algorithms.items():
            try:
                hash_result = func(face_img)
                if hash_result is not None:
                    hashes[name] = hash_result
                else:
                    print(f"  ⚠️  {name} returned None")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
        return hashes
    
    def compare_faces(self, img1_path, img2_path, same_person=True):
        """Compare two face images with all algorithm-metric combinations"""
        face1 = self.extract_face(img1_path)
        face2 = self.extract_face(img2_path)
        
        if face1 is None or face2 is None:
            return  # Skip failed extractions
        
        hashes1 = self.compute_hashes(face1)
        hashes2 = self.compute_hashes(face2)
        
        # Test all combinations
        for algo_name in self.algorithms.keys():
            if algo_name not in hashes1 or algo_name not in hashes2:
                continue  # Skip if hash computation failed
                
            for metric_name, metric_func in self.metrics.items():
                try:
                    # Pass algorithm name for DinoHash-specific handling
                    if metric_name in ['cosine', 'jaccard']:
                        distance = metric_func(hashes1[algo_name], hashes2[algo_name], algo_name)
                    else:
                        distance = metric_func(hashes1[algo_name], hashes2[algo_name])
                    
                    self.results.append({
                        'algorithm': algo_name,
                        'metric': metric_name,
                        'distance': distance,
                        'same_person': same_person,
                        'img1': os.path.basename(img1_path),
                        'img2': os.path.basename(img2_path)
                    })
                except Exception as e:
                    print(f"Error with {algo_name}-{metric_name}: {e}")
    
    def test_folder(self, folder_path, max_pairs=30):
        """Test all image pairs in a folder"""
        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if len(images) < 2:
            print(f"Need at least 2 images in {folder_path}")
            return
        
        print(f"Found {len(images)} images")
        
        # Generate pairs (same person = True since all images are of same person)
        pairs = list(combinations(images, 2))[:max_pairs]
        
        print(f"Testing {len(pairs)} pairs...")
        for i, (img1, img2) in enumerate(pairs, 1):
            print(f"  {i}/{len(pairs)}: {os.path.basename(img1)} vs {os.path.basename(img2)}")
            self.compare_faces(img1, img2, same_person=True)
    
    def analyze_results(self):
        """Analyze and return performance results"""
        if not self.results:
            print("No results to analyze")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        performance = []
        
        for algo in df['algorithm'].unique():
            for metric in df['metric'].unique():
                subset = df[(df['algorithm'] == algo) & (df['metric'] == metric)]
                if len(subset) < 3:
                    continue
                
                distances = subset['distance'].values
                labels = subset['same_person'].values
                
                # Find optimal threshold using median
                threshold = np.median(distances)
                predictions = distances <= threshold
                
                # Calculate metrics
                accuracy = accuracy_score(labels, predictions)
                f1 = f1_score(labels, predictions, zero_division=0)
                
                performance.append({
                    'algorithm': algo,
                    'metric': metric,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'threshold': threshold,
                    'avg_distance': np.mean(distances),
                    'std_distance': np.std(distances),
                    'n_samples': len(subset)
                })
        
        return pd.DataFrame(performance).sort_values('f1_score', ascending=False)
    
    def save_results(self, output_file="face_test_results.csv"):
        """Save results to CSV"""
        perf_df = self.analyze_results()
        if len(perf_df) > 0:
            perf_df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        return perf_df


def main():
    """Main function - Python 3.11+ compatible workflow"""
    
    # Configuration
    folder_path = "./images/ronnychieng/"  # Update this path
    max_pairs = 20  # Limit for quick testing
    
    print("=== Python 3.11+ Face Recognition Tester ===")
    print(f"Python version: {os.sys.version}")
    print(f"Testing folder: {folder_path}")
    
    # Check folder
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        print("Please update the folder_path variable")
        return
    
    # Initialize and run tests
    print("Initializing tester...")
    tester = SimpleFaceTester()
    
    print(f"Algorithms: {list(tester.algorithms.keys())}")
    print(f"Metrics: {list(tester.metrics.keys())}")
    print(f"Total combinations: {len(tester.algorithms) * len(tester.metrics)}")
    
    # Run tests
    tester.test_folder(folder_path, max_pairs)
    
    # Analyze results
    print("\n=== Results ===")
    results_df = tester.save_results()
    
    if len(results_df) > 0:
        print("\nTop 5 combinations by F1-score:")
        print(results_df.head()[['algorithm', 'metric', 'f1_score', 'accuracy']].round(3))
        
        print(f"\nBest overall: {results_df.iloc[0]['algorithm']} + {results_df.iloc[0]['metric']}")
        print(f"F1-score: {results_df.iloc[0]['f1_score']:.3f}")
        print(f"Accuracy: {results_df.iloc[0]['accuracy']:.3f}")
        
        # Algorithm rankings
        algo_scores = results_df.groupby('algorithm')['f1_score'].mean().sort_values(ascending=False)
        print(f"\nAlgorithm rankings:")
        for algo, score in algo_scores.items():
            print(f"  {algo}: {score:.3f}")
        
        # Metric rankings  
        metric_scores = results_df.groupby('metric')['f1_score'].mean().sort_values(ascending=False)
        print(f"\nMetric rankings:")
        for metric, score in metric_scores.items():
            print(f"  {metric}: {score:.3f}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()