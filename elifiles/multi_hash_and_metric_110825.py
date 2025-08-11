#!/usr/bin/env python3
"""
Simplified Face Recognition Testing Framework
Concise version with core functionality only
"""

import os
import glob
import tempfile
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
    """Simplified face recognition tester with essential algorithms and metrics"""
    
    def __init__(self):
        # Core algorithms (most scientifically validated)
        self.algorithms = {
            'aHash': lambda img: str(imagehash.average_hash(img, 8)),
            'pHash': lambda img: str(imagehash.phash(img, 8)),        # Most cited
            'dHash': lambda img: str(imagehash.dhash(img, 8)),        # Most efficient
            'wHash': lambda img: str(imagehash.whash(img, 8)),        # Frequency domain
        }
        
        # Core distance metrics
        self.metrics = {
            'hamming': self._hamming_dist,      # Standard for hashing
            'euclidean': self._euclidean_dist,  # Geometric
            'manhattan': self._manhattan_dist,  # Robust to outliers
            'cosine': self._cosine_dist,        # High-dimensional
        }
        
        self.detector = MTCNN()
        self.results = []
    
    def _hamming_dist(self, h1, h2):
        """Hamming distance (normalized) - use ImageHash built-in"""
        hash1 = imagehash.hex_to_hash(h1)
        hash2 = imagehash.hex_to_hash(h2)
        return (hash1 - hash2) / 64  # Normalized by 64 bits (8x8)
    
    def _euclidean_dist(self, h1, h2):
        """Euclidean distance using hash difference"""
        hash1 = imagehash.hex_to_hash(h1)
        hash2 = imagehash.hex_to_hash(h2)
        hamming_dist = hash1 - hash2
        return hamming_dist / 64  # Normalize by max possible distance
    
    def _manhattan_dist(self, h1, h2):
        """Manhattan distance - same as hamming for binary data"""
        return self._hamming_dist(h1, h2)
    
    def _cosine_dist(self, h1, h2):
        """Cosine distance using bit representation"""
        try:
            # Convert to bit arrays using string representation
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
        except:
            # Fallback to hamming distance
            return self._hamming_dist(h1, h2)
    
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
        return {name: func(face_img) for name, func in self.algorithms.items()}
    
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
            for metric_name, metric_func in self.metrics.items():
                try:
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
    """Main function - simplified workflow"""
    
    # Configuration
    folder_path = "./elifiles/images/ronnychieng/"  # Update this path
    max_pairs = 20  # Limit for quick testing
    
    print("=== Simplified Face Recognition Tester ===")
    print(f"Testing folder: {folder_path}")
    
    # Check folder
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        print("Please update the folder_path variable")
        return
    
    # Initialize and run tests
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