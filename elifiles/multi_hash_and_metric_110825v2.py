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

    # --- helpers ---
    @staticmethod
    def _bits_from_hex(h: str) -> np.ndarray:
        """Convert a hex string to a fixed-length 0/1 numpy array (4 bits per hex)."""
        if not h:
            return np.zeros(0, dtype=np.uint8)
        b = bytes.fromhex(h)
        bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
        need = 4 * len(h)  # each hex digit = 4 bits
        if bits.size < need:
            bits = np.concatenate([np.zeros(need - bits.size, dtype=np.uint8), bits])
        return bits.astype(np.uint8)

    def __init__(self):
        self.detector = MTCNN()
        self.results = []
        self._setup_algorithms_and_metrics()

    def _setup_algorithms_and_metrics(self):
        """Setup algorithms and metrics after all methods are defined"""
        self.algorithms = {
            'aHash': lambda img: str(imagehash.average_hash(img, 8)),
            'pHash': lambda img: str(imagehash.phash(img, 8)),
            'dHash': lambda img: str(imagehash.dhash(img, 8)),
            'wHash': lambda img: str(imagehash.whash(img, 8))
            # 'dinohash': self.compute_dinohash,  # returns hex string (binarized)
        }
        self._test_dinohash_availability()

        self.metrics = {
            'hamming': self.hamming_dist,        # normalized [0,1]
            'euclidean': self.euclidean_dist,    # normalized [0,1]
            'cosine': self.cosine_dist,          # normalized [0,1]
            'jaccard': self.jaccard_dist,        # normalized [0,1]
            # 'chebyshev': self.chebyshev_dist,    # 0 or 1
        }

    def hamming_dist(self, h1: str, h2: str) -> float:
        try:
            x = self._bits_from_hex(h1); y = self._bits_from_hex(h2)
            x, y = self._align_bits(x, y)
            if x.size == 0: return 1.0
            return float(np.mean(x != y))  # normalized [0,1]
        except Exception as e:
            print(f"Hamming distance error: {e}")
            return 1.0

    def euclidean_dist(self, h1: str, h2: str) -> float:
        try:
            x = self._bits_from_hex(h1).astype(np.float64)
            y = self._bits_from_hex(h2).astype(np.float64)
            x, y = self._align_bits(x, y)
            if x.size == 0: return 1.0
            return float(np.linalg.norm(x - y) / np.sqrt(x.size))  # [0,1]
        except Exception as e:
            print(f"Euclidean distance error: {e}")
            return 1.0

    def chebyshev_dist(self, h1: str, h2: str, _algo_name: str | None = None) -> float:
        try:
            x = self._bits_from_hex(h1); y = self._bits_from_hex(h2)
            x, y = self._align_bits(x, y)
            if x.size == 0: return 1.0
            return float(np.max(np.abs(x - y)))  # 0 or 1
        except Exception as e:
            print(f"Chebyshev distance error: {e}")
            return 1.0

    def cosine_dist(self, h1: str, h2: str, _algo_name: str | None = None) -> float:
        try:
            x = self._bits_from_hex(h1).astype(np.float32)
            y = self._bits_from_hex(h2).astype(np.float32)
            x, y = self._align_bits(x, y)
            if x.size == 0: return 1.0
            n1 = np.linalg.norm(x); n2 = np.linalg.norm(y)
            if n1 == 0.0 or n2 == 0.0: return 1.0
            cos_sim = float(np.dot(x, y) / (n1 * n2))
            cos_sim = max(0.0, min(1.0, cos_sim))
            return 1.0 - cos_sim
        except Exception as e:
            print(f"Cosine distance error: {e}")
            return 1.0

    def jaccard_dist(self, h1: str, h2: str, _algo_name: str | None = None) -> float:
        try:
            x = self._bits_from_hex(h1); y = self._bits_from_hex(h2)
            x, y = self._align_bits(x, y)
            if x.size == 0: return 1.0
            A = set(np.where(x == 1)[0]); B = set(np.where(y == 1)[0])
            U = len(A | B)
            return 0.0 if U == 0 else 1.0 - (len(A & B) / U)
        except Exception as e:
            print(f"Jaccard distance error: {e}")
            return 1.0

    
    def _test_dinohash_availability(self):
        """Test if DinoHash is working using your proven method"""
        try:
            print("🧪 Testing DinoHash availability...")
            
            # Create a small test image
            test_img = Image.new('RGB', (64, 64), color='white')
            result = self.compute_dinohash(test_img)
            
            if result is not None:
                print("✅ DinoHash is available and working!")
                self.algorithms['dinohash'] = self.compute_dinohash
                return True
            else:
                print("⚠️  DinoHash test failed - using traditional algorithms only")
                return False
                
        except Exception as e:
            print(f"⚠️  DinoHash test error: {str(e)[:50]}... - continuing without it")
            return False
    
    def _align_bits(self, x, y):
        if x.size == y.size:
            return x, y
        n = max(x.size, y.size)
        if x.size < n:
            x = np.concatenate([np.zeros(n - x.size, dtype=np.uint8), x])
        if y.size < n:
            y = np.concatenate([np.zeros(n - y.size, dtype=np.uint8), y])
        return x, y

    def compute_dinohash(self, img):
        try:
            # ensure 3-channel for Dino
            if img.mode != 'RGB':
                img = img.convert('RGB')

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                img.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            try:
                result = subprocess.run(
                    ['python3', 'hashes/dinohash.py', tmp_path],
                    capture_output=True, text=True,
                    cwd='/home/ssm-user/dinohash-perceptual-hash',
                    timeout=30
                )
                if result.returncode != 0:
                    print(f"  ⚠️  DinoHash error: {result.stderr}")
                    return None

                out = result.stdout.strip()
                # extract hex after optional "0x" and label
                m = re.search(r'0x([0-9a-fA-F]+)', out)
                if not m:
                    m = re.search(r'([0-9a-fA-F]{8,})$', out)
                if not m:
                    print(f"  ⚠️  Could not parse DinoHash from: {out[:80]}")
                    return None
                return m.group(1).lower()
            finally:
                try: os.unlink(tmp_path)
                except: pass
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
                if name == 'dinohash':
                    print(f"    Computing {name}...", end=" ")
                out = func(face_img)
                if out is not None:
                    hashes[name] = out
                    if name == 'dinohash': print("✓")
                else:
                    if name == 'dinohash': print("✗")
                    print(f"    ⚠️  {name} returned None")
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
        """Summarize distances for positives-only (no F1/accuracy)."""
        if not self.results:
            print("No results to analyze")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        rows = []
        for algo in df['algorithm'].unique():
            for metric in df['metric'].unique():
                sub = df[(df['algorithm'] == algo) & (df['metric'] == metric)]
                if len(sub) < 3:
                    continue
                d = sub['distance'].values
                rows.append({
                    'algorithm': algo,
                    'metric': metric,
                    'avg_distance': float(np.mean(d)),
                    'std_distance': float(np.std(d)),
                    'median_distance': float(np.median(d)),
                    'p25': float(np.percentile(d, 25)),
                    'p75': float(np.percentile(d, 75)),
                    'min': float(np.min(d)),
                    'max': float(np.max(d)),
                    'n_samples': int(len(sub)),
                })
        if not rows:
            return pd.DataFrame()
        # Sort ascending (lower distance = better for same-person pairs)
        return pd.DataFrame(rows).sort_values(['avg_distance', 'std_distance'], ascending=[True, True])

    
    def save_results(self, output_file="face_test_results.csv"):
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