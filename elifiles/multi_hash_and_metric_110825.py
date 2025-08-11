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
import pandas as pd
from datetime import datetime
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.spatial.distance import euclidean, manhattan, cosine, chebyshev
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedFaceHashTester:
    """
    Advanced face recognition testing framework with multiple perceptual hashing algorithms
    and comprehensive distance metrics for scientific evaluation.
    """
    
    def __init__(self):
        self.algorithms = {
            'aHash_8': {'func': self.compute_ahash, 'hash_size': 8, 'type': 'traditional'},
            'aHash_16': {'func': self.compute_ahash, 'hash_size': 16, 'type': 'traditional'},
            'pHash_8': {'func': self.compute_phash, 'hash_size': 8, 'type': 'traditional'},
            'pHash_16': {'func': self.compute_phash, 'hash_size': 16, 'type': 'traditional'},
            'dHash_8': {'func': self.compute_dhash, 'hash_size': 8, 'type': 'traditional'},
            'dHash_16': {'func': self.compute_dhash, 'hash_size': 16, 'type': 'traditional'},
            'wHash_8': {'func': self.compute_whash, 'hash_size': 8, 'type': 'traditional'},
            'wHash_16': {'func': self.compute_whash, 'hash_size': 16, 'type': 'traditional'},
            'colorhash': {'func': self.compute_colorhash, 'hash_size': 8, 'type': 'traditional'},
            'crop_resistant': {'func': self.compute_crop_resistant, 'hash_size': 8, 'type': 'traditional'},
            'dinohash': {'func': self.compute_dinohash, 'hash_size': None, 'type': 'deep_learning'}
        }
        
        self.distance_metrics = {
            'hamming': self.hamming_distance,
            'euclidean': self.euclidean_distance, 
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_distance,
            'chebyshev': self.chebyshev_distance,
            'jaccard': self.jaccard_distance,
            'ssim': self.ssim_distance
        }
        
        self.results = []
        self.detector = MTCNN()
    
    def compute_ahash(self, image_path, hash_size=8):
        """Average Hash - based on mean pixel values"""
        try:
            img = Image.open(image_path).convert('L')
            hash_val = imagehash.average_hash(img, hash_size=hash_size)
            return str(hash_val)
        except Exception as e:
            print(f"Error computing aHash: {e}")
            return None
    
    def compute_phash(self, image_path, hash_size=8):
        """Perceptual Hash - DCT based, most scientifically validated"""
        try:
            img = Image.open(image_path).convert('L')
            hash_val = imagehash.phash(img, hash_size=hash_size)
            return str(hash_val)
        except Exception as e:
            print(f"Error computing pHash: {e}")
            return None
    
    def compute_dhash(self, image_path, hash_size=8):
        """Difference Hash - gradient based, highly efficient"""
        try:
            img = Image.open(image_path).convert('L')
            hash_val = imagehash.dhash(img, hash_size=hash_size)
            return str(hash_val)
        except Exception as e:
            print(f"Error computing dHash: {e}")
            return None
    
    def compute_whash(self, image_path, hash_size=8):
        """Wavelet Hash - uses DWT, frequency domain analysis"""
        try:
            img = Image.open(image_path).convert('L')
            hash_val = imagehash.whash(img, hash_size=hash_size)
            return str(hash_val)
        except Exception as e:
            print(f"Error computing wHash: {e}")
            return None
    
    def compute_colorhash(self, image_path, hash_size=8):
        """Color Hash - considers color information"""
        try:
            img = Image.open(image_path)
            hash_val = imagehash.colorhash(img, binbits=3)
            return str(hash_val)
        except Exception as e:
            print(f"Error computing colorhash: {e}")
            return None
    
    def compute_crop_resistant(self, image_path, hash_size=8):
        """Crop-resistant Hash - robust to cropping"""
        try:
            img = Image.open(image_path).convert('L')
            hash_val = imagehash.crop_resistant_hash(img, hash_size=hash_size)
            return str(hash_val)
        except Exception as e:
            print(f"Error computing crop-resistant hash: {e}")
            return None
    
    def compute_dinohash(self, image_path, hash_size=None):
        """DinoHash - deep learning based perceptual hash"""
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
                print(f"DinoHash error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Failed to run dinohash: {e}")
            return None
    
    def hamming_distance(self, hash1, hash2, hash_type):
        """Hamming distance - count of differing bits"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_int = int(hash1, 16)
                h2_int = int(hash2, 16)
                hamming_dist = bin(h1_int ^ h2_int).count('1')
                total_bits = len(hash1) * 4
                return hamming_dist / total_bits  # Normalized
            else:
                h1 = imagehash.hex_to_hash(hash1)
                h2 = imagehash.hex_to_hash(hash2)
                return (h1 - h2) / len(str(h1))  # Normalized
        except:
            return float('inf')
    
    def euclidean_distance(self, hash1, hash2, hash_type):
        """Euclidean distance between hash bit vectors"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_bits = [int(bit) for bit in bin(int(hash1, 16))[2:].zfill(len(hash1)*4)]
                h2_bits = [int(bit) for bit in bin(int(hash2, 16))[2:].zfill(len(hash2)*4)]
            else:
                h1_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash1))]
                h2_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash2))]
            
            return euclidean(h1_bits, h2_bits) / len(h1_bits)**0.5  # Normalized
        except:
            return float('inf')
    
    def manhattan_distance(self, hash1, hash2, hash_type):
        """Manhattan (L1) distance"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_bits = [int(bit) for bit in bin(int(hash1, 16))[2:].zfill(len(hash1)*4)]
                h2_bits = [int(bit) for bit in bin(int(hash2, 16))[2:].zfill(len(hash2)*4)]
            else:
                h1_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash1))]
                h2_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash2))]
            
            return manhattan(h1_bits, h2_bits) / len(h1_bits)  # Normalized
        except:
            return float('inf')
    
    def cosine_distance(self, hash1, hash2, hash_type):
        """Cosine distance - angle between vectors"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_bits = [int(bit) for bit in bin(int(hash1, 16))[2:].zfill(len(hash1)*4)]
                h2_bits = [int(bit) for bit in bin(int(hash2, 16))[2:].zfill(len(hash2)*4)]
            else:
                h1_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash1))]
                h2_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash2))]
            
            return cosine(h1_bits, h2_bits)
        except:
            return float('inf')
    
    def chebyshev_distance(self, hash1, hash2, hash_type):
        """Chebyshev distance - maximum difference in any dimension"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_bits = [int(bit) for bit in bin(int(hash1, 16))[2:].zfill(len(hash1)*4)]
                h2_bits = [int(bit) for bit in bin(int(hash2, 16))[2:].zfill(len(hash2)*4)]
            else:
                h1_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash1))]
                h2_bits = [int(bit) for bit in str(imagehash.hex_to_hash(hash2))]
            
            return chebyshev(h1_bits, h2_bits)
        except:
            return float('inf')
    
    def jaccard_distance(self, hash1, hash2, hash_type):
        """Jaccard distance for binary vectors"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_bits = set([i for i, bit in enumerate(bin(int(hash1, 16))[2:].zfill(len(hash1)*4)) if bit == '1'])
                h2_bits = set([i for i, bit in enumerate(bin(int(hash2, 16))[2:].zfill(len(hash2)*4)) if bit == '1'])
            else:
                h1_str = str(imagehash.hex_to_hash(hash1))
                h2_str = str(imagehash.hex_to_hash(hash2))
                h1_bits = set([i for i, bit in enumerate(h1_str) if bit == '1'])
                h2_bits = set([i for i, bit in enumerate(h2_str) if bit == '1'])
            
            intersection = len(h1_bits & h2_bits)
            union = len(h1_bits | h2_bits)
            
            if union == 0:
                return 0.0
            return 1 - (intersection / union)
        except:
            return float('inf')
    
    def ssim_distance(self, hash1, hash2, hash_type):
        """SSIM-based distance using hash reconstruction"""
        if not hash1 or not hash2:
            return float('inf')
        
        try:
            if hash_type == 'dinohash':
                h1_bits = np.array([int(bit) for bit in bin(int(hash1, 16))[2:].zfill(len(hash1)*4)])
                h2_bits = np.array([int(bit) for bit in bin(int(hash2, 16))[2:].zfill(len(hash2)*4)])
            else:
                h1_bits = np.array([int(bit) for bit in str(imagehash.hex_to_hash(hash1))])
                h2_bits = np.array([int(bit) for bit in str(imagehash.hex_to_hash(hash2))])
            
            # Reshape to square for SSIM
            size = int(len(h1_bits) ** 0.5)
            if size * size != len(h1_bits):
                # Pad or truncate to make square
                target_size = int(np.ceil(len(h1_bits) ** 0.5)) ** 2
                h1_bits = np.pad(h1_bits, (0, target_size - len(h1_bits)), 'constant')
                h2_bits = np.pad(h2_bits, (0, target_size - len(h2_bits)), 'constant')
                size = int(target_size ** 0.5)
            
            h1_2d = h1_bits[:size*size].reshape(size, size)
            h2_2d = h2_bits[:size*size].reshape(size, size)
            
            ssim_val = ssim(h1_2d, h2_2d, data_range=1)
            return 1 - ssim_val  # Convert similarity to distance
        except:
            return float('inf')
    
    def extract_and_align_face(self, image_path, output_path):
        """Extract and align face using MTCNN"""
        try:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
            
            faces = self.detector.detect_faces(img_array)
            
            if not faces:
                return False, "No face detected"
            
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
            
            face_crop = pil_img.crop((x, y, x + w, y + h))
            
            # Align face
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dy, dx))
            
            center_x = (left_eye[0] + right_eye[0]) / 2 - x
            center_y = (left_eye[1] + right_eye[1]) / 2 - y
            
            aligned_face = face_crop.rotate(-angle, expand=False, center=(center_x, center_y))
            
            # Normalize and enhance
            normalized = aligned_face.resize((200, 200), Image.Resampling.LANCZOS)
            enhancer = ImageEnhance.Contrast(normalized)
            enhanced = enhancer.enhance(1.2)
            
            enhanced.save(output_path, 'JPEG', quality=95)
            
            return True, f"Success (confidence: {best_face['confidence']:.3f})"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def compute_all_hashes(self, image_path):
        """Compute all hash types for an image"""
        hashes = {}
        
        for algo_name, algo_info in self.algorithms.items():
            try:
                if algo_info['hash_size']:
                    hash_val = algo_info['func'](image_path, algo_info['hash_size'])
                else:
                    hash_val = algo_info['func'](image_path)
                hashes[algo_name] = hash_val
            except Exception as e:
                print(f"Error computing {algo_name}: {e}")
                hashes[algo_name] = None
        
        return hashes
    
    def test_image_pair(self, img1_path, img2_path, ground_truth_same_person=True):
        """Test a pair of images with all algorithms and metrics"""
        
        # Extract faces
        with tempfile.NamedTemporaryFile(suffix='_face1.jpg', delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix='_face2.jpg', delete=False) as tmp2:
            
            face1_path = tmp1.name
            face2_path = tmp2.name
        
        try:
            success1, msg1 = self.extract_and_align_face(img1_path, face1_path)
            success2, msg2 = self.extract_and_align_face(img2_path, face2_path)
            
            if not (success1 and success2):
                return None  # Skip failed extractions
            
            # Compute hashes
            hashes1 = self.compute_all_hashes(face1_path)
            hashes2 = self.compute_all_hashes(face2_path)
            
            # Test all algorithm-metric combinations
            for algo_name in self.algorithms.keys():
                if hashes1[algo_name] is None or hashes2[algo_name] is None:
                    continue
                
                for metric_name, metric_func in self.distance_metrics.items():
                    try:
                        distance = metric_func(hashes1[algo_name], hashes2[algo_name], algo_name)
                        
                        self.results.append({
                            'algorithm': algo_name,
                            'metric': metric_name,
                            'distance': distance,
                            'ground_truth': ground_truth_same_person,
                            'img1': os.path.basename(img1_path),
                            'img2': os.path.basename(img2_path),
                            'hash1': hashes1[algo_name],
                            'hash2': hashes2[algo_name],
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        print(f"Error with {algo_name}-{metric_name}: {e}")
        
        finally:
            # Cleanup
            try:
                os.unlink(face1_path)
                os.unlink(face2_path)
            except:
                pass
    
    def optimize_thresholds(self, validation_split=0.3):
        """Find optimal thresholds for each algorithm-metric combination"""
        if not self.results:
            print("No results to optimize thresholds")
            return {}
        
        df = pd.DataFrame(self.results)
        optimal_thresholds = {}
        
        for algo in df['algorithm'].unique():
            for metric in df['metric'].unique():
                subset = df[(df['algorithm'] == algo) & (df['metric'] == metric)]
                if len(subset) < 10:  # Skip if too few samples
                    continue
                
                # Split data
                n_val = int(len(subset) * validation_split)
                val_data = subset.sample(n=n_val, random_state=42)
                
                distances = val_data['distance'].values
                labels = val_data['ground_truth'].values
                
                # Try different thresholds
                thresholds = np.linspace(distances.min(), distances.max(), 100)
                best_f1 = 0
                best_threshold = None
                
                for threshold in thresholds:
                    predictions = distances <= threshold
                    f1 = f1_score(labels, predictions)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                optimal_thresholds[f"{algo}_{metric}"] = {
                    'threshold': best_threshold,
                    'f1_score': best_f1
                }
        
        return optimal_thresholds
    
    def evaluate_performance(self, thresholds=None):
        """Evaluate performance of all algorithm-metric combinations"""
        if not self.results:
            print("No results to evaluate")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        performance_results = []
        
        for algo in df['algorithm'].unique():
            for metric in df['metric'].unique():
                subset = df[(df['algorithm'] == algo) & (df['metric'] == metric)]
                if len(subset) < 5:  # Skip if too few samples
                    continue
                
                distances = subset['distance'].values
                labels = subset['ground_truth'].values
                
                # Use provided threshold or median
                key = f"{algo}_{metric}"
                if thresholds and key in thresholds:
                    threshold = thresholds[key]['threshold']
                else:
                    threshold = np.median(distances)
                
                predictions = distances <= threshold
                
                # Calculate metrics
                accuracy = accuracy_score(labels, predictions)
                precision = precision_score(labels, predictions, zero_division=0)
                recall = recall_score(labels, predictions, zero_division=0)
                f1 = f1_score(labels, predictions, zero_division=0)
                
                # Additional statistics
                same_person_distances = distances[labels == True]
                diff_person_distances = distances[labels == False]
                
                performance_results.append({
                    'algorithm': algo,
                    'metric': metric,
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'n_samples': len(subset),
                    'mean_same_person_dist': np.mean(same_person_distances) if len(same_person_distances) > 0 else np.nan,
                    'mean_diff_person_dist': np.mean(diff_person_distances) if len(diff_person_distances) > 0 else np.nan,
                    'std_same_person_dist': np.std(same_person_distances) if len(same_person_distances) > 0 else np.nan,
                    'std_diff_person_dist': np.std(diff_person_distances) if len(diff_person_distances) > 0 else np.nan
                })
        
        return pd.DataFrame(performance_results)
    
    def save_results(self, base_folder="face_recognition_results"):
        """Save results organized by algorithm and metric"""
        if not self.results:
            print("No results to save")
            return
        
        # Create base folder
        os.makedirs(base_folder, exist_ok=True)
        
        # Get performance results
        thresholds = self.optimize_thresholds()
        performance_df = self.evaluate_performance(thresholds)
        
        # Save overall summary
        summary_path = os.path.join(base_folder, "summary_performance.csv")
        performance_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
        
        # Save detailed results for each algorithm-metric combination
        df = pd.DataFrame(self.results)
        
        for algo in df['algorithm'].unique():
            for metric in df['metric'].unique():
                subset = df[(df['algorithm'] == algo) & (df['metric'] == metric)]
                if len(subset) == 0:
                    continue
                
                # Create folder for this combination
                folder_name = f"{algo}_{metric}"
                combo_folder = os.path.join(base_folder, folder_name)
                os.makedirs(combo_folder, exist_ok=True)
                
                # Save detailed results
                detail_path = os.path.join(combo_folder, "detailed_results.csv")
                subset.to_csv(detail_path, index=False)
                
                # Save performance summary for this combination
                perf_subset = performance_df[
                    (performance_df['algorithm'] == algo) & 
                    (performance_df['metric'] == metric)
                ]
                if len(perf_subset) > 0:
                    perf_path = os.path.join(combo_folder, "performance_summary.csv")
                    perf_subset.to_csv(perf_path, index=False)
        
        # Save raw results and thresholds
        raw_path = os.path.join(base_folder, "raw_results.json")
        with open(raw_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        thresh_path = os.path.join(base_folder, "optimal_thresholds.json")
        with open(thresh_path, 'w') as f:
            json.dump(thresholds, f, indent=2, default=str)
        
        print(f"Results saved to: {base_folder}")
        print(f"Found {len(performance_df)} algorithm-metric combinations")
        
        return performance_df
    
    def test_folder(self, folder_path, max_comparisons=None):
        """Test all images in a folder against each other (same person)"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(folder_path, ext)))
        
        print(f"Found {len(images)} images in {folder_path}")
        
        if len(images) < 2:
            print("Need at least 2 images to compare")
            return
        
        # Generate all pairs
        pairs = list(combinations(images, 2))
        if max_comparisons and len(pairs) > max_comparisons:
            pairs = pairs[:max_comparisons]
        
        print(f"Testing {len(pairs)} image pairs (same person)")
        
        for i, (img1, img2) in enumerate(pairs):
            print(f"Processing pair {i+1}/{len(pairs)}: {os.path.basename(img1)} vs {os.path.basename(img2)}")
            self.test_image_pair(img1, img2, ground_truth_same_person=True)
        
        print(f"Completed testing {len(pairs)} pairs")


def main():
    """Main function to run the face recognition testing"""
    
    # Initialize tester
    tester = AdvancedFaceHashTester()
    
    # Test parameters
    ronnychieng_folder = "./elifiles/images/ronnychieng/"
    
    # Check if folder exists
    if not os.path.exists(ronnychieng_folder):
        print(f"❌ Folder not found: {ronnychieng_folder}")
        print("Please update the folder path")
        return
    
    print("=" * 80)
    print("ADVANCED FACE RECOGNITION TESTING FRAMEWORK")
    print("Testing Ronny Chieng images against each other")
    print("=" * 80)
    
    print(f"\nTesting algorithms: {list(tester.algorithms.keys())}")
    print(f"Testing distance metrics: {list(tester.distance_metrics.keys())}")
    
    # Run tests
    tester.test_folder(ronnychieng_folder, max_comparisons=50)  # Limit for testing
    
    # Analyze and save results
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS")
    print("=" * 80)
    
    performance_df = tester.save_results()
    
    if len(performance_df) > 0:
        # Display top performers
        print("\nTOP 10 ALGORITHM-METRIC COMBINATIONS BY F1 SCORE:")
        print("-" * 60)
        top_performers = performance_df.nlargest(10, 'f1_score')[
            ['algorithm', 'metric', 'f1_score', 'accuracy', 'precision', 'recall']
        ]
        print(top_performers.to_string(index=False))
        
        print("\nALGORITHM RANKINGS BY AVERAGE F1 SCORE:")
        print("-" * 60)
        algo_rankings = performance_df.groupby('algorithm')['f1_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        print(algo_rankings.round(4))
        
        print("\nMETRIC RANKINGS BY AVERAGE F1 SCORE:")
        print("-" * 60)
        metric_rankings = performance_df.groupby('metric')['f1_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        print(metric_rankings.round(4))
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("Results saved in 'face_recognition_results' folder")
    print("Each algorithm-metric combination has its own subfolder with detailed results")


if __name__ == "__main__":
    main()