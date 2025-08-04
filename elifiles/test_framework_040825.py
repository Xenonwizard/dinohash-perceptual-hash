import subprocess
import os
import cv2
from mtcnn import MTCNN
import tempfile
import numpy as np
from PIL import Image
import math
import csv
from datetime import datetime
import glob
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

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

def extract_hex_hash(hash_string):
    """Extract hex hash from dinohash output string"""
    if "0x" in hash_string:
        return hash_string.split("0x")[1].strip()
    return hash_string.strip()

def align_face_with_landmarks(image, keypoints):
    """Align face using eye landmarks to make eyes horizontal"""
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    
    # Calculate the angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = math.degrees(math.atan2(dy, dx))
    
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
    """Normalize face crop to standard size and enhance contrast"""
    resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return normalized

def extract_and_align_face(image_path, output_path):
    """Extract face, align it using landmarks, and save it"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Suppress MTCNN progress bars
        import sys
        from contextlib import redirect_stdout
        import io
        
        with redirect_stdout(io.StringIO()):
            detector = MTCNN()
            faces = detector.detect_faces(img_rgb)
        
        if not faces:
            return False
        
        best_face = max(faces, key=lambda x: x['confidence'])
        
        if 'keypoints' not in best_face:
            return False
        
        keypoints = best_face['keypoints']
        
        if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
            return False
        
        # Extract face bounding box with padding
        x, y, w, h = best_face['box']
        padding = max(w, h) // 8
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_rgb.shape[1] - x, w + 2*padding)
        h = min(img_rgb.shape[0] - y, h + 2*padding)
        
        face_region = img_rgb[y:y+h, x:x+w]
        
        # Adjust keypoints relative to cropped face
        adjusted_keypoints = {}
        for key, point in keypoints.items():
            adjusted_keypoints[key] = (point[0] - x, point[1] - y)
        
        # Align and normalize the face
        aligned_face, rotation_angle = align_face_with_landmarks(face_region, adjusted_keypoints)
        normalized_face = normalize_face_crop(aligned_face, target_size=(200, 200))
        face_bgr = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, face_bgr)
        return True
        
    except Exception as e:
        print(f"Error extracting face from {image_path}: {e}")
        return False

def calculate_similarity(hash1, hash2):
    """Calculate similarity between two hashes"""
    try:
        hash1 = extract_hex_hash(hash1)
        hash2 = extract_hex_hash(hash2)
        
        hash1_int = int(hash1, 16)
        hash2_int = int(hash2, 16)
        hamming_distance = bin(hash1_int ^ hash2_int).count('1')
        total_bits = len(hash1) * 4
        similarity = 1 - (hamming_distance / total_bits)
        
        return similarity, hamming_distance, total_bits
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0, -1, -1

def compare_faces_detailed(img1_path, img2_path):
    """Compare faces and return detailed metrics"""
    with tempfile.NamedTemporaryFile(suffix='_aligned_face1.jpg', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='_aligned_face2.jpg', delete=False) as tmp2:
        
        face1_path = tmp1.name
        face2_path = tmp2.name
    
    try:
        face1_extracted = extract_and_align_face(img1_path, face1_path)
        face2_extracted = extract_and_align_face(img2_path, face2_path)
        
        if not face1_extracted or not face2_extracted:
            return None
        
        hash1 = get_dinohash(face1_path)
        hash2 = get_dinohash(face2_path)
        
        if hash1 and hash2:
            similarity, hamming_distance, total_bits = calculate_similarity(hash1, hash2)
            
            return {
                'similarity': similarity,
                'hamming_distance': hamming_distance,
                'total_bits': total_bits,
                'hash1': hash1,
                'hash2': hash2
            }
        
        return None
            
    finally:
        try:
            os.unlink(face1_path)
            os.unlink(face2_path)
        except:
            pass

class FaceComparisonValidator:
    def __init__(self, dataset_path):
        """
        Initialize validator with dataset path
        
        Args:
            dataset_path (str): Path to the celeb dataset (e.g., './DEEPFACE/celeb-dataset')
        """
        self.dataset_path = dataset_path
        self.results = []
        self.person_folders = []
        self.load_dataset_structure()
    
    def load_dataset_structure(self):
        """Load the dataset structure and identify person folders"""
        # Look for person folders in each ethnicity category
        ethnicity_folders = ['caucasian', 'chinese', 'indian', 'malay']
        
        for ethnicity in ethnicity_folders:
            ethnicity_path = os.path.join(self.dataset_path, ethnicity)
            if os.path.exists(ethnicity_path):
                # Get all person folders in this ethnicity
                person_dirs = [d for d in os.listdir(ethnicity_path) 
                              if os.path.isdir(os.path.join(ethnicity_path, d))]
                
                for person_dir in person_dirs:
                    person_path = os.path.join(ethnicity_path, person_dir)
                    images = glob.glob(os.path.join(person_path, "*.jpg")) + \
                            glob.glob(os.path.join(person_path, "*.jpeg")) + \
                            glob.glob(os.path.join(person_path, "*.png"))
                    
                    if len(images) >= 2:  # Need at least 2 images for comparison
                        self.person_folders.append({
                            'person_id': person_dir,
                            'ethnicity': ethnicity,
                            'images': images,
                            'image_count': len(images)
                        })
        
        print(f"Found {len(self.person_folders)} people with 2+ images")
        for person in self.person_folders[:5]:  # Show first 5
            print(f"  {person['person_id']} ({person['ethnicity']}): {person['image_count']} images")
    
    def test_same_person_accuracy(self, max_comparisons_per_person=10):
        """Test accuracy on same-person comparisons"""
        print("Testing same-person comparisons...")
        same_person_results = []
        
        for person in self.person_folders:
            images = person['images']
            comparisons_made = 0
            
            # Compare pairs of images from the same person
            for img1, img2 in combinations(images, 2):
                if comparisons_made >= max_comparisons_per_person:
                    break
                
                result = compare_faces_detailed(img1, img2)
                if result:
                    result.update({
                        'person_id': person['person_id'],
                        'ethnicity': person['ethnicity'],
                        'comparison_type': 'same_person',
                        'image1': os.path.basename(img1),
                        'image2': os.path.basename(img2),
                        'timestamp': datetime.now().isoformat()
                    })
                    same_person_results.append(result)
                    comparisons_made += 1
        
        self.results.extend(same_person_results)
        print(f"Completed {len(same_person_results)} same-person comparisons")
        return same_person_results
    
    def test_different_person_accuracy(self, max_comparisons=200):
        """Test accuracy on different-person comparisons"""
        print("Testing different-person comparisons...")
        different_person_results = []
        comparisons_made = 0
        
        # Compare people from different folders
        for person1, person2 in combinations(self.person_folders, 2):
            if comparisons_made >= max_comparisons:
                break
            
            # Take one image from each person
            img1 = person1['images'][0]
            img2 = person2['images'][0]
            
            result = compare_faces_detailed(img1, img2)
            if result:
                result.update({
                    'person1_id': person1['person_id'],
                    'person2_id': person2['person_id'],
                    'ethnicity1': person1['ethnicity'],
                    'ethnicity2': person2['ethnicity'],
                    'comparison_type': 'different_person',
                    'image1': os.path.basename(img1),
                    'image2': os.path.basename(img2),
                    'timestamp': datetime.now().isoformat()
                })
                different_person_results.append(result)
                comparisons_made += 1
        
        self.results.extend(different_person_results)
        print(f"Completed {len(different_person_results)} different-person comparisons")
        return different_person_results
    
    def analyze_threshold_performance(self, thresholds=None):
        """Analyze performance at different similarity thresholds"""
        if thresholds is None:
            thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        
        same_person_similarities = [r['similarity'] for r in self.results if r['comparison_type'] == 'same_person']
        different_person_similarities = [r['similarity'] for r in self.results if r['comparison_type'] == 'different_person']
        
        performance_data = []
        
        for threshold in thresholds:
            # Calculate metrics for same person (should be identified as same)
            same_correct = sum(1 for sim in same_person_similarities if sim >= threshold)
            same_total = len(same_person_similarities)
            same_accuracy = same_correct / same_total if same_total > 0 else 0
            
            # Calculate metrics for different person (should be identified as different)
            diff_correct = sum(1 for sim in different_person_similarities if sim < threshold)
            diff_total = len(different_person_similarities)
            diff_accuracy = diff_correct / diff_total if diff_total > 0 else 0
            
            # Overall accuracy
            total_correct = same_correct + diff_correct
            total_comparisons = same_total + diff_total
            overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0
            
            # False positives and negatives
            false_positives = diff_total - diff_correct  # Different people identified as same
            false_negatives = same_total - same_correct  # Same people identified as different
            
            performance_data.append({
                'threshold': threshold,
                'same_person_accuracy': same_accuracy,
                'different_person_accuracy': diff_accuracy,
                'overall_accuracy': overall_accuracy,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'same_person_count': same_total,
                'different_person_count': diff_total
            })
        
        return performance_data
    
    def generate_report(self, output_file='face_validation_report.json'):
        """Generate comprehensive validation report"""
        if not self.results:
            print("No results to analyze. Run tests first.")
            return
        
        # Basic statistics
        same_person_results = [r for r in self.results if r['comparison_type'] == 'same_person']
        different_person_results = [r for r in self.results if r['comparison_type'] == 'different_person']
        
        same_similarities = [r['similarity'] for r in same_person_results]
        diff_similarities = [r['similarity'] for r in different_person_results]
        
        # Threshold analysis
        performance_data = self.analyze_threshold_performance()
        
        # Find optimal threshold (highest overall accuracy)
        best_threshold = max(performance_data, key=lambda x: x['overall_accuracy'])
        
        report = {
            'dataset_info': {
                'total_people': len(self.person_folders),
                'total_comparisons': len(self.results),
                'same_person_comparisons': len(same_person_results),
                'different_person_comparisons': len(different_person_results)
            },
            'similarity_statistics': {
                'same_person': {
                    'mean': np.mean(same_similarities) if same_similarities else 0,
                    'std': np.std(same_similarities) if same_similarities else 0,
                    'min': min(same_similarities) if same_similarities else 0,
                    'max': max(same_similarities) if same_similarities else 0,
                    'median': np.median(same_similarities) if same_similarities else 0
                },
                'different_person': {
                    'mean': np.mean(diff_similarities) if diff_similarities else 0,
                    'std': np.std(diff_similarities) if diff_similarities else 0,
                    'min': min(diff_similarities) if diff_similarities else 0,
                    'max': max(diff_similarities) if diff_similarities else 0,
                    'median': np.median(diff_similarities) if diff_similarities else 0
                }
            },
            'optimal_threshold': best_threshold,
            'threshold_analysis': performance_data,
            'recommendations': self.generate_recommendations(performance_data, same_similarities, diff_similarities)
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_file}")
        return report
    
    def generate_recommendations(self, performance_data, same_similarities, diff_similarities):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check if there's good separation between same/different person similarities
        same_mean = np.mean(same_similarities) if same_similarities else 0
        diff_mean = np.mean(diff_similarities) if diff_similarities else 0
        separation = same_mean - diff_mean
        
        if separation < 0.1:
            recommendations.append("WARNING: Poor separation between same-person and different-person similarities. Consider improving face alignment or using a different hashing method.")
        elif separation > 0.3:
            recommendations.append("GOOD: Strong separation between same-person and different-person similarities.")
        else:
            recommendations.append("MODERATE: Decent separation between similarity distributions.")
        
        # Find best performing threshold
        best_perf = max(performance_data, key=lambda x: x['overall_accuracy'])
        recommendations.append(f"Recommended threshold: {best_perf['threshold']:.2f} (Overall accuracy: {best_perf['overall_accuracy']:.3f})")
        
        # Check for bias in accuracy
        accuracy_diff = abs(best_perf['same_person_accuracy'] - best_perf['different_person_accuracy'])
        if accuracy_diff > 0.2:
            recommendations.append("WARNING: Significant bias detected. The system performs much better on one type of comparison.")
        
        return recommendations
    
    def save_detailed_results(self, filename='detailed_face_comparison_results.csv'):
        """Save all detailed results to CSV"""
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Detailed results saved to: {filename}")
    
    def run_full_validation(self, max_same_person_per_person=5, max_different_person=100):
        """Run complete validation suite"""
        print("Starting full face comparison validation...")
        print(f"Dataset: {self.dataset_path}")
        print(f"People found: {len(self.person_folders)}")
        print("="*60)
        
        # Test same-person accuracy
        self.test_same_person_accuracy(max_same_person_per_person)
        
        # Test different-person accuracy
        self.test_different_person_accuracy(max_different_person)
        
        # Generate comprehensive report
        report = self.generate_report()
        
        # Save detailed results
        self.save_detailed_results()
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total comparisons: {len(self.results)}")
        print(f"Same-person comparisons: {report['dataset_info']['same_person_comparisons']}")
        print(f"Different-person comparisons: {report['dataset_info']['different_person_comparisons']}")
        print(f"\nOptimal threshold: {report['optimal_threshold']['threshold']}")
        print(f"Overall accuracy at optimal threshold: {report['optimal_threshold']['overall_accuracy']:.3f}")
        print(f"Same-person accuracy: {report['optimal_threshold']['same_person_accuracy']:.3f}")
        print(f"Different-person accuracy: {report['optimal_threshold']['different_person_accuracy']:.3f}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        return report

# Usage example
if __name__ == "__main__":
    # Initialize validator with your dataset path
    validator = FaceComparisonValidator('./images/celeb-dataset')
    
    # Run full validation
    report = validator.run_full_validation(
        max_same_person_per_person=3,  # Max comparisons per person
        max_different_person=50        # Max different-person comparisons
    )