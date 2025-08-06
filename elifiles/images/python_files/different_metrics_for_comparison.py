import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import pearsonr
import pandas as pd
from sklearn.metrics import roc_curve, auc
import hashlib

class DistanceMetricsComparator:
    """Compare different distance metrics for face comparison"""
    
    def __init__(self):
        self.distance_functions = {
            'hamming': self.hamming_distance,
            'cosine': self.cosine_distance,
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'jaccard': self.jaccard_distance,
            'normalized_hamming': self.normalized_hamming_distance,
            'weighted_hamming': self.weighted_hamming_distance
        }
    
    def hamming_distance(self, hash1, hash2):
        """Traditional Hamming distance (bit differences)"""
        if isinstance(hash1, str):
            hash1 = int(hash1, 16)
        if isinstance(hash2, str):
            hash2 = int(hash2, 16)
        
        xor_result = hash1 ^ hash2
        return bin(xor_result).count('1')
    
    def normalized_hamming_distance(self, hash1, hash2):
        """Hamming distance normalized by hash length"""
        hamming_dist = self.hamming_distance(hash1, hash2)
        if isinstance(hash1, str):
            total_bits = len(hash1) * 4  # 4 bits per hex char
        else:
            total_bits = hash1.bit_length()
        return hamming_dist / total_bits
    
    def weighted_hamming_distance(self, hash1, hash2):
        """Hamming distance with position-based weighting"""
        if isinstance(hash1, str):
            hash1 = int(hash1, 16)
        if isinstance(hash2, str):
            hash2 = int(hash2, 16)
        
        xor_result = hash1 ^ hash2
        distance = 0
        position = 0
        
        while xor_result > 0:
            if xor_result & 1:
                # Weight early bits more heavily (face structure)
                weight = 1.0 + (1.0 / (position + 1))
                distance += weight
            xor_result >>= 1
            position += 1
        
        return distance
    
    def cosine_distance(self, vec1, vec2):
        """Cosine distance between feature vectors"""
        return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def euclidean_distance(self, vec1, vec2):
        """Euclidean distance between feature vectors"""
        return np.sqrt(np.sum((vec1 - vec2) ** 2))
    
    def manhattan_distance(self, vec1, vec2):
        """Manhattan (L1) distance between feature vectors"""
        return np.sum(np.abs(vec1 - vec2))
    
    def jaccard_distance(self, hash1, hash2):
        """Jaccard distance for binary features"""
        if isinstance(hash1, str):
            hash1 = int(hash1, 16)
        if isinstance(hash2, str):
            hash2 = int(hash2, 16)
        
        # Convert to binary arrays
        max_bits = max(hash1.bit_length(), hash2.bit_length())
        bin1 = np.array([int(b) for b in format(hash1, f'0{max_bits}b')])
        bin2 = np.array([int(b) for b in format(hash2, f'0{max_bits}b')])
        
        intersection = np.sum(bin1 & bin2)
        union = np.sum(bin1 | bin2)
        
        return 1 - (intersection / union) if union > 0 else 0
    
    def hash_to_vector(self, hash_string):
        """Convert hash string to feature vector for vector-based metrics"""
        if isinstance(hash_string, str):
            # Convert hex string to binary array
            hash_int = int(hash_string, 16)
            binary_str = format(hash_int, f'0{len(hash_string)*4}b')
            return np.array([int(b) for b in binary_str], dtype=float)
        return hash_string
    
    def compare_distance_metrics(self, same_person_hashes, different_person_hashes):
        """Compare all distance metrics on the same dataset"""
        
        results = {}
        
        for metric_name, metric_func in self.distance_functions.items():
            print(f"Evaluating {metric_name}...")
            
            same_distances = []
            different_distances = []
            
            # Calculate distances for same person pairs
            for hash1, hash2 in same_person_hashes:
                try:
                    if metric_name in ['cosine', 'euclidean', 'manhattan']:
                        vec1 = self.hash_to_vector(hash1)
                        vec2 = self.hash_to_vector(hash2)
                        distance = metric_func(vec1, vec2)
                    else:
                        distance = metric_func(hash1, hash2)
                    same_distances.append(distance)
                except Exception as e:
                    print(f"Error with {metric_name}: {e}")
                    continue
            
            # Calculate distances for different person pairs
            for hash1, hash2 in different_person_hashes:
                try:
                    if metric_name in ['cosine', 'euclidean', 'manhattan']:
                        vec1 = self.hash_to_vector(hash1)
                        vec2 = self.hash_to_vector(hash2)
                        distance = metric_func(vec1, vec2)
                    else:
                        distance = metric_func(hash1, hash2)
                    different_distances.append(distance)
                except Exception as e:
                    continue
            
            if same_distances and different_distances:
                # Calculate separation metrics
                same_mean = np.mean(same_distances)
                diff_mean = np.mean(different_distances)
                separation = abs(diff_mean - same_mean)
                
                # Calculate overlap (how much distributions overlap)
                same_max = max(same_distances)
                diff_min = min(different_distances)
                overlap = max(0, same_max - diff_min) if diff_mean > same_mean else max(0, max(different_distances) - min(same_distances))
                
                # Calculate ROC AUC (treating distance as negative similarity)
                y_true = [0] * len(same_distances) + [1] * len(different_distances)
                y_scores = same_distances + different_distances
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    roc_auc = auc(fpr, tpr)
                except:
                    roc_auc = 0.5
                
                results[metric_name] = {
                    'same_distances': same_distances,
                    'different_distances': different_distances,
                    'same_mean': same_mean,
                    'same_std': np.std(same_distances),
                    'diff_mean': diff_mean,
                    'diff_std': np.std(different_distances),
                    'separation': separation,
                    'overlap': overlap,
                    'roc_auc': roc_auc
                }
        
        return results
    
    def visualize_distance_comparison(self, results, save_path='distance_metrics_comparison.png'):
        """Create comprehensive visualization of distance metrics"""
        
        n_metrics = len(results)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Distance Metrics Comparison for Face Recognition', fontsize=16)
        
        axes = axes.flatten()
        
        # Plot 1-6: Distribution plots for each metric
        for i, (metric_name, data) in enumerate(results.items()):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # Plot histograms
            ax.hist(data['same_distances'], bins=30, alpha=0.7, 
                   label='Same Person', color='green', density=True)
            ax.hist(data['different_distances'], bins=30, alpha=0.7, 
                   label='Different Person', color='red', density=True)
            
            # Add mean lines
            ax.axvline(data['same_mean'], color='darkgreen', linestyle='--', linewidth=2)
            ax.axvline(data['diff_mean'], color='darkred', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Distance')
            ax.set_ylabel('Density')
            ax.set_title(f'{metric_name.title()}\nSep: {data["separation"]:.3f}, AUC: {data["roc_auc"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 7: Separation comparison
        ax7 = axes[6]
        metric_names = list(results.keys())
        separations = [results[m]['separation'] for m in metric_names]
        
        bars = ax7.bar(range(len(metric_names)), separations, color='skyblue', alpha=0.8)
        ax7.set_xticks(range(len(metric_names)))
        ax7.set_xticklabels(metric_names, rotation=45, ha='right')
        ax7.set_ylabel('Separation (|mean_diff - mean_same|)')
        ax7.set_title('Distance Metric Separation')
        ax7.grid(True, alpha=0.3)
        
        # Highlight best separation
        best_idx = np.argmax(separations)
        bars[best_idx].set_color('gold')
        
        # Plot 8: ROC AUC comparison
        ax8 = axes[7]
        roc_aucs = [results[m]['roc_auc'] for m in metric_names]
        
        bars = ax8.bar(range(len(metric_names)), roc_aucs, color='lightcoral', alpha=0.8)
        ax8.set_xticks(range(len(metric_names)))
        ax8.set_xticklabels(metric_names, rotation=45, ha='right')
        ax8.set_ylabel('ROC AUC')
        ax8.set_title('ROC AUC Comparison (Higher = Better)')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 1)
        
        # Highlight best AUC
        best_auc_idx = np.argmax(roc_aucs)
        bars[best_auc_idx].set_color('gold')
        
        # Plot 9: Summary table
        ax9 = axes[8]
        ax9.axis('off')
        
        # Create summary table
        summary_data = []
        for metric in metric_names:
            data = results[metric]
            summary_data.append([
                metric,
                f"{data['separation']:.3f}",
                f"{data['roc_auc']:.3f}",
                f"{data['same_mean']:.2f}±{data['same_std']:.2f}",
                f"{data['diff_mean']:.2f}±{data['diff_std']:.2f}"
            ])
        
        table = ax9.table(cellText=summary_data,
                         colLabels=['Metric', 'Separation', 'ROC AUC', 'Same (μ±σ)', 'Diff (μ±σ)'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax9.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Distance metrics comparison saved to: {save_path}")
    
    def recommend_best_metric(self, results):
        """Recommend the best distance metric based on multiple criteria"""
        
        print("\n" + "="*60)
        print("DISTANCE METRIC RECOMMENDATION")
        print("="*60)
        
        # Score each metric
        scores = {}
        
        for metric_name, data in results.items():
            score = 0
            
            # Separation score (higher = better)
            separation_score = data['separation'] * 10
            
            # ROC AUC score (higher = better)
            auc_score = data['roc_auc'] * 10
            
            # Overlap penalty (lower overlap = better)
            overlap_penalty = -data['overlap'] * 5
            
            # Stability score (lower std = better)
            stability_score = 5 / (1 + data['same_std'] + data['diff_std'])
            
            total_score = separation_score + auc_score + overlap_penalty + stability_score
            scores[metric_name] = {
                'total': total_score,
                'separation': separation_score,
                'auc': auc_score,
                'overlap_penalty': overlap_penalty,
                'stability': stability_score
            }
        
        # Sort by total score
        ranked_metrics = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print("RANKING (Higher score = Better):")
        print("-" * 60)
        for i, (metric, score_breakdown) in enumerate(ranked_metrics, 1):
            print(f"{i}. {metric.upper()}")
            print(f"   Total Score: {score_breakdown['total']:.2f}")
            print(f"   - Separation: {score_breakdown['separation']:.2f}")
            print(f"   - ROC AUC: {score_breakdown['auc']:.2f}")
            print(f"   - Overlap Penalty: {score_breakdown['overlap_penalty']:.2f}")
            print(f"   - Stability: {score_breakdown['stability']:.2f}")
            
            # Add interpretation
            data = results[metric]
            print(f"   Stats: Same={data['same_mean']:.2f}±{data['same_std']:.2f}, "
                  f"Diff={data['diff_mean']:.2f}±{data['diff_std']:.2f}")
            print()
        
        best_metric = ranked_metrics[0][0]
        print(f"🏆 RECOMMENDED: {best_metric.upper()}")