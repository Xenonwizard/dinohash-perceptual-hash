import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from collections import defaultdict

class FaceValidationVisualizer:
    def __init__(self, results_csv='detailed_face_comparison_results.csv', 
                 report_json='face_validation_report.json'):
        """Initialize with results files"""
        self.results_df = pd.read_csv(results_csv) if results_csv else None
        
        with open(report_json, 'r') as f:
            self.report = json.load(f)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_similarity_distributions(self, save_path='similarity_distributions.png'):
        """Plot similarity score distributions for same vs different person comparisons"""
        if self.results_df is None:
            print("No results data available")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        same_person = self.results_df[self.results_df['comparison_type'] == 'same_person']['similarity']
        different_person = self.results_df[self.results_df['comparison_type'] == 'different_person']['similarity']
        
        # Histogram comparison
        ax1.hist(same_person, bins=30, alpha=0.7, label='Same Person', color='green', density=True)
        ax1.hist(different_person, bins=30, alpha=0.7, label='Different Person', color='red', density=True)
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Similarity Score Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_for_box = [same_person, different_person]
        labels = ['Same Person', 'Different Person']
        colors = ['lightgreen', 'lightcoral']
        
        box_plot = ax2.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Similarity Score Distributions (Box Plot)')
        ax2.grid(True, alpha=0.3)
        
        # Violin plot
        ax3.violinplot(data_for_box, positions=[1, 2], showmeans=True, showmedians=True)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['Same Person', 'Different Person'])
        ax3.set_ylabel('Similarity Score')
        ax3.set_title('Similarity Score Distributions (Violin Plot)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Similarity distributions plot saved to: {save_path}")
    
    def plot_threshold_analysis(self, save_path='threshold_analysis.png'):
        """Plot accuracy metrics across different thresholds"""
        threshold_data = pd.DataFrame(self.report['threshold_analysis'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall accuracy
        ax1.plot(threshold_data['threshold'], threshold_data['overall_accuracy'], 
                'bo-', linewidth=2, markersize=8, label='Overall Accuracy')
        ax1.plot(threshold_data['threshold'], threshold_data['same_person_accuracy'], 
                'go-', linewidth=2, markersize=6, alpha=0.7, label='Same Person Accuracy')
        ax1.plot(threshold_data['threshold'], threshold_data['different_person_accuracy'], 
                'ro-', linewidth=2, markersize=6, alpha=0.7, label='Different Person Accuracy')
        
        # Highlight optimal threshold
        optimal_threshold = self.report['optimal_threshold']['threshold']
        optimal_accuracy = self.report['optimal_threshold']['overall_accuracy']
        ax1.axvline(x=optimal_threshold, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax1.scatter([optimal_threshold], [optimal_accuracy], color='purple', s=100, zorder=5)
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # False positives and negatives
        ax2.plot(threshold_data['threshold'], threshold_data['false_positives'], 
                'ro-', linewidth=2, label='False Positives')
        ax2.plot(threshold_data['threshold'], threshold_data['false_negatives'], 
                'bo-', linewidth=2, label='False Negatives')
        
        ax2.axvline(x=optimal_threshold, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Count')
        ax2.set_title('False Positives/Negatives vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ROC-like curve (True Positive Rate vs False Positive Rate)
        tpr = threshold_data['same_person_accuracy']  # True Positive Rate
        fpr = 1 - threshold_data['different_person_accuracy']  # False Positive Rate
        
        ax3.plot(fpr, tpr, 'bo-', linewidth=2, markersize=6)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # Highlight optimal point
        optimal_fpr = 1 - self.report['optimal_threshold']['different_person_accuracy']
        optimal_tpr = self.report['optimal_threshold']['same_person_accuracy']
        ax3.scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5, label='Optimal Threshold')
        
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC-like Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        
        # Threshold vs Total Errors
        total_errors = threshold_data['false_positives'] + threshold_data['false_negatives']
        ax4.plot(threshold_data['threshold'], total_errors, 'mo-', linewidth=2, markersize=6)
        
        # Find and highlight minimum error threshold
        min_error_idx = total_errors.idxmin()
        min_error_threshold = threshold_data.iloc[min_error_idx]['threshold']
        min_error_count = total_errors.iloc[min_error_idx]
        
        ax4.axvline(x=min_error_threshold, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax4.scatter([min_error_threshold], [min_error_count], color='orange', s=100, zorder=5)
        ax4.axvline(x=optimal_threshold, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Total Errors')
        ax4.set_title('Total Errors vs Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Threshold analysis plot saved to: {save_path}")
        print(f"Minimum error threshold: {min_error_threshold:.3f} (Total errors: {min_error_count})")
    
    def plot_ethnicity_analysis(self, save_path='ethnicity_analysis.png'):
        """Analyze performance across different ethnicities"""
        if self.results_df is None:
            print("No results data available")
            return
        
        # Analyze same-person comparisons by ethnicity
        same_person_df = self.results_df[self.results_df['comparison_type'] == 'same_person'].copy()
        
        if 'ethnicity' not in same_person_df.columns:
            print("Ethnicity data not available in results")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot by ethnicity
        ethnicities = same_person_df['ethnicity'].unique()
        ethnicity_data = [same_person_df[same_person_df['ethnicity'] == eth]['similarity'] 
                         for eth in ethnicities]
        
        box_plot = ax1.boxplot(ethnicity_data, labels=ethnicities, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(ethnicities)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Similarity Score')
        ax1.set_title('Same-Person Similarity by Ethnicity')