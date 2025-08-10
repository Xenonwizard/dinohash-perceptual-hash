import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import cv2
from pathlib import Path
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import time

class MTCNNOriginalComparator:
    def __init__(self, output_dir='mtcnn_analysis_results'):
        """Initialize comparator with output directory structure"""
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Store results for each metric
        self.all_results = []
        self.metric_summaries = {}
        
    def setup_directories(self):
        """Create organized directory structure for results"""
        directories = [
            'plots/quality_metrics',
            'plots/comparisons', 
            'plots/distributions',
            'data/individual_results',
            'data/summaries',
            'reports/detailed',
            'reports/summaries',
            'images/processed_faces',
            'images/original_crops',
            'images/side_by_side'
        ]
        
        for dir_path in directories:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Created analysis directory structure in: {self.output_dir}")

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for analysis"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
        except Exception as e:
            print(f"Error loading {image_path}: {str(e)}")
            return None

    def detect_and_extract_face(self, image, min_face_size=40, scale_factor=0.709):
        """Detect face using MTCNN and extract face region"""
        try:
            from mtcnn import MTCNN
            detector = MTCNN(min_face_size=min_face_size, 
                           scale_factor=scale_factor)
            
            start_time = time.time()
            detections = detector.detect_faces(image)
            detection_time = time.time() - start_time
            
            if not detections:
                return None, None, 0, detection_time
            
            # Get the most confident detection
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            x, y, width, height = best_detection['box']
            confidence = best_detection['confidence']
            
            # Extract face region
            face_region = image[y:y+height, x:x+width]
            
            return face_region, best_detection, confidence, detection_time
            
        except ImportError:
            print("MTCNN not installed. Install with: pip install mtcnn tensorflow")
            return None, None, 0, 0
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return None, None, 0, 0

    # def calculate_comprehensive_metrics(self, original_img, processed_img):
    #     """Calculate comprehensive quality metrics between original and processed images"""
    #     metrics = {}
        
    #     try:
    #         # Ensure both images are same size for fair comparison
    #         if original_img.shape != processed_img.shape:
    #             # Resize processed to match original for comparison
    #             processed_resized = cv2.resize(processed_img, 
    #                                          (original_img.shape[1], original_img.shape[0]))
    #         else:
    #             processed_resized = processed_img
            
    #         # Convert to grayscale for certain metrics
    #         orig_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) if len(original_img.shape) == 3 else original_img
    #         proc_gray = cv2.cvtColor(processed_resized, cv2.COLOR_RGB2GRAY) if len(processed_resized.shape) == 3 else processed_resized
            
    #         # 1. SSIM (Structural Similarity Index)
    #         metrics['ssim_score'] = ssim(orig_gray, proc_gray, data_range=255)
            
    #         # 2. PSNR (Peak Signal-to-Noise Ratio)  
    #         metrics['psnr_score'] = psnr(orig_gray, proc_gray, data_range=255)
            
    #         # 3. MSE (Mean Squared Error)
    #         metrics['mse_score'] = mse(orig_gray, proc_gray)
            
    #         # 4. MAE (Mean Absolute Error)
    #         metrics['mae_score'] = np.mean(np.abs(orig_gray.astype(float) - proc_gray.astype(float)))
            
    #         # 5. Blur metrics
    #         original_blur = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    #         processed_blur = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
    #         metrics['original_blur'] = original_blur
    #         metrics['processed_blur'] = processed_blur
    #         metrics['blur_degradation'] = (original_blur - processed_blur) / original_blur * 100 if original_blur > 0 else 0
            
    #         # 6. Sharpness metrics (Gradient magnitude)
    #         def calculate_sharpness(img):
    #             grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    #             grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    #             return np.sqrt(grad_x**2 + grad_y**2).mean()
            
    #         original_sharpness = calculate_sharpness(orig_gray)
    #         processed_sharpness = calculate_sharpness(proc_gray)
    #         metrics['original_sharpness'] = original_sharpness
    #         metrics['processed_sharpness'] = processed_sharpness
    #         metrics['sharpness_degradation'] = (original_sharpness - processed_sharpness) / original_sharpness * 100 if original_sharpness > 0 else 0
            
    #         # 7. Contrast metrics
    #         original_contrast = orig_gray.std()
    #         processed_contrast = proc_gray.std()
    #         metrics['original_contrast'] = original_contrast
    #         metrics['processed_contrast'] = processed_contrast
    #         metrics['contrast_change'] = (processed_contrast - original_contrast) / original_contrast * 100 if original_contrast > 0 else 0
            
    #         # 8. Histogram correlation
    #         hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
    #         hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
    #         metrics['histogram_correlation'] = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)
            
    #         # 9. Edge preservation
    #         edges_orig = cv2.Canny(orig_gray, 50, 150)
    #         edges_proc = cv2.Canny(proc_gray, 50, 150)
    #         edge_similarity = ssim(edges_orig, edges_proc, data_range=255)
    #         metrics['edge_preservation'] = edge_similarity
            
    #         # 10. Pixel value statistics
    #         metrics['mean_pixel_diff'] = np.mean(orig_gray.astype(float) - proc_gray.astype(float))
    #         metrics['std_pixel_diff'] = np.std(orig_gray.astype(float) - proc_gray.astype(float))
            
    #         return metrics
            
    #     except Exception as e:
    #         print(f"Error calculating metrics: {str(e)}")
    #         return {}

    def save_comparison_images(self, original, processed, image_id, face_box=None):
        """Save original, processed, and side-by-side comparison images"""
        try:
            # Save original crop (if face_box provided)
            if face_box is not None:
                x, y, w, h = face_box
                original_crop = original[y:y+h, x:x+w]
                original_path = self.output_dir / 'images/original_crops' / f'{image_id}_original.jpg'
                cv2.imwrite(str(original_path), cv2.cvtColor(original_crop, cv2.COLOR_RGB2BGR))
            
            # Save processed face
            processed_path = self.output_dir / 'images/processed_faces' / f'{image_id}_processed.jpg'
            cv2.imwrite(str(processed_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            
            # Create side-by-side comparison
            if face_box is not None:
                # Resize processed to match original crop for comparison
                original_crop_resized = cv2.resize(original_crop, (processed.shape[1], processed.shape[0]))
                comparison = np.hstack([original_crop_resized, processed])
            else:
                comparison = processed
            
            comparison_path = self.output_dir / 'images/side_by_side' / f'{image_id}_comparison.jpg'
            cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            
            return {
                'processed_path': str(processed_path),
                'comparison_path': str(comparison_path),
                'original_crop_path': str(original_path) if face_box else None
            }
            
        except Exception as e:
            print(f"Error saving images for {image_id}: {str(e)}")
            return {}

    def analyze_single_image(self, image_path, image_id=None, target_size=(160, 160), 
                           mtcnn_params=None):
        """Analyze single image: detect face, compare with original, save results"""
        
        if image_id is None:
            image_id = Path(image_path).stem
        
        if mtcnn_params is None:
            mtcnn_params = {'min_face_size': 40, 'thresholds': [0.6, 0.7, 0.7], 'scale_factor': 0.709}
        
        print(f"Analyzing image: {image_id}")
        
        # Load original image
        original_img = self.load_and_preprocess_image(image_path)
        if original_img is None:
            return None
        
        # Detect and extract face
        face_region, detection_info, confidence, detection_time = self.detect_and_extract_face(
            original_img, **mtcnn_params
        )
        
        if face_region is None:
            print(f"  ❌ No face detected in {image_id}")
            return None
        
        print(f"  ✅ Face detected with confidence: {confidence:.3f}")
        
        # Resize extracted face to target size (simulate typical face recognition preprocessing)
        processed_face = cv2.resize(face_region, target_size)
        
        # For fair comparison, resize the original face region to same target size
        original_face_resized = cv2.resize(face_region, target_size)
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(original_face_resized, processed_face)
        
        if not metrics:
            print(f"  ❌ Failed to calculate metrics for {image_id}")
            return None
        
        # Save comparison images
        image_paths = self.save_comparison_images(
            original_img, processed_face, image_id, detection_info['box'] if detection_info else None
        )
        
        # Compile complete results
        result = {
            'image_id': image_id,
            'image_path': str(image_path),
            'face_confidence': confidence,
            'detection_time': detection_time,
            'original_size': original_img.shape[0] * original_img.shape[1],
            'face_region_size': face_region.shape[0] * face_region.shape[1] if face_region is not None else 0,
            'processed_size': target_size[0] * target_size[1],
            'face_box': detection_info['box'] if detection_info else None,
            **mtcnn_params,
            **metrics,
            **image_paths
        }
        
        # Save individual result
        individual_path = self.output_dir / 'data/individual_results' / f'{image_id}_metrics.json'
        with open(individual_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Print key metrics
        print(f"  📊 SSIM: {metrics.get('ssim_score', 0):.3f}")
        print(f"  📊 PSNR: {metrics.get('psnr_score', 0):.1f} dB")
        print(f"  📊 Blur degradation: {metrics.get('blur_degradation', 0):.1f}%")
        print(f"  📊 Sharpness degradation: {metrics.get('sharpness_degradation', 0):.1f}%")
        print(f"  💾 Results saved to: {individual_path}")
        
        return result

    def process_image_dataset(self, image_paths, mtcnn_params=None):
        """Process entire dataset and generate comprehensive analysis"""
        print(f"\n🚀 Starting analysis of {len(image_paths)} images...")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
        if mtcnn_params is None:
            mtcnn_params = {'min_face_size': 40, 'thresholds': [0.6, 0.7, 0.7], 'scale_factor': 0.709}
        
        successful_results = []
        failed_images = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}]", end=" ")
            
            result = self.analyze_single_image(image_path, f"img_{i:04d}", mtcnn_params=mtcnn_params)
            
            if result:
                successful_results.append(result)
            else:
                failed_images.append(str(image_path))
        
        self.all_results = successful_results
        
        # Save combined results
        if successful_results:
            # Create DataFrame
            results_df = pd.DataFrame(successful_results)
            results_csv_path = self.output_dir / 'data/summaries/all_results.csv'
            results_df.to_csv(results_csv_path, index=False)
            
            print(f"\n✅ Successfully analyzed {len(successful_results)} images")
            print(f"❌ Failed to analyze {len(failed_images)} images")
            print(f"💾 Combined results saved to: {results_csv_path}")
            
            # Generate analysis reports
            self.generate_all_analyses(results_df)
            
        else:
            print("\n❌ No images were successfully analyzed")
        
        return successful_results, failed_images

    def generate_all_analyses(self, results_df):
        """Generate all analysis plots and reports"""
        print("\n📈 Generating comprehensive analysis...")
        
        # 1. Quality metrics distribution
        self.plot_quality_distributions(results_df)
        
        # 2. MTCNN vs Original comparisons
        self.plot_original_vs_processed_comparison(results_df)
        
        # 3. Degradation analysis
        self.plot_quality_degradation_analysis(results_df)
        
        # 4. Face confidence impact
        self.plot_confidence_quality_relationship(results_df)
        
        # 5. Processing impact analysis
        self.plot_processing_impact_analysis(results_df)
        
        # 6. Generate detailed reports
        self.generate_detailed_reports(results_df)
        
        print("✅ All analyses completed!")

    def plot_quality_distributions(self, df):
        """Plot distributions of all quality metrics"""
        print("  📊 Generating quality distributions...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.ravel()
        
        metrics_to_plot = [
            ('ssim_score', 'SSIM Score', 'Higher = Better'),
            ('psnr_score', 'PSNR (dB)', 'Higher = Better'), 
            ('mse_score', 'Mean Squared Error', 'Lower = Better'),
            ('blur_degradation', 'Blur Degradation (%)', 'Lower = Better'),
            ('sharpness_degradation', 'Sharpness Degradation (%)', 'Lower = Better'),
            ('contrast_change', 'Contrast Change (%)', 'Closer to 0 = Better'),
            ('histogram_correlation', 'Histogram Correlation', 'Higher = Better'),
            ('edge_preservation', 'Edge Preservation', 'Higher = Better'),
            ('face_confidence', 'MTCNN Confidence', 'Higher = Better')
        ]
        
        for i, (metric, title, interpretation) in enumerate(metrics_to_plot):
            if metric in df.columns:
                axes[i].hist(df[metric], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(df[metric].mean(), color='red', linestyle='--', 
                              label=f'Mean: {df[metric].mean():.3f}')
                axes[i].set_title(f'{title}\n({interpretation})')
                axes[i].set_xlabel(title)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots/quality_metrics/quality_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"    💾 Saved to: {save_path}")

    def plot_original_vs_processed_comparison(self, df):
        """Compare original vs processed metrics side by side"""
        print("  📊 Generating original vs processed comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Blur comparison
        ax1.scatter(df['original_blur'], df['processed_blur'], alpha=0.6, c=df['face_confidence'], cmap='viridis')
        ax1.plot([df['original_blur'].min(), df['original_blur'].max()], 
                [df['original_blur'].min(), df['original_blur'].max()], 'r--', alpha=0.8, label='Perfect preservation')
        ax1.set_xlabel('Original Blur Score')
        ax1.set_ylabel('Processed Blur Score') 
        ax1.set_title('Blur: Original vs Processed\n(colored by MTCNN confidence)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(ax1.collections[0], ax=ax1, label='Face Confidence')
        
        # Sharpness comparison
        ax2.scatter(df['original_sharpness'], df['processed_sharpness'], alpha=0.6, c=df['face_confidence'], cmap='viridis')
        ax2.plot([df['original_sharpness'].min(), df['original_sharpness'].max()], 
                [df['original_sharpness'].min(), df['original_sharpness'].max()], 'r--', alpha=0.8, label='Perfect preservation')
        ax2.set_xlabel('Original Sharpness')
        ax2.set_ylabel('Processed Sharpness')
        ax2.set_title('Sharpness: Original vs Processed\n(colored by MTCNN confidence)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(ax2.collections[0], ax=ax2, label='Face Confidence')
        
        # Contrast comparison
        ax3.scatter(df['original_contrast'], df['processed_contrast'], alpha=0.6, c=df['face_confidence'], cmap='viridis')
        ax3.plot([df['original_contrast'].min(), df['original_contrast'].max()], 
                [df['original_contrast'].min(), df['original_contrast'].max()], 'r--', alpha=0.8, label='Perfect preservation')
        ax3.set_xlabel('Original Contrast')
        ax3.set_ylabel('Processed Contrast')
        ax3.set_title('Contrast: Original vs Processed\n(colored by MTCNN confidence)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(ax3.collections[0], ax=ax3, label='Face Confidence')
        
        # Quality degradation overview
        degradation_metrics = ['blur_degradation', 'sharpness_degradation']
        degradation_data = [df[metric] for metric in degradation_metrics if metric in df.columns]
        degradation_labels = [metric.replace('_', ' ').title() for metric in degradation_metrics if metric in df.columns]
        
        box_plot = ax4.boxplot(degradation_data, labels=degradation_labels, patch_artist=True)
        colors = ['lightcoral', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='No degradation')
        ax4.set_ylabel('Degradation Percentage')
        ax4.set_title('Quality Degradation Distribution\n(Positive = Worse quality)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots/comparisons/original_vs_processed.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"    💾 Saved to: {save_path}")

    def plot_quality_degradation_analysis(self, df):
        """Analyze patterns in quality degradation"""
        print("  📊 Generating quality degradation analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # SSIM vs degradation metrics
        ax1.scatter(df['ssim_score'], df['blur_degradation'], alpha=0.6, color='red', label='Blur degradation')
        ax1.scatter(df['ssim_score'], df['sharpness_degradation'], alpha=0.6, color='blue', label='Sharpness degradation')
        ax1.set_xlabel('SSIM Score (Overall Quality)')
        ax1.set_ylabel('Degradation Percentage')
        ax1.set_title('Overall Quality vs Specific Degradations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Face confidence vs quality preservation
        ax2.scatter(df['face_confidence'], df['ssim_score'], alpha=0.6, c=df['blur_degradation'], cmap='Reds')
        ax2.set_xlabel('MTCNN Face Confidence')
        ax2.set_ylabel('SSIM Score')
        ax2.set_title('Face Detection Confidence vs Quality\n(colored by blur degradation)')
        plt.colorbar(ax2.collections[0], ax=ax2, label='Blur Degradation %')
        ax2.grid(True, alpha=0.3)
        
        # Processing time impact
        if 'detection_time' in df.columns:
            ax3.scatter(df['detection_time'], df['ssim_score'], alpha=0.6, c=df['face_confidence'], cmap='viridis')
            ax3.set_xlabel('MTCNN Processing Time (seconds)')
            ax3.set_ylabel('SSIM Score')
            ax3.set_title('Processing Time vs Quality\n(colored by confidence)')
            plt.colorbar(ax3.collections[0], ax=ax3, label='Face Confidence')
            ax3.grid(True, alpha=0.3)
        
        # Quality categories analysis
        def categorize_quality(ssim):
            if ssim >= 0.95:
                return 'Excellent'
            elif ssim >= 0.85:
                return 'Good'
            elif ssim >= 0.75:
                return 'Fair'
            else:
                return 'Poor'
        
        df['quality_category'] = df['ssim_score'].apply(categorize_quality)
        quality_counts = df['quality_category'].value_counts()
        
        colors_pie = ['green', 'lightgreen', 'orange', 'red']
        ax4.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%', 
               colors=colors_pie[:len(quality_counts)])
        ax4.set_title('Quality Distribution Categories\n(Based on SSIM scores)')
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots/quality_metrics/degradation_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"    💾 Saved to: {save_path}")

    def plot_confidence_quality_relationship(self, df):
        """Analyze relationship between MTCNN confidence and resulting quality"""
        print("  📊 Generating confidence-quality relationship analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Confidence bins analysis
        df['confidence_bin'] = pd.cut(df['face_confidence'], 
                                     bins=[0, 0.7, 0.85, 0.95, 1.0], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Quality by confidence bin
        quality_by_conf = df.groupby('confidence_bin')['ssim_score'].agg(['mean', 'std', 'count']).reset_index()
        
        bars = ax1.bar(range(len(quality_by_conf)), quality_by_conf['mean'], 
                      yerr=quality_by_conf['std'], capsize=5, alpha=0.8)
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Average SSIM Score')
        ax1.set_title('Quality vs Confidence Level')
        ax1.set_xticks(range(len(quality_by_conf)))
        ax1.set_xticklabels(quality_by_conf['confidence_bin'])
        ax1.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={quality_by_conf.iloc[i]["count"]}', ha='center', va='bottom')
        
        # Detailed scatter with trend line
        ax2.scatter(df['face_confidence'], df['ssim_score'], alpha=0.6, color='blue')
        
        # Add trend line
        z = np.polyfit(df['face_confidence'], df['ssim_score'], 1)
        p = np.poly1d(z)
        ax2.plot(df['face_confidence'], p(df['face_confidence']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        correlation = df['face_confidence'].corr(df['ssim_score'])
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax2.set_xlabel('Face Confidence')
        ax2.set_ylabel('SSIM Score') 
        ax2.set_title('Confidence vs Quality (with trend)')
        ax2.grid(True, alpha=0.3)
        
        # Low confidence analysis
        low_conf_threshold = 0.8
        low_conf_mask = df['face_confidence'] < low_conf_threshold
        
        if low_conf_mask.sum() > 0:
            low_conf_df = df[low_conf_mask]
            high_conf_df = df[~low_conf_mask]
            
            metrics_comparison = pd.DataFrame({
                'Low Confidence': [
                    low_conf_df['ssim_score'].mean(),
                    low_conf_df['psnr_score'].mean(),
                    low_conf_df['blur_degradation'].mean(),
                    low_conf_df['sharpness_degradation'].mean()
                ],
                'High Confidence': [
                    high_conf_df['ssim_score'].mean(),
                    high_conf_df['psnr_score'].mean(),
                    high_conf_df['blur_degradation'].mean(),
                    high_conf_df['sharpness_degradation'].mean()
                ]
            }, index=['SSIM', 'PSNR', 'Blur Deg.', 'Sharpness Deg.'])
            
            metrics_comparison.plot(kind='bar', ax=ax3, alpha=0.8)
            ax3.set_title(f'Quality Metrics: Low vs High Confidence\n(Threshold: {low_conf_threshold})')
            ax3.set_ylabel('Average Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Detection success vs quality
        detection_quality_stats = df.groupby('quality_category').agg({
            'face_confidence': ['mean', 'count'],
            'detection_time': 'mean'
        }).round(3)
        
        ax4.bar(range(len(detection_quality_stats)), 
               detection_quality_stats['face_confidence']['mean'],
               alpha=0.8, color=['red', 'orange', 'lightgreen', 'green'])
        ax4.set_xlabel('Quality Category')
        ax4.set_ylabel('Average Face Confidence')
        ax4.set_title('Face Detection Confidence by Quality Category')
        ax4.set_xticks(range(len(detection_quality_stats)))
        ax4.set_xticklabels(detection_quality_stats.index, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add count labels
        for i, count in enumerate(detection_quality_stats['face_confidence']['count']):
            ax4.text(i, detection_quality_stats['face_confidence']['mean'].iloc[i] + 0.01,
                    f'n={count}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots/quality_metrics/confidence_quality_relationship.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"    💾 Saved to: {save_path}")

    def plot_processing_impact_analysis(self, df):
        """Analyze the impact of MTCNN processing parameters"""
        print("  📊 Generating processing impact analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Size reduction impact
        df['size_reduction_ratio'] = df['processed_size'] / df['original_size']
        
        ax1.scatter(df['size_reduction_ratio'], df['ssim_score'], 
                   alpha=0.6, c=df['face_confidence'], cmap='viridis')
        ax1.set_xlabel('Size Reduction Ratio (Processed/Original)')
        ax1.set_ylabel('SSIM Score')
        ax1.set_title('Image Size Reduction vs Quality\n(colored by face confidence)')
        plt.colorbar(ax1.collections[0], ax=ax1, label='Face Confidence')
        ax1.grid(True, alpha=0.3)
        
        # Processing efficiency analysis
        if 'detection_time' in df.columns:
            ax2.scatter(df['detection_time'], df['face_confidence'], 
                       alpha=0.6, c=df['ssim_score'], cmap='RdYlGn')
            ax2.set_xlabel('Detection Time (seconds)')
            ax2.set_ylabel('Face Confidence')
            ax2.set_title('Processing Time vs Detection Quality\n(colored by SSIM score)')
            plt.colorbar(ax2.collections[0], ax=ax2, label='SSIM Score')
            ax2.grid(True, alpha=0.3)
        
        # Quality preservation by processing parameters
        if 'min_face_size' in df.columns:
            face_size_groups = df.groupby('min_face_size')['ssim_score'].agg(['mean', 'std', 'count']).reset_index()
            
            bars = ax3.bar(range(len(face_size_groups)), face_size_groups['mean'],
                          yerr=face_size_groups['std'], capsize=5, alpha=0.8, color='lightblue')
            ax3.set_xlabel('Minimum Face Size Parameter')
            ax3.set_ylabel('Average SSIM Score')
            ax3.set_title('Quality vs Min Face Size Parameter')
            ax3.set_xticks(range(len(face_size_groups)))
            ax3.set_xticklabels(face_size_groups['min_face_size'])
            ax3.grid(True, alpha=0.3)
            
            # Add count labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={face_size_groups.iloc[i]["count"]}', ha='center', va='bottom')
        
        # Error analysis - identify problematic cases
        poor_quality_threshold = 0.75
        poor_quality_mask = df['ssim_score'] < poor_quality_threshold
        
        if poor_quality_mask.sum() > 0:
            # Compare characteristics of poor vs good quality results
            comparison_df = pd.DataFrame({
                'Poor Quality': [
                    df[poor_quality_mask]['face_confidence'].mean(),
                    df[poor_quality_mask]['detection_time'].mean() if 'detection_time' in df.columns else 0,
                    df[poor_quality_mask]['size_reduction_ratio'].mean(),
                    df[poor_quality_mask]['blur_degradation'].mean(),
                    df[poor_quality_mask]['sharpness_degradation'].mean()
                ],
                'Good Quality': [
                    df[~poor_quality_mask]['face_confidence'].mean(),
                    df[~poor_quality_mask]['detection_time'].mean() if 'detection_time' in df.columns else 0,
                    df[~poor_quality_mask]['size_reduction_ratio'].mean(),
                    df[~poor_quality_mask]['blur_degradation'].mean(),
                    df[~poor_quality_mask]['sharpness_degradation'].mean()
                ]
            }, index=['Confidence', 'Det. Time', 'Size Ratio', 'Blur Deg.', 'Sharp. Deg.'])
            
            comparison_df.plot(kind='bar', ax=ax4, alpha=0.8, color=['red', 'green'])
            ax4.set_title(f'Poor vs Good Quality Characteristics\n(Threshold: SSIM < {poor_quality_threshold})')
            ax4.set_ylabel('Average Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            # Add sample counts
            ax4.text(0.02, 0.98, f'Poor: n={poor_quality_mask.sum()}\nGood: n={(~poor_quality_mask).sum()}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        save_path = self.output_dir / 'plots/quality_metrics/processing_impact_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"    💾 Saved to: {save_path}")

    def generate_detailed_reports(self, df):
        """Generate comprehensive text and JSON reports"""
        print("  📄 Generating detailed reports...")
        
        # 1. Summary statistics report
        self.generate_summary_statistics_report(df)
        
        # 2. Quality assessment report  
        self.generate_quality_assessment_report(df)
        
        # 3. Recommendations report
        self.generate_recommendations_report(df)
        
        # 4. Individual image analysis report
        self.generate_individual_analysis_report(df)
        
        # 5. JSON summary for programmatic access
        self.generate_json_summary(df)

    def generate_summary_statistics_report(self, df):
        """Generate summary statistics report"""
        report_path = self.output_dir / 'reports/summaries/summary_statistics.txt'
        
        with open(report_path, 'w') as f:
            f.write("MTCNN VS ORIGINAL IMAGE QUALITY ANALYSIS - SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total images processed: {len(df)}\n")
            f.write(f"Successful face detections: {len(df)} (100%)\n")
            f.write(f"Average face confidence: {df['face_confidence'].mean():.3f} (±{df['face_confidence'].std():.3f})\n")
            f.write(f"Average detection time: {df['detection_time'].mean():.3f} seconds\n\n")
            
            f.write("QUALITY METRICS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            metrics = [
                ('ssim_score', 'SSIM (Structural Similarity)', 'higher is better'),
                ('psnr_score', 'PSNR (Peak Signal-to-Noise Ratio)', 'higher is better'),
                ('mse_score', 'MSE (Mean Squared Error)', 'lower is better'),
                ('mae_score', 'MAE (Mean Absolute Error)', 'lower is better'),
                ('blur_degradation', 'Blur Degradation (%)', 'lower is better'),
                ('sharpness_degradation', 'Sharpness Degradation (%)', 'lower is better'),
                ('contrast_change', 'Contrast Change (%)', 'closer to 0 is better'),
                ('histogram_correlation', 'Histogram Correlation', 'higher is better'),
                ('edge_preservation', 'Edge Preservation', 'higher is better')
            ]
            
            for metric, name, interpretation in metrics:
                if metric in df.columns:
                    f.write(f"{name}:\n")
                    f.write(f"  Mean: {df[metric].mean():.4f} (±{df[metric].std():.4f})\n")
                    f.write(f"  Median: {df[metric].median():.4f}\n")
                    f.write(f"  Range: [{df[metric].min():.4f}, {df[metric].max():.4f}]\n")
                    f.write(f"  Interpretation: {interpretation}\n\n")
            
            f.write("QUALITY CATEGORY DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            quality_counts = df['quality_category'].value_counts()
            total = len(df)
            for category, count in quality_counts.items():
                percentage = (count / total) * 100
                f.write(f"{category}: {count} images ({percentage:.1f}%)\n")
            
            f.write(f"\nPROCESSING EFFICIENCY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average size reduction: {(1 - df['size_reduction_ratio'].mean()) * 100:.1f}%\n")
            f.write(f"Average processing time per image: {df['detection_time'].mean():.3f} seconds\n")
            f.write(f"Total processing time: {df['detection_time'].sum():.2f} seconds\n")
        
        print(f"    💾 Summary statistics saved to: {report_path}")

    def generate_quality_assessment_report(self, df):
        """Generate detailed quality assessment report"""
        report_path = self.output_dir / 'reports/detailed/quality_assessment.txt'
        
        with open(report_path, 'w') as f:
            f.write("MTCNN QUALITY ASSESSMENT - DETAILED ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall quality assessment
            avg_ssim = df['ssim_score'].mean()
            f.write("OVERALL QUALITY ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            
            if avg_ssim >= 0.95:
                assessment = "EXCELLENT - Minimal quality degradation"
                emoji = "🟢"
            elif avg_ssim >= 0.85:
                assessment = "GOOD - Acceptable quality preservation"
                emoji = "🟡"
            elif avg_ssim >= 0.75:
                assessment = "FAIR - Noticeable quality degradation"
                emoji = "🟠"
            else:
                assessment = "POOR - Significant quality loss"
                emoji = "🔴"
            
            f.write(f"{emoji} Overall Rating: {assessment}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
            
            # Specific quality issues
            f.write("QUALITY DEGRADATION ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            blur_deg = df['blur_degradation'].mean()
            sharp_deg = df['sharpness_degradation'].mean()
            contrast_change = abs(df['contrast_change'].mean())
            
            f.write(f"Blur degradation: {blur_deg:.2f}% average\n")
            if blur_deg > 10:
                f.write("  ⚠️  WARNING: Significant blur increase detected\n")
            elif blur_deg < -5:
                f.write("  ℹ️  INFO: Processing appears to reduce blur (possibly due to interpolation)\n")
            else:
                f.write("  ✅ GOOD: Minimal blur degradation\n")
            
            f.write(f"Sharpness degradation: {sharp_deg:.2f}% average\n")
            if sharp_deg > 15:
                f.write("  ⚠️  WARNING: Significant sharpness loss detected\n")
            elif sharp_deg < 0:
                f.write("  ℹ️  INFO: Processing appears to increase sharpness\n")
            else:
                f.write("  ✅ GOOD: Minimal sharpness degradation\n")
            
            f.write(f"Contrast change: {contrast_change:.2f}% average\n")
            if contrast_change > 10:
                f.write("  ⚠️  WARNING: Significant contrast alteration\n")
            else:
                f.write("  ✅ GOOD: Minimal contrast change\n")
            
            f.write(f"\nEDGE PRESERVATION:\n")
            edge_pres = df['edge_preservation'].mean()
            f.write(f"Average edge preservation: {edge_pres:.4f}\n")
            if edge_pres >= 0.9:
                f.write("  ✅ EXCELLENT: Edges very well preserved\n")
            elif edge_pres >= 0.8:
                f.write("  ✅ GOOD: Edges adequately preserved\n")
            else:
                f.write("  ⚠️  WARNING: Significant edge information loss\n")
            
            # Confidence-quality relationship
            f.write(f"\nCONFIDENCE-QUALITY RELATIONSHIP:\n")
            f.write("-" * 40 + "\n")
            correlation = df['face_confidence'].corr(df['ssim_score'])
            f.write(f"Correlation between face confidence and quality: {correlation:.4f}\n")
            
            if correlation > 0.5:
                f.write("  ✅ STRONG: Higher confidence typically means better quality\n")
            elif correlation > 0.3:
                f.write("  ℹ️  MODERATE: Some relationship between confidence and quality\n")
            else:
                f.write("  ⚠️  WEAK: Confidence is not a reliable quality predictor\n")
            
            # Identify problem cases
            f.write(f"\nPROBLEM CASES ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            poor_quality = df[df['ssim_score'] < 0.75]
            if len(poor_quality) > 0:
                f.write(f"Found {len(poor_quality)} poor quality cases (SSIM < 0.75):\n")
                f.write(f"  Average confidence: {poor_quality['face_confidence'].mean():.3f}\n")
                f.write(f"  Average blur degradation: {poor_quality['blur_degradation'].mean():.2f}%\n")
                f.write(f"  Average sharpness degradation: {poor_quality['sharpness_degradation'].mean():.2f}%\n")
                
                # List worst cases
                worst_cases = poor_quality.nsmallest(5, 'ssim_score')
                f.write(f"\n  Worst quality images:\n")
                for _, row in worst_cases.iterrows():
                    f.write(f"    {row['image_id']}: SSIM={row['ssim_score']:.3f}, Confidence={row['face_confidence']:.3f}\n")
            else:
                f.write("✅ No poor quality cases detected (all SSIM >= 0.75)\n")
        
        print(f"    💾 Quality assessment saved to: {report_path}")

    def generate_recommendations_report(self, df):
        """Generate actionable recommendations based on analysis"""
        report_path = self.output_dir / 'reports/summaries/recommendations.txt'
        
        with open(report_path, 'w') as f:
            f.write("MTCNN OPTIMIZATION RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            avg_ssim = df['ssim_score'].mean()
            poor_count = (df['ssim_score'] < 0.75).sum()
            avg_conf = df['face_confidence'].mean()
            
            f.write(f"Overall quality score: {avg_ssim:.3f}/1.000\n")
            f.write(f"Problem cases: {poor_count}/{len(df)} ({poor_count/len(df)*100:.1f}%)\n")
            f.write(f"Average detection confidence: {avg_conf:.3f}/1.000\n\n")
            
            f.write("SPECIFIC RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            
            # Quality-based recommendations
            if avg_ssim >= 0.9:
                f.write("✅ QUALITY: Current settings provide excellent quality preservation\n")
                f.write("   → Consider optimizing for speed if processing time is a concern\n\n")
            elif avg_ssim >= 0.8:
                f.write("⚠️  QUALITY: Good quality but room for improvement\n")
                if df['blur_degradation'].mean() > 10:
                    f.write("   → Consider increasing minimum face size to reduce blur\n")
                if df['sharpness_degradation'].mean() > 15:
                    f.write("   → Consider adjusting scale factor (try 0.6-0.7 range)\n")
                f.write("\n")
            else:
                f.write("🔴 QUALITY: Significant quality issues detected\n")
                f.write("   → URGENT: Review MTCNN parameters\n")
                f.write("   → Consider increasing minimum face size\n")
                f.write("   → Consider lowering detection thresholds\n")
                f.write("   → Evaluate input image quality\n\n")
            
            # Confidence-based recommendations
            low_conf_count = (df['face_confidence'] < 0.7).sum()
            if low_conf_count > len(df) * 0.1:  # More than 10% low confidence
                f.write(f"⚠️  DETECTION: {low_conf_count} images have low detection confidence\n")
                f.write("   → Consider lowering detection thresholds (try [0.5, 0.6, 0.6])\n")
                f.write("   → Review input image preprocessing\n")
                f.write("   → Consider image enhancement before face detection\n\n")
            
            # Processing efficiency recommendations
            avg_time = df['detection_time'].mean()
            if avg_time > 0.2:
                f.write("⚠️  SPEED: Processing time may be suboptimal\n")
                f.write(f"   → Current: {avg_time:.3f}s per image\n")
                f.write("   → Consider increasing minimum face size for speed\n")
                f.write("   → Consider increasing scale factor to reduce pyramid levels\n\n")
            elif avg_time < 0.05:
                f.write("✅ SPEED: Very fast processing\n")
                f.write("   → Consider if quality can be improved with more processing time\n\n")
            
            # Parameter-specific recommendations
            f.write("PARAMETER OPTIMIZATION SUGGESTIONS:\n")
            f.write("-" * 40 + "\n")
            
            if 'min_face_size' in df.columns:
                current_min_size = df['min_face_size'].iloc[0]
                f.write(f"Current minimum face size: {current_min_size}\n")
                
                if poor_count > 0:
                    f.write("   → Try increasing to 60-80 for better quality\n")
                elif avg_time > 0.15:
                    f.write("   → Try increasing to 50-60 for faster processing\n")
                else:
                    f.write("   → Current setting appears optimal\n")
            
            f.write(f"\nCurrent detection thresholds: {df.iloc[0].get('thresholds', 'Not available')}\n")
            if avg_conf < 0.8:
                f.write("   → Try lowering to [0.5, 0.6, 0.6] for better detection\n")
            elif poor_count == 0 and avg_conf > 0.9:
                f.write("   → Try raising to [0.7, 0.8, 0.8] for more selective detection\n")
            else:
                f.write("   → Current thresholds appear reasonable\n")
            
            f.write(f"\nCurrent scale factor: {df.iloc[0].get('scale_factor', 'Not available')}\n")
            if avg_time > 0.2:
                f.write("   → Try increasing to 0.8-0.9 for faster processing\n")
            elif poor_count > len(df) * 0.05:
                f.write("   → Try decreasing to 0.5-0.6 for better detection\n")
            else:
                f.write("   → Current scale factor appears optimal\n")
            
            f.write(f"\nNEXT STEPS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Review individual problem cases in detailed results\n")
            f.write("2. Test recommended parameter changes on a subset\n")
            f.write("3. Consider input image preprocessing if quality issues persist\n")
            f.write("4. Monitor processing time vs quality trade-offs\n")
            f.write("5. Re-run analysis after parameter adjustments\n")
        
        print(f"    💾 Recommendations saved to: {report_path}")

    def generate_individual_analysis_report(self, df):
        """Generate report focusing on individual image analysis"""
        report_path = self.output_dir / 'reports/detailed/individual_analysis.txt'
        
        with open(report_path, 'w') as f:
            f.write("INDIVIDUAL IMAGE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Best performing images
            f.write("TOP 10 BEST QUALITY RESULTS:\n")
            f.write("-" * 40 + "\n")
            best_results = df.nlargest(10, 'ssim_score')
            
            f.write(f"{'Image ID':<15} {'SSIM':<8} {'PSNR':<8} {'Confidence':<12} {'Notes':<20}\n")
            f.write("-" * 70 + "\n")
            
            for _, row in best_results.iterrows():
                notes = []
                if row['blur_degradation'] < 0:
                    notes.append("Less blur")
                if row['sharpness_degradation'] < 0:
                    notes.append("Sharper")
                note_str = ", ".join(notes) if notes else "Standard"
                
                f.write(f"{row['image_id']:<15} {row['ssim_score']:<8.3f} {row['psnr_score']:<8.1f} "
                       f"{row['face_confidence']:<12.3f} {note_str:<20}\n")
            
            # Worst performing images
            f.write(f"\nBOTTOM 10 WORST QUALITY RESULTS:\n")
            f.write("-" * 40 + "\n")
            worst_results = df.nsmallest(10, 'ssim_score')
            
            f.write(f"{'Image ID':<15} {'SSIM':<8} {'PSNR':<8} {'Confidence':<12} {'Issues':<30}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in worst_results.iterrows():
                issues = []
                if row['blur_degradation'] > 15:
                    issues.append("High blur degradation")
                if row['sharpness_degradation'] > 20:
                    issues.append("High sharpness loss")
                if row['face_confidence'] < 0.7:
                    issues.append("Low confidence")
                if row.get('edge_preservation', 1) < 0.8:
                    issues.append("Poor edge preservation")
                
                issue_str = "; ".join(issues) if issues else "Unknown cause"
                
                f.write(f"{row['image_id']:<15} {row['ssim_score']:<8.3f} {row['psnr_score']:<8.1f} "
                       f"{row['face_confidence']:<12.3f} {issue_str:<30}\n")
            
            # Statistical outliers
            f.write(f"\nSTATISTICAL OUTLIERS:\n")
            f.write("-" * 40 + "\n")
            
            # Images with unusual blur patterns
            blur_threshold = df['blur_degradation'].mean() + 2 * df['blur_degradation'].std()
            blur_outliers = df[df['blur_degradation'] > blur_threshold]
            
            if len(blur_outliers) > 0:
                f.write(f"Images with excessive blur degradation (>{blur_threshold:.1f}%):\n")
                for _, row in blur_outliers.iterrows():
                    f.write(f"  {row['image_id']}: {row['blur_degradation']:.1f}% degradation\n")
            
            # Images with unusual confidence vs quality patterns
            high_conf_low_quality = df[(df['face_confidence'] > 0.9) & (df['ssim_score'] < 0.8)]
            low_conf_high_quality = df[(df['face_confidence'] < 0.7) & (df['ssim_score'] > 0.9)]
            
            if len(high_conf_low_quality) > 0:
                f.write(f"\nHigh confidence but low quality ({len(high_conf_low_quality)} images):\n")
                for _, row in high_conf_low_quality.head(5).iterrows():
                    f.write(f"  {row['image_id']}: Conf={row['face_confidence']:.3f}, SSIM={row['ssim_score']:.3f}\n")
            
            if len(low_conf_high_quality) > 0:
                f.write(f"\nLow confidence but high quality ({len(low_conf_high_quality)} images):\n")
                for _, row in low_conf_high_quality.head(5).iterrows():
                    f.write(f"  {row['image_id']}: Conf={row['face_confidence']:.3f}, SSIM={row['ssim_score']:.3f}\n")
        
        print(f"    💾 Individual analysis saved to: {report_path}")

    def generate_json_summary(self, df):
        """Generate JSON summary for programmatic access"""
        summary_path = self.output_dir / 'data/summaries/analysis_summary.json'
        
        summary = {
            'dataset_info': {
                'total_images': len(df),
                'successful_detections': len(df),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'quality_metrics': {
                'overall_ssim': {
                    'mean': float(df['ssim_score'].mean()),
                    'std': float(df['ssim_score'].std()),
                    'median': float(df['ssim_score'].median()),
                    'min': float(df['ssim_score'].min()),
                    'max': float(df['ssim_score'].max())
                },
                'overall_psnr': {
                    'mean': float(df['psnr_score'].mean()),
                    'std': float(df['psnr_score'].std()),
                    'median': float(df['psnr_score'].median())
                },
                'degradation_metrics': {
                    'blur_degradation_percent': float(df['blur_degradation'].mean()),
                    'sharpness_degradation_percent': float(df['sharpness_degradation'].mean()),
                    'contrast_change_percent': float(df['contrast_change'].mean())
                }
            },
            'quality_distribution': df['quality_category'].value_counts().to_dict(),
            'detection_performance': {
                'average_confidence': float(df['face_confidence'].mean()),
                'low_confidence_count': int((df['face_confidence'] < 0.7).sum()),
                'average_detection_time': float(df['detection_time'].mean())
            },
            'processing_efficiency': {
                'average_size_reduction_ratio': float(df['size_reduction_ratio'].mean()),
                'total_processing_time': float(df['detection_time'].sum())
            },
            'problem_cases': {
                'poor_quality_count': int((df['ssim_score'] < 0.75).sum()),
                'low_confidence_count': int((df['face_confidence'] < 0.7).sum()),
                'high_degradation_count': int((df['blur_degradation'] > 20).sum())
            },
            'correlations': {
                'confidence_quality': float(df['face_confidence'].corr(df['ssim_score'])),
                'time_quality': float(df['detection_time'].corr(df['ssim_score'])),
                'size_quality': float(df['size_reduction_ratio'].corr(df['ssim_score']))
            },
            'best_images': df.nlargest(5, 'ssim_score')[['image_id', 'ssim_score', 'face_confidence']].to_dict('records'),
            'worst_images': df.nsmallest(5, 'ssim_score')[['image_id', 'ssim_score', 'face_confidence']].to_dict('records')
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"    💾 JSON summary saved to: {summary_path}")

# Usage example and utility functions
def run_complete_analysis(image_directory, output_dir='mtcnn_analysis_results', 
                         mtcnn_params=None, image_extensions=None):
    """Run complete MTCNN vs Original analysis on a directory of images"""
    
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    if mtcnn_params is None:
        mtcnn_params = {
            'min_face_size': 40,
            'thresholds': [0.6, 0.7, 0.7],
            'scale_factor': 0.709
        }
    
    # Find all image files
    image_dir = Path(image_directory)
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f'*{ext}')))
        image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))
    
    if not image_paths:
        print(f"❌ No images found in {image_directory}")
        return None
    
    print(f"🔍 Found {len(image_paths)} images in {image_directory}")
    
    # Initialize comparator
    comparator = MTCNNOriginalComparator(output_dir)
    
    # Process all images
    successful_results, failed_images = comparator.process_image_dataset(
        image_paths, mtcnn_params
    )
    
    if successful_results:
        print(f"\n🎉 Analysis complete!")
        print(f"📁 All results saved to: {output_dir}")
        print(f"📊 Processed: {len(successful_results)} successful, {len(failed_images)} failed")
        
        return comparator
    else:
        print("❌ Analysis failed - no successful results")
        return None


def compare_mtcnn_parameters(image_directory, output_base_dir='parameter_comparison', 
                           parameter_sets=None):
    """Compare different MTCNN parameter configurations"""
    
    if parameter_sets is None:
        parameter_sets = {
            'default': {'min_face_size': 40, 'thresholds': [0.6, 0.7, 0.7], 'scale_factor': 0.709},
            'high_quality': {'min_face_size': 60, 'thresholds': [0.5, 0.6, 0.6], 'scale_factor': 0.6},
            'fast_processing': {'min_face_size': 80, 'thresholds': [0.7, 0.8, 0.8], 'scale_factor': 0.8},
            'sensitive': {'min_face_size': 20, 'thresholds': [0.5, 0.5, 0.5], 'scale_factor': 0.5}
        }
    
    print(f"🔄 Comparing {len(parameter_sets)} parameter configurations...")
    
    comparison_results = {}
    
    for config_name, params in parameter_sets.items():
        print(f"\n🧪 Testing configuration: {config_name}")
        print(f"   Parameters: {params}")
        
        output_dir = Path(output_base_dir) / config_name
        
        comparator = run_complete_analysis(
            image_directory, 
            output_dir=str(output_dir),
            mtcnn_params=params
        )
        
        if comparator and comparator.all_results:
            df = pd.DataFrame(comparator.all_results)
            
            # Store summary metrics for comparison
            comparison_results[config_name] = {
                'params': params,
                'results': {
                    'avg_ssim': df['ssim_score'].mean(),
                    'avg_psnr': df['psnr_score'].mean(),
                    'avg_confidence': df['face_confidence'].mean(),
                    'avg_detection_time': df['detection_time'].mean(),
                    'success_rate': len(df) / len(list(Path(image_directory).glob('*.jpg'))),  # Approximate
                    'poor_quality_rate': (df['ssim_score'] < 0.75).mean(),
                    'blur_degradation': df['blur_degradation'].mean(),
                    'sharpness_degradation': df['sharpness_degradation'].mean()
                }
            }
    
    # Generate comparison report
    if comparison_results:
        generate_parameter_comparison_report(comparison_results, output_base_dir)
    
    return comparison_results


def generate_parameter_comparison_report(comparison_results, output_dir):
    """Generate comparison report across different parameter sets"""
    
    output_path = Path(output_dir) / 'parameter_comparison_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("MTCNN PARAMETER COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        # Create comparison table
        configs = list(comparison_results.keys())
        
        f.write(f"{'Metric':<25}")
        for config in configs:
            f.write(f"{config:<15}")
        f.write("\n" + "-" * (25 + 15 * len(configs)) + "\n")
        
        metrics = [
            ('avg_ssim', 'SSIM Score', '{:.3f}'),
            ('avg_psnr', 'PSNR (dB)', '{:.1f}'),
            ('avg_confidence', 'Face Confidence', '{:.3f}'),
            ('avg_detection_time', 'Detection Time (s)', '{:.3f}'),
            ('poor_quality_rate', 'Poor Quality Rate', '{:.1%}'),
            ('blur_degradation', 'Blur Degradation %', '{:.1f}'),
            ('sharpness_degradation', 'Sharpness Deg. %', '{:.1f}')
        ]
        
        for metric_key, metric_name, format_str in metrics:
            f.write(f"{metric_name:<25}")
            for config in configs:
                value = comparison_results[config]['results'][metric_key]
                f.write(f"{format_str.format(value):<15}")
            f.write("\n")
        
        f.write("\n\nRANKING BY QUALITY (SSIM):\n")
        f.write("-" * 40 + "\n")
        
        # Sort by SSIM score
        quality_ranking = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['results']['avg_ssim'],
            reverse=True
        )
        
        for i, (config, data) in enumerate(quality_ranking, 1):
            ssim = data['results']['avg_ssim']
            time = data['results']['avg_detection_time']
            f.write(f"{i}. {config}: SSIM={ssim:.3f}, Time={time:.3f}s\n")
        
        f.write("\n\nRANKING BY SPEED:\n")
        f.write("-" * 40 + "\n")
        
        # Sort by detection time
        speed_ranking = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['results']['avg_detection_time']
        )
        
        for i, (config, data) in enumerate(speed_ranking, 1):
            time = data['results']['avg_detection_time']
            ssim = data['results']['avg_ssim']
            f.write(f"{i}. {config}: Time={time:.3f}s, SSIM={ssim:.3f}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        
        best_quality = quality_ranking[0]
        fastest = speed_ranking[0]
        
        f.write(f"Best Quality: {best_quality[0]} (SSIM: {best_quality[1]['results']['avg_ssim']:.3f})\n")
        f.write(f"Fastest Processing: {fastest[0]} (Time: {fastest[1]['results']['avg_detection_time']:.3f}s)\n")
        
        # Find balanced option
        balanced_scores = {}
        for config, data in comparison_results.items():
            # Normalize scores (0-1) and combine quality + speed
            quality_norm = data['results']['avg_ssim']  # Already 0-1
            speed_norm = 1 / (1 + data['results']['avg_detection_time'])  # Invert time for scoring
            balanced_scores[config] = (quality_norm + speed_norm) / 2
        
        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
        f.write(f"Best Balanced: {best_balanced[0]} (Combined score: {best_balanced[1]:.3f})\n")
        
        f.write(f"\nUse '{best_quality[0]}' for maximum quality\n")
        f.write(f"Use '{fastest[0]}' for maximum speed\n")
        f.write(f"Use '{best_balanced[0]}' for balanced performance\n")
    
    print(f"📄 Parameter comparison report saved to: {output_path}")


def batch_analyze_datasets(dataset_configs, base_output_dir='batch_analysis'):
    """Analyze multiple datasets with different configurations"""
    
    print(f"🔄 Starting batch analysis of {len(dataset_configs)} datasets...")
    
    all_results = {}
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n📁 Processing dataset: {dataset_name}")
        
        output_dir = Path(base_output_dir) / dataset_name
        
        comparator = run_complete_analysis(
            config['image_directory'],
            output_dir=str(output_dir),
            mtcnn_params=config.get('mtcnn_params')
        )
        
        if comparator and comparator.all_results:
            df = pd.DataFrame(comparator.all_results)
            
            all_results[dataset_name] = {
                'comparator': comparator,
                'dataframe': df,
                'config': config,
                'summary': {
                    'total_images': len(df),
                    'avg_ssim': df['ssim_score'].mean(),
                    'avg_confidence': df['face_confidence'].mean(),
                    'success_rate': len(df)
                }
            }
    
    # Generate cross-dataset comparison
    if all_results:
        generate_cross_dataset_report(all_results, base_output_dir)
    
    return all_results


def generate_cross_dataset_report(all_results, output_dir):
    """Generate comparison report across different datasets"""
    
    report_path = Path(output_dir) / 'cross_dataset_comparison.txt'
    
    with open(report_path, 'w') as f:
        f.write("CROSS-DATASET MTCNN ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET COMPARISON SUMMARY:\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"{'Dataset':<20} {'Images':<8} {'Avg SSIM':<10} {'Avg Conf.':<10} {'Notes':<30}\n")
        f.write("-" * 80 + "\n")
        
        for dataset_name, data in all_results.items():
            df = data['dataframe']
            notes = []
            
            if df['ssim_score'].mean() > 0.9:
                notes.append("High quality")
            if df['face_confidence'].mean() > 0.9:
                notes.append("High confidence")
            if (df['ssim_score'] < 0.75).sum() > len(df) * 0.1:
                notes.append("Some poor quality")
            
            note_str = "; ".join(notes) if notes else "Standard"
            
            f.write(f"{dataset_name:<20} {len(df):<8} {df['ssim_score'].mean():<10.3f} "
                   f"{df['face_confidence'].mean():<10.3f} {note_str:<30}\n")
        
        f.write("\n\nDETAILED ANALYSIS BY DATASET:\n")
        f.write("-" * 40 + "\n")
        
        for dataset_name, data in all_results.items():
            df = data['dataframe']
            
            f.write(f"\n{dataset_name.upper()}:\n")
            f.write(f"  Total images: {len(df)}\n")
            f.write(f"  Quality metrics:\n")
            f.write(f"    SSIM: {df['ssim_score'].mean():.3f} ± {df['ssim_score'].std():.3f}\n")
            f.write(f"    PSNR: {df['psnr_score'].mean():.1f} ± {df['psnr_score'].std():.1f} dB\n")
            f.write(f"  Detection performance:\n")
            f.write(f"    Confidence: {df['face_confidence'].mean():.3f} ± {df['face_confidence'].std():.3f}\n")
            f.write(f"    Processing time: {df['detection_time'].mean():.3f}s per image\n")
            f.write(f"  Quality issues:\n")
            f.write(f"    Poor quality (SSIM<0.75): {(df['ssim_score'] < 0.75).sum()}/{len(df)} ({(df['ssim_score'] < 0.75).mean()*100:.1f}%)\n")
            f.write(f"    Low confidence (<0.7): {(df['face_confidence'] < 0.7).sum()}/{len(df)} ({(df['face_confidence'] < 0.7).mean()*100:.1f}%)\n")
    
    print(f"📄 Cross-dataset report saved to: {report_path}")


# Main execution function
# if __name__ == "__main__":
#     print("🚀 MTCNN vs Original Image Quality Analyzer")
#     print("=" * 50)
    
#     # Example usage - uncomment and modify paths as needed
    
#     # Single dataset analysis
#     """
#     image_directory = "path/to/your/images"
#     comparator = run_complete_analysis(
#         image_directory,
#         output_dir='my_mtcnn_analysis',
#         mtcnn_params={'min_face_size': 40, 'thresholds': [0.6, 0.7, 0.7], 'scale_factor': 0.709}
#     )
#     """
    
#     # Parameter comparison
#     """
#     comparison_results = compare_mtcnn_parameters(
#         image_directory="path/to/your/images",
#         output_base_dir='parameter_comparison_results'
#     )
#     """
    
#     # Batch analysis of multiple datasets
#     """
#     dataset_configs = {
#         'high_res_faces': {
#             'image_directory': 'path/to/high_res_dataset',
#             'mtcnn_params': {'min_face_size': 60, 'thresholds': [0.6, 0.7, 0.7], 'scale_factor': 0.7}
#         },
#         'mobile_photos': {
#             'image_directory': 'path/to/mobile_dataset',
#             'mtcnn_params': {'min_face_size': 40, 'thresholds': [0.5, 0.6, 0.6], 'scale_factor': 0.709}
#         }
#     }
    
#     batch_results = batch_analyze_datasets(dataset_configs, 'batch_analysis_results')
#     """
    
#     # For testing with sample data
#     print("Creating sample data for demonstration...")
    
#     # Generate sample results for testing
#     sample_data = []
#     np.random.seed(42)
    
#     for i in range(100):
#         # Simulate realistic MTCNN processing results
#         confidence = np.random.beta(5, 1.5)  # Higher confidence more likely
        
#         # Quality correlates with confidence but has noise
#         base_ssim = 0.7 + 0.25 * confidence + np.random.normal(0, 0.05)
#         base_ssim = np.clip(base_ssim, 0.3, 0.99)
        
#         psnr = 15 + base_ssim * 20 + np.random.normal(0, 2)
        
#         # Degradation metrics
#         blur_deg = np.random.exponential(5) if confidence < 0.8 else np.random.exponential(2)
#         sharp_deg = np.random.exponential(8) if confidence < 0.7 else np.random.exponential(3)
        
#         sample_data.append({
#             'image_id': f'sample_{i:03d}',
#             'image_path': f'/fake/path/sample_{i:03d}.jpg',
#             'face_confidence': confidence,
#             'detection_time': np.random.exponential(0.1),
#             'original_size': np.random.randint(500000, 2000000),
#             'face_region_size': np.random.randint(50000, 200000),
#             'processed_size': 25600,  # 160x160
#             'face_box': [np.random.randint(50, 200), np.random.randint(50, 200), 
#                         np.random.randint(100, 300), np.random.randint(100, 300)],
#             'min_face_size': 40,
#             'thresholds': [0.6, 0.7, 0.7],
#             'scale_factor': 0.709,
#             'ssim_score': base_ssim,
#             'psnr_score': psnr,
#             'mse_score': np.random.exponential(100),
#             'mae_score': np.random.exponential(20),
#             'original_blur': np.random.exponential(100),
#             'processed_blur': np.random.exponential(100),
#             'blur_degradation': blur_deg,
#             'original_sharpness': np.random.gamma(2, 500),
#             'processed_sharpness': np.random.gamma(2, 500),
#             'sharpness_degradation': sharp_deg,
#             'original_contrast': np.random.normal(50, 10),
#             'processed_contrast': np.random.normal(50, 10),
#             'contrast_change': np.random.normal(0, 5),
#             'histogram_correlation': np.random.beta(8, 2),
#             'edge_preservation': np.random.beta(6, 2),
#             'mean_pixel_diff': np.random.normal(0, 10),
#             'std_pixel_diff': np.random.exponential(15)
#         })
    
#     # Initialize comparator and test with sample data
#     comparator = MTCNNOriginalComparator('sample_analysis_results')
#     comparator.all_results = sample_data
    
#     # Create DataFrame and run analysis
#     sample_df = pd.DataFrame(sample_data)
    
#     # Add quality categories
#     def categorize_quality(ssim):
#         if ssim >= 0.95:
#             return 'Excellent'
#         elif ssim >= 0.85:
#             return 'Good'
#         elif ssim >= 0.75:
#             return 'Fair'
#         else:
#             return 'Poor'
    
#     sample_df['quality_category'] = sample_df['ssim_score'].apply(categorize_quality)
#     sample_df['size_reduction_ratio'] = sample_df['processed_size'] / sample_df['face_region_size']
    
#     print(f"✅ Generated sample data with {len(sample_df)} entries")
#     print("🔄 Running complete analysis on sample data...")
    
#     # Save sample data
#     sample_df.to_csv(comparator.output_dir / 'data/summaries/all_results.csv', index=False)
    
#     # Generate all analyses
#     comparator.generate_all_analyses(sample_df)
    
#     print(f"\n🎉 Sample analysis complete!")
#     print(f"📁 Results saved to: {comparator.output_dir}")
#     print("\n📋 Generated files:")
#     print("  📊 Quality distribution plots")
#     print("  📈 Original vs processed comparisons") 
#     print("  📉 Degradation analysis")
#     print("  🔍 Confidence-quality relationships")
#     print("  ⚙️  Processing impact analysis")
#     print("  📄 Detailed text reports")
#     print("  💾 JSON summary data")
#     print("  🖼️  Individual result files")
    
#     print(f"\n💡 To analyze your own images, modify the paths in the script and run:")
#     print(f"   comparator = run_complete_analysis('your/image/directory')")
# Celebrity Dataset Analysis Configuration
import os
import glob
import random
from pathlib import Path

def setup_celebrity_analysis(base_path="./images/celeb-dataset"):
    """
    Set up analysis for celebrity dataset with 20 images per celebrity
    """
    
    # Define the structure based on your file tree
    races = ['caucasian', 'chinese', 'indian', 'malay']
    
    dataset_configs = {}
    
    for race in races:
        race_path = Path(base_path) / race
        
        # Get all celebrity folders in this race
        celebrity_folders = [d for d in race_path.iterdir() if d.is_dir() and not d.name.endswith('_test')]
        
        for celeb_folder in celebrity_folders:
            celeb_name = celeb_folder.name
            
            # Get all image files in celebrity folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            all_images = []
            
            for ext in image_extensions:
                all_images.extend(celeb_folder.glob(ext))
            
            # Randomly sample 20 images (or all if less than 20)
            if len(all_images) >= 20:
                selected_images = random.sample(all_images, 20)
            else:
                selected_images = all_images
                print(f"⚠️ {race}/{celeb_name} only has {len(all_images)} images (using all)")
            
            # Create config for this celebrity
            config_key = f"{race}_{celeb_name}"
            dataset_configs[config_key] = {
                'image_directory': str(celeb_folder),
                'selected_images': [str(img) for img in selected_images],
                'race': race,
                'celebrity': celeb_name,
                'image_count': len(selected_images),
                'mtcnn_params': {
                    'min_face_size': 40, 
                    'scale_factor': 0.709
                }
            }
    
    return dataset_configs


def run_celebrity_batch_analysis():
    """
    Run the complete celebrity analysis with inf/nan protection
    """
    print("🎭 Celebrity Dataset MTCNN Analysis (Safe Mode)")
    print("=" * 60)
    
    # ... existing setup code ...
    dataset_configs = setup_celebrity_analysis()
    
    try:
        batch_results = batch_analyze_datasets(
            dataset_configs, 
            'celebrity_mtcnn_analysis'
        )
        
        # Clean the results dataframe
        if batch_results and hasattr(batch_results, '__iter__'):
            for config_name, df in batch_results.items():
                if isinstance(df, pd.DataFrame):
                    batch_results[config_name] = clean_dataframe(df)
        
        print(f"\n🎉 Celebrity analysis complete!")
        
        return batch_results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("💡 Try running with smaller batch sizes or check image quality")
        return None
def clean_dataframe(df):
    """Clean the entire dataframe by replacing inf and nan values"""
    print("🧹 Cleaning infinite and NaN values from dataset...")
    
    # Replace inf values with nan first
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Count problematic values
    inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    nan_count = df.isna().sum().sum()
    
    if inf_count > 0 or nan_count > 0:
        print(f"   Found {inf_count} infinite values and {nan_count} NaN values")
        
        # Fill numeric columns with appropriate defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if 'degradation' in col.lower() or 'change' in col.lower():
                df[col].fillna(0.0, inplace=True)
            elif 'score' in col.lower() or 'correlation' in col.lower():
                df[col].fillna(0.0, inplace=True)
            elif 'blur' in col.lower() or 'sharpness' in col.lower():
                df[col].fillna(1.0, inplace=True)
            elif 'contrast' in col.lower():
                df[col].fillna(50.0, inplace=True)
            elif 'time' in col.lower():
                df[col].fillna(0.1, inplace=True)
            else:
                df[col].fillna(0.0, inplace=True)
        
        print("   ✅ Cleaned all problematic values")
    
    return df

def _get_default_metrics(self):
    """Return default metrics when calculation fails"""
    return {
        'ssim_score': 0.0,
        'psnr_score': 0.0,
        'mse_score': 0.0,
        'mae_score': 0.0,
        'original_blur': 1.0,
        'processed_blur': 1.0,
        'blur_degradation': 0.0,
        'original_sharpness': 1.0,
        'processed_sharpness': 1.0,
        'sharpness_degradation': 0.0,
        'original_contrast': 50.0,
        'processed_contrast': 50.0,
        'contrast_change': 0.0,
        'histogram_correlation': 0.0,
        'edge_preservation': 0.0,
        'mean_pixel_diff': 0.0,
        'std_pixel_diff': 0.0
    }
def clean_metrics_data(metrics_dict):
    """
    Clean metrics dictionary by replacing inf and nan values
    """
    cleaned = {}
    
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            # Replace inf and nan values
            if np.isinf(value) or np.isnan(value):
                # Set reasonable defaults based on metric type
                if 'degradation' in key.lower() or 'change' in key.lower():
                    cleaned[key] = 0.0  # No degradation/change
                elif 'score' in key.lower() or 'correlation' in key.lower():
                    cleaned[key] = 0.0  # Poor score/correlation
                elif 'blur' in key.lower() or 'sharpness' in key.lower():
                    cleaned[key] = 1.0  # Minimal blur/sharpness
                elif 'contrast' in key.lower():
                    cleaned[key] = 50.0  # Average contrast
                else:
                    cleaned[key] = 0.0  # Default to 0
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value
    
    return cleaned

def safe_percentage_calculation(original, processed):
    """
    Safely calculate percentage change avoiding division by zero
    """
    if original == 0 or np.isnan(original) or np.isinf(original):
        return 0.0
    
    result = (original - processed) / original * 100
    
    # Cap extreme values
    if np.isinf(result) or np.isnan(result):
        return 0.0
    elif result > 100:
        return 100.0
    elif result < -100:
        return -100.0
    
    return result

def calculate_comprehensive_metrics(self, original_img, processed_img):
    """Calculate comprehensive quality metrics with inf/nan protection"""
    metrics = {}
    
    try:
        # Ensure both images are same size for fair comparison
        if original_img.shape != processed_img.shape:
            processed_resized = cv2.resize(processed_img, 
                                         (original_img.shape[1], original_img.shape[0]))
        else:
            processed_resized = processed_img
        
        # Convert to grayscale for certain metrics
        orig_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) if len(original_img.shape) == 3 else original_img
        proc_gray = cv2.cvtColor(processed_resized, cv2.COLOR_RGB2GRAY) if len(processed_resized.shape) == 3 else processed_resized
        
        # Check for empty/invalid images
        if orig_gray.size == 0 or proc_gray.size == 0:
            return self._get_default_metrics()
        
        # 1. SSIM (Structural Similarity Index)
        try:
            metrics['ssim_score'] = ssim(orig_gray, proc_gray, data_range=255)
        except:
            metrics['ssim_score'] = 0.0
        
        # 2. PSNR (Peak Signal-to-Noise Ratio)
        try:
            psnr_val = psnr(orig_gray, proc_gray, data_range=255)
            metrics['psnr_score'] = psnr_val if np.isfinite(psnr_val) else 0.0
        except:
            metrics['psnr_score'] = 0.0
        
        # 3. MSE (Mean Squared Error)
        try:
            metrics['mse_score'] = mse(orig_gray, proc_gray)
        except:
            metrics['mse_score'] = 0.0
        
        # 4. MAE (Mean Absolute Error)
        try:
            metrics['mae_score'] = np.mean(np.abs(orig_gray.astype(float) - proc_gray.astype(float)))
        except:
            metrics['mae_score'] = 0.0
        
        # 5. Blur metrics with safe calculation
        try:
            original_blur = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            processed_blur = cv2.Laplacian(proc_gray, cv2.CV_64F).var()
            
            metrics['original_blur'] = original_blur if np.isfinite(original_blur) else 1.0
            metrics['processed_blur'] = processed_blur if np.isfinite(processed_blur) else 1.0
            metrics['blur_degradation'] = safe_percentage_calculation(original_blur, processed_blur)
        except:
            metrics['original_blur'] = 1.0
            metrics['processed_blur'] = 1.0
            metrics['blur_degradation'] = 0.0
        
        # 6. Sharpness metrics with safe calculation
        def calculate_sharpness_safe(img):
            try:
                grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                result = np.sqrt(grad_x**2 + grad_y**2).mean()
                return result if np.isfinite(result) else 1.0
            except:
                return 1.0
        
        original_sharpness = calculate_sharpness_safe(orig_gray)
        processed_sharpness = calculate_sharpness_safe(proc_gray)
        
        metrics['original_sharpness'] = original_sharpness
        metrics['processed_sharpness'] = processed_sharpness
        metrics['sharpness_degradation'] = safe_percentage_calculation(original_sharpness, processed_sharpness)
        
        # 7. Contrast metrics with safe calculation
        try:
            original_contrast = orig_gray.std()
            processed_contrast = proc_gray.std()
            
            metrics['original_contrast'] = original_contrast if np.isfinite(original_contrast) else 50.0
            metrics['processed_contrast'] = processed_contrast if np.isfinite(processed_contrast) else 50.0
            metrics['contrast_change'] = safe_percentage_calculation(processed_contrast, original_contrast)  # Note: inverted for "change"
        except:
            metrics['original_contrast'] = 50.0
            metrics['processed_contrast'] = 50.0
            metrics['contrast_change'] = 0.0
        
        # 8. Histogram correlation with safe calculation
        try:
            hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
            hist_proc = cv2.calcHist([proc_gray], [0], None, [256], [0, 256])
            correlation = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)
            metrics['histogram_correlation'] = correlation if np.isfinite(correlation) else 0.0
        except:
            metrics['histogram_correlation'] = 0.0
        
        # 9. Edge preservation with safe calculation
        try:
            edges_orig = cv2.Canny(orig_gray, 50, 150)
            edges_proc = cv2.Canny(proc_gray, 50, 150)
            edge_similarity = ssim(edges_orig, edges_proc, data_range=255)
            metrics['edge_preservation'] = edge_similarity if np.isfinite(edge_similarity) else 0.0
        except:
            metrics['edge_preservation'] = 0.0
        
        # 10. Pixel value statistics with safe calculation
        try:
            mean_diff = np.mean(orig_gray.astype(float) - proc_gray.astype(float))
            std_diff = np.std(orig_gray.astype(float) - proc_gray.astype(float))
            
            metrics['mean_pixel_diff'] = mean_diff if np.isfinite(mean_diff) else 0.0
            metrics['std_pixel_diff'] = std_diff if np.isfinite(std_diff) else 0.0
        except:
            metrics['mean_pixel_diff'] = 0.0
            metrics['std_pixel_diff'] = 0.0
        
        # Final cleanup - ensure no inf/nan values remain
        metrics = clean_metrics_data(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return self._get_default_metrics()



def generate_race_summary(batch_results):
    """
    Generate summary statistics by race
    """
    print(f"\n📊 Generating race-based summary...")
    
    # You can add race-specific analysis here
    # This would aggregate results by race for comparison
    
    pass

# Main execution for celebrity dataset
if __name__ == "__main__":
    # Set random seed for reproducible image selection
    random.seed(42)
    
    # Run the celebrity analysis
    results = run_celebrity_batch_analysis()
    
    # Optional: Additional analysis
    if results:
        print(f"\n💡 Next steps:")
        print(f"  📈 Check race-specific quality patterns")
        print(f"  🔍 Analyze celebrity-specific results") 
        print(f"  📊 Compare processing effectiveness across demographics")