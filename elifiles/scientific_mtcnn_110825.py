import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import pearsonr
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
import random
import time
import json
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DetailedFaceQualityAssessment:
    """
    Detailed Face Quality Assessment with individual metric scores
    Includes standardized metrics plus VIF and full FSIM
    NO overall scoring - shows individual metric performance
    """
    
    def __init__(self, output_dir='detailed_metrics_results'):
        # Output directory setup
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Metric descriptions for transparency
        self.metrics_info = {
            'ssim': 'IEEE Structural Similarity Index',
            'psnr': 'ITU-R Peak Signal-to-Noise Ratio (dB)',
            'mae': 'Mean Absolute Error (0-255 scale)',
            'lbp_similarity': 'Local Binary Pattern Histogram Similarity',
            'pixel_correlation': 'Pearson Pixel Correlation',
            'vif': 'Visual Information Fidelity',
            'fsim': 'Feature Similarity Index (full implementation)'
        }
    
    def load_image(self, image_path):
        """Load and convert image to RGB"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def detect_face(self, image, min_face_size=40, scale_factor=0.709):
        """Detect face using MTCNN"""
        try:
            from mtcnn import MTCNN
            detector = MTCNN(min_face_size=min_face_size, scale_factor=scale_factor)
            
            start_time = time.time()
            detections = detector.detect_faces(image)
            detection_time = time.time() - start_time
            
            if not detections:
                return None, 0, detection_time
            
            # Get most confident detection
            best = max(detections, key=lambda x: x['confidence'])
            return best, best['confidence'], detection_time
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, 0, 0
    
    def calculate_individual_metrics(self, original_face, processed_face):
        """
        Calculate individual quality metrics WITHOUT overall scoring
        Returns raw scores for each metric separately
        """
        
        # Ensure same dimensions for comparison
        if original_face.shape != processed_face.shape:
            h, w = original_face.shape[:2]
            processed_face = cv2.resize(processed_face, (w, h))
        
        # Convert to grayscale for analysis
        if len(original_face.shape) == 3:
            orig_gray = cv2.cvtColor(original_face, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(processed_face, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray, proc_gray = original_face, processed_face
        
        metrics = {}
        
        # 1. STRUCTURAL SIMILARITY INDEX (SSIM) - IEEE Standard
        try:
            ssim_score = ssim(orig_gray, proc_gray, data_range=255)
            metrics['ssim'] = ssim_score
        except Exception as e:
            metrics['ssim'] = None
            print(f"SSIM calculation failed: {e}")
        
        # 2. PEAK SIGNAL-TO-NOISE RATIO (PSNR) - ITU-R Standard (raw dB value)
        try:
            mse = np.mean((orig_gray.astype(np.float64) - proc_gray.astype(np.float64)) ** 2)
            if mse == 0:
                metrics['psnr'] = float('inf')  # Perfect match
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                metrics['psnr'] = psnr
        except Exception as e:
            metrics['psnr'] = None
            print(f"PSNR calculation failed: {e}")
        
        # 3. MEAN ABSOLUTE ERROR (MAE) - Raw error value
        try:
            mae = np.mean(np.abs(orig_gray.astype(np.float64) - proc_gray.astype(np.float64)))
            metrics['mae'] = mae
        except Exception as e:
            metrics['mae'] = None
            print(f"MAE calculation failed: {e}")
        
        # 4. LOCAL BINARY PATTERNS (LBP) - Histogram Similarity
        try:
            lbp_orig = local_binary_pattern(orig_gray, 8, 1, method='uniform')
            lbp_proc = local_binary_pattern(proc_gray, 8, 1, method='uniform')
            
            hist_orig, _ = np.histogram(lbp_orig.ravel(), bins=10, range=(0, 9))
            hist_proc, _ = np.histogram(lbp_proc.ravel(), bins=10, range=(0, 9))
            
            if np.sum(hist_orig) > 0 and np.sum(hist_proc) > 0:
                hist_orig = hist_orig / np.sum(hist_orig)
                hist_proc = hist_proc / np.sum(hist_proc)
                
                # Bhattacharyya coefficient
                bhatt_coeff = np.sum(np.sqrt(hist_orig * hist_proc))
                metrics['lbp_similarity'] = bhatt_coeff
            else:
                metrics['lbp_similarity'] = 0
        except Exception as e:
            metrics['lbp_similarity'] = None
            print(f"LBP calculation failed: {e}")
        
        # 5. PEARSON CORRELATION - Raw correlation value
        try:
            if np.std(orig_gray) > 0 and np.std(proc_gray) > 0:
                correlation, _ = pearsonr(orig_gray.ravel(), proc_gray.ravel())
                metrics['pixel_correlation'] = correlation
            else:
                metrics['pixel_correlation'] = 1.0 if np.array_equal(orig_gray, proc_gray) else 0
        except Exception as e:
            metrics['pixel_correlation'] = None
            print(f"Correlation calculation failed: {e}")
        
        # 6. VISUAL INFORMATION FIDELITY (VIF) - Full implementation
        try:
            vif_score = self._calculate_vif(orig_gray, proc_gray)
            metrics['vif'] = vif_score
        except Exception as e:
            metrics['vif'] = None
            print(f"VIF calculation failed: {e}")
        
        # 7. FEATURE SIMILARITY INDEX (FSIM) - Full implementation
        try:
            fsim_score = self._calculate_fsim(orig_gray, proc_gray)
            metrics['fsim'] = fsim_score
        except Exception as e:
            metrics['fsim'] = None
            print(f"FSIM calculation failed: {e}")
        
        return metrics
    
    def _calculate_vif(self, ref, dist):
        """
        Calculate Visual Information Fidelity (VIF)
        Based on Sheikh & Bovik, 2006
        """
        # Convert to float64 for precision
        ref = ref.astype(np.float64)
        dist = dist.astype(np.float64)
        
        # Multi-scale analysis
        scales = 4
        vif_vals = []
        
        for scale in range(scales):
            # Downsample for multi-scale analysis
            if scale > 0:
                ref = self._downsample(ref)
                dist = self._downsample(dist)
            
            if ref.shape[0] < 8 or ref.shape[1] < 8:
                break
            
            # Calculate VIF for this scale
            vif_scale = self._vif_single_scale(ref, dist)
            if vif_scale is not None:
                vif_vals.append(vif_scale)
        
        return np.mean(vif_vals) if vif_vals else 0
    
    def _vif_single_scale(self, ref, dist):
        """Calculate VIF for a single scale"""
        try:
            # Parameters
            sigma_nsq = 2
            
            # Apply Gaussian filtering
            ref_filt = ndimage.gaussian_filter(ref, sigma=1.5)
            dist_filt = ndimage.gaussian_filter(dist, sigma=1.5)
            
            # Calculate local means
            mu1 = ndimage.uniform_filter(ref_filt, size=8)
            mu2 = ndimage.uniform_filter(dist_filt, size=8)
            
            # Calculate local variances and covariance
            mu1_sq = ndimage.uniform_filter(ref_filt * ref_filt, size=8)
            mu2_sq = ndimage.uniform_filter(dist_filt * dist_filt, size=8)
            mu1_mu2 = ndimage.uniform_filter(ref_filt * dist_filt, size=8)
            
            sigma1_sq = mu1_sq - mu1 * mu1
            sigma2_sq = mu2_sq - mu2 * mu2
            sigma12 = mu1_mu2 - mu1 * mu2
            
            # Avoid division by zero
            sigma1_sq = np.maximum(sigma1_sq, 0)
            sigma2_sq = np.maximum(sigma2_sq, 0)
            
            # Calculate VIF
            g = sigma12 / (sigma1_sq + 1e-10)
            sv_sq = sigma2_sq - g * sigma12
            
            g = np.maximum(g, 0)
            sv_sq = np.maximum(sv_sq, sigma_nsq)
            
            # Information content
            num = np.log2(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq))
            den = np.log2(1 + sigma1_sq / sigma_nsq)
            
            vif_map = num / (den + 1e-10)
            
            return np.mean(vif_map)
            
        except Exception:
            return None
    
    def _calculate_fsim(self, ref, dist):
        """
        Calculate Feature Similarity Index (FSIM)
        Full implementation based on Zhang et al., 2011
        """
        try:
            # Calculate phase congruency
            pc1 = self._phase_congruency(ref)
            pc2 = self._phase_congruency(dist)
            
            # Calculate gradient magnitude
            dx1 = cv2.Sobel(ref, cv2.CV_64F, 1, 0, ksize=3)
            dy1 = cv2.Sobel(ref, cv2.CV_64F, 0, 1, ksize=3)
            grad1 = np.sqrt(dx1**2 + dy1**2)
            
            dx2 = cv2.Sobel(dist, cv2.CV_64F, 1, 0, ksize=3)
            dy2 = cv2.Sobel(dist, cv2.CV_64F, 0, 1, ksize=3)
            grad2 = np.sqrt(dx2**2 + dy2**2)
            
            # Constants
            T1 = 0.85  # threshold for PC
            T2 = 160   # threshold for gradient magnitude
            
            # Calculate similarity maps
            # Phase congruency similarity
            pc_sim = (2 * pc1 * pc2 + T1) / (pc1**2 + pc2**2 + T1)
            
            # Gradient magnitude similarity
            grad_sim = (2 * grad1 * grad2 + T2) / (grad1**2 + grad2**2 + T2)
            
            # Combined feature similarity
            feat_sim = pc_sim * grad_sim
            
            # Weighting function
            pc_max = np.maximum(pc1, pc2)
            
            # Calculate FSIM
            fsim = np.sum(feat_sim * pc_max) / (np.sum(pc_max) + 1e-10)
            
            return fsim
            
        except Exception:
            return None
    
    def _phase_congruency(self, img):
        """
        Calculate phase congruency using Log-Gabor filters
        Simplified implementation
        """
        try:
            # Convert to frequency domain
            f_img = np.fft.fft2(img)
            f_img_shift = np.fft.fftshift(f_img)
            
            h, w = img.shape
            
            # Create frequency coordinates
            u = np.arange(-w//2, w//2)
            v = np.arange(-h//2, h//2)
            U, V = np.meshgrid(u, v)
            
            # Calculate radius
            radius = np.sqrt(U**2 + V**2)
            radius[radius == 0] = 1
            
            # Log-Gabor filter parameters
            center_freq = 0.25
            sigma_freq = 0.65
            
            # Create Log-Gabor filter
            log_gabor = np.exp(-0.5 * (np.log(radius / center_freq) / sigma_freq)**2)
            log_gabor[radius < center_freq / 3] = 0
            
            # Apply filter
            filtered = f_img_shift * log_gabor
            
            # Convert back to spatial domain
            result = np.fft.ifft2(np.fft.ifftshift(filtered))
            
            # Calculate phase congruency
            pc = np.abs(result)
            
            # Normalize
            pc = pc / (np.max(pc) + 1e-10)
            
            return pc
            
        except Exception:
            # Fallback to gradient magnitude if phase congruency fails
            dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            return magnitude / (np.max(magnitude) + 1e-10)
    
    def _downsample(self, img):
        """Downsample image by factor of 2"""
        return cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    
    def analyze_single_image(self, image_path, target_size=(160, 160)):
        """Analyze single image and return individual metric scores"""
        # Load image
        original_img = self.load_image(image_path)
        if original_img is None:
            return None
        
        # Detect face
        detection, confidence, detection_time = self.detect_face(original_img)
        if detection is None:
            print(f"  ❌ No face detected in {Path(image_path).name}")
            return None
        
        # Extract face region
        x, y, w, h = detection['box']
        face_region = original_img[y:y+h, x:x+w]
        
        # Create processed version
        processed_face = cv2.resize(face_region, target_size)
        
        # Calculate individual metrics
        metrics = self.calculate_individual_metrics(face_region, processed_face)
        
        # Compile results with individual scores
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'face_confidence': confidence,
            'detection_time': detection_time,
            'original_size': f"{face_region.shape[1]}x{face_region.shape[0]}",
            'processed_size': f"{target_size[0]}x{target_size[1]}",
            
            # Individual metric scores (NO overall score)
            'ssim': metrics.get('ssim'),
            'psnr_db': metrics.get('psnr'),
            'mae': metrics.get('mae'),
            'lbp_similarity': metrics.get('lbp_similarity'),
            'pixel_correlation': metrics.get('pixel_correlation'),
            'vif': metrics.get('vif'),
            'fsim': metrics.get('fsim'),
            
            # Metadata
            'metrics_info': self.metrics_info
        }
        
        return result
    
    def analyze_celebrity_folder(self, folder_path, max_images=20):
        """Analyze all images for one celebrity - showing individual scores"""
        folder_path = Path(folder_path)
        celebrity_name = folder_path.name
        
        print(f"\n🎭 Analyzing: {celebrity_name}")
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        all_images = []
        for ext in image_extensions:
            all_images.extend(folder_path.glob(ext))
        
        if not all_images:
            print(f"  ⚠️  No images found in {celebrity_name}")
            return None
        
        # Sample images
        if len(all_images) > max_images:
            selected_images = random.sample(all_images, max_images)
        else:
            selected_images = all_images
        
        print(f"  Processing {len(selected_images)} images...")
        
        results = []
        successful = 0
        
        for i, img_path in enumerate(selected_images):
            print(f"  [{i+1}/{len(selected_images)}] {img_path.name}...", end=" ")
            
            result = self.analyze_single_image(img_path)
            if result:
                results.append(result)
                successful += 1
                # Show individual metrics instead of overall score
                ssim_val = result['ssim'] if result['ssim'] is not None else 'N/A'
                psnr_val = f"{result['psnr_db']:.1f}dB" if result['psnr_db'] is not None else 'N/A'
                print(f"✅ SSIM:{ssim_val} PSNR:{psnr_val}")
            else:
                print("❌ Failed")
        
        if not results:
            print(f"  No successful analyses for {celebrity_name}")
            return None
        
        # Calculate statistics for each metric separately
        df = pd.DataFrame(results)
        
        # Summary with individual metric averages (NO overall score)
        summary = {
            'celebrity': celebrity_name,
            'total_images': len(selected_images),
            'successful_images': successful,
            'success_rate': successful / len(selected_images),
            'avg_face_confidence': df['face_confidence'].mean(),
            
            # Individual metric averages
            'avg_ssim': df['ssim'].mean() if df['ssim'].notna().any() else None,
            'avg_psnr_db': df['psnr_db'].mean() if df['psnr_db'].notna().any() else None,
            'avg_mae': df['mae'].mean() if df['mae'].notna().any() else None,
            'avg_lbp_similarity': df['lbp_similarity'].mean() if df['lbp_similarity'].notna().any() else None,
            'avg_pixel_correlation': df['pixel_correlation'].mean() if df['pixel_correlation'].notna().any() else None,
            'avg_vif': df['vif'].mean() if df['vif'].notna().any() else None,
            'avg_fsim': df['fsim'].mean() if df['fsim'].notna().any() else None,
            
            # Standard deviations
            'std_ssim': df['ssim'].std() if df['ssim'].notna().any() else None,
            'std_psnr_db': df['psnr_db'].std() if df['psnr_db'].notna().any() else None,
            'std_mae': df['mae'].std() if df['mae'].notna().any() else None,
        }
        
        # Save detailed results
        output_file = self.output_dir / f"{celebrity_name}_individual_metrics.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': summary,
                'individual_results': results,
                'metrics_info': self.metrics_info
            }, f, indent=2, default=str)
        
        # Print individual metric summary
        print(f"\n📊 {celebrity_name} Individual Metrics Summary:")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  SSIM: {summary['avg_ssim']:.3f} ± {summary['std_ssim']:.3f}" if summary['avg_ssim'] else "  SSIM: N/A")
        print(f"  PSNR: {summary['avg_psnr_db']:.1f} ± {summary['std_psnr_db']:.1f} dB" if summary['avg_psnr_db'] else "  PSNR: N/A")
        print(f"  MAE: {summary['avg_mae']:.1f} ± {summary['std_mae']:.1f}" if summary['avg_mae'] else "  MAE: N/A")
        print(f"  VIF: {summary['avg_vif']:.3f}" if summary['avg_vif'] else "  VIF: N/A")
        print(f"  FSIM: {summary['avg_fsim']:.3f}" if summary['avg_fsim'] else "  FSIM: N/A")
        print(f"  💾 Individual scores: {output_file}")
        
        return summary

    def analyze_dataset(self, base_path):
        """Analyze entire dataset showing individual metric performance"""
        base_path = Path(base_path)
        
        if not base_path.exists():
            print(f"❌ Dataset path not found: {base_path}")
            return []
        
        races = ['caucasian', 'chinese', 'indian', 'malay']
        all_summaries = []
        
        for race in races:
            race_path = base_path / race
            if not race_path.exists():
                print(f"⚠️  Race folder not found: {race_path}")
                continue
                
            print(f"\n🌍 Processing {race.title()} celebrities...")
            
            celebrity_folders = [d for d in race_path.iterdir() 
                               if d.is_dir() and not d.name.endswith('_test')]
            
            for celeb_folder in celebrity_folders:
                summary = self.analyze_celebrity_folder(celeb_folder)
                if summary:
                    summary['race'] = race
                    all_summaries.append(summary)
        
        if all_summaries:
            self.save_individual_metrics_summary(all_summaries)
        
        return all_summaries
    
    def save_individual_metrics_summary(self, summaries):
        """Save summary focusing on individual metric performance"""
        df = pd.DataFrame(summaries)
        
        # Individual metric statistics
        metric_stats = {}
        for metric in ['avg_ssim', 'avg_psnr_db', 'avg_mae', 'avg_lbp_similarity', 
                      'avg_pixel_correlation', 'avg_vif', 'avg_fsim']:
            if metric in df.columns:
                valid_data = df[metric].dropna()
                if len(valid_data) > 0:
                    metric_stats[metric] = {
                        'mean': valid_data.mean(),
                        'std': valid_data.std(),
                        'min': valid_data.min(),
                        'max': valid_data.max(),
                        'count': len(valid_data)
                    }
        
        # Best performers for each metric
        best_performers = {}
        for metric in metric_stats.keys():
            if len(df[metric].dropna()) > 0:
                if metric == 'avg_mae':  # Lower is better for MAE
                    best_idx = df[metric].idxmin()
                else:  # Higher is better for others
                    best_idx = df[metric].idxmax()
                best_performers[metric] = {
                    'celebrity': df.loc[best_idx, 'celebrity'],
                    'score': df.loc[best_idx, metric]
                }
        
        summary_data = {
            'dataset_overview': {
                'total_celebrities': len(df),
                'total_successful_images': df['successful_images'].sum(),
                'overall_success_rate': df['success_rate'].mean()
            },
            'individual_metric_statistics': metric_stats,
            'best_performers_by_metric': best_performers,
            'by_race_analysis': df.groupby('race').agg({
                col: ['mean', 'std', 'count'] for col in metric_stats.keys()
            }).round(3).to_dict() if 'race' in df.columns else {},
            'metrics_info': self.metrics_info
        }
        
        # Save comprehensive summary
        summary_file = self.output_dir / 'individual_metrics_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'summary_statistics': summary_data,
                'celebrity_details': summaries
            }, f, indent=2, default=str)
        
        # Save CSV
        csv_file = self.output_dir / 'individual_metrics_data.csv'
        df.to_csv(csv_file, index=False)
        
        # Print individual metric results
        print(f"\n🎉 Individual Metrics Analysis Complete!")
        print(f"📊 Total celebrities: {len(df)}")
        print(f"\n📈 INDIVIDUAL METRIC PERFORMANCE:")
        
        for metric, stats in metric_stats.items():
            print(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
        
        print(f"\n🏆 BEST PERFORMERS BY METRIC:")
        for metric, best in best_performers.items():
            print(f"  {metric.upper()}: {best['celebrity']} ({best['score']:.3f})")
        
        print(f"\n💾 Individual metrics summary: {summary_file}")
        print(f"📊 CSV data: {csv_file}")


def main():
    """Main execution - shows individual metric scores without overall scoring"""
    print("🔬 Detailed Individual Metrics Analysis")
    print("Shows individual SSIM, PSNR, MAE, LBP, Correlation, VIF, FSIM scores")
    print("NO overall scoring - pure individual metric performance")
    print("=" * 70)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Initialize analyzer
    analyzer = DetailedFaceQualityAssessment('individual_metrics_results')
    
    # Run analysis
    print("\n📊 Running Individual Metrics Analysis...")
    try:
        summaries = analyzer.analyze_dataset('./images/celeb-dataset')
        if summaries:
            print(f"✅ Successfully analyzed {len(summaries)} celebrities")
        else:
            print("❌ No celebrities analyzed")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    print(f"\n✅ Individual metrics analysis complete!")
    print(f"📁 All individual metric scores saved to: {analyzer.output_dir}")
    
    return summaries


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")