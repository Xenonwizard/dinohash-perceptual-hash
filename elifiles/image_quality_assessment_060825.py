import pandas as pd
import numpy as np
import cv2
import json
import time
import random
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# HELPER FUNCTIONS
def safe_percentage_calculation(original, processed):
    """Safely calculate percentage change avoiding division by zero"""
    if np.isnan(original) or np.isinf(original) or np.isnan(processed) or np.isinf(processed):
        return 0.0
    
    if abs(original) < 1e-10:
        return 0.0
    
    result = (original - processed) / original * 100
    
    if np.isinf(result) or np.isnan(result):
        return 0.0
    
    return float(result)

def clean_metrics(metrics):
    """Clean metrics by replacing inf/nan values"""
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isinf(value) or np.isnan(value):
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned

class MTCNNQualityAnalyzer:
    def __init__(self, output_dir='mtcnn_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
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

    def calculate_sharpness(self, img_gray):
        """Calculate sharpness using Sobel operator"""
        try:
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            sharpness = np.sqrt(grad_x**2 + grad_y**2).mean()
            return sharpness if np.isfinite(sharpness) else 0.0
        except:
            return 0.0

    def calculate_quality_metrics(self, original_img, processed_img):
        """Calculate the 4 core quality metrics"""
        try:
            # Resize images to same size for fair comparison
            if original_img.shape != processed_img.shape:
                h, w = original_img.shape[:2]
                processed_img = cv2.resize(processed_img, (w, h))
            
            # Convert to grayscale
            orig_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) if len(original_img.shape) == 3 else original_img
            proc_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY) if len(processed_img.shape) == 3 else processed_img
            
            metrics = {}
            
            # 1. SSIM - Structural similarity
            try:
                metrics['ssim'] = ssim(orig_gray, proc_gray, data_range=255)
            except:
                metrics['ssim'] = 0.0
            
            # 2. PSNR - Peak signal-to-noise ratio
            try:
                psnr_val = psnr(orig_gray, proc_gray, data_range=255)
                metrics['psnr'] = psnr_val if np.isfinite(psnr_val) else 0.0
            except:
                metrics['psnr'] = 0.0
            
            # 3. Edge preservation using Canny
            try:
                edges_orig = cv2.Canny(orig_gray, 50, 150)
                edges_proc = cv2.Canny(proc_gray, 50, 150)
                metrics['edge_preservation'] = ssim(edges_orig, edges_proc, data_range=255)
            except:
                metrics['edge_preservation'] = 0.0
            
            # 4. Sharpness degradation
            try:
                orig_sharpness = self.calculate_sharpness(orig_gray)
                proc_sharpness = self.calculate_sharpness(proc_gray)
                metrics['original_sharpness'] = orig_sharpness
                metrics['processed_sharpness'] = proc_sharpness
                metrics['sharpness_degradation'] = safe_percentage_calculation(orig_sharpness, proc_sharpness)
            except:
                metrics['original_sharpness'] = 0.0
                metrics['processed_sharpness'] = 0.0
                metrics['sharpness_degradation'] = 0.0
            
            return clean_metrics(metrics)
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'ssim': 0.0, 'psnr': 0.0, 'edge_preservation': 0.0,
                'original_sharpness': 0.0, 'processed_sharpness': 0.0, 'sharpness_degradation': 0.0
            }

    def analyze_single_image(self, image_path, target_size=(160, 160)):
        """Analyze a single image"""
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
        
        # Create processed version (what MTCNN pipeline typically does)
        processed_face = cv2.resize(face_region, target_size)
        
        # Calculate quality metrics (original face region vs processed)
        metrics = self.calculate_quality_metrics(face_region, processed_face)
        
        # Compile results
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'face_confidence': confidence,
            'detection_time': detection_time,
            'original_size': f"{face_region.shape[1]}x{face_region.shape[0]}",
            'processed_size': f"{target_size[0]}x{target_size[1]}",
            **metrics
        }
        
        return result

    def analyze_celebrity_folder(self, folder_path, max_images=20):
        """Analyze all images for one celebrity"""
        folder_path = Path(folder_path)
        celebrity_name = folder_path.name
        
        print(f"\n🎭 Analyzing: {celebrity_name}")
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_images = []
        for ext in image_extensions:
            all_images.extend(folder_path.glob(ext))
        
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
                print(f"✅ SSIM: {result['ssim']:.3f}")
            else:
                print("❌ Failed")
        
        if not results:
            print(f"  No successful analyses for {celebrity_name}")
            return None
        
        # Calculate celebrity summary
        df = pd.DataFrame(results)
        summary = {
            'celebrity': celebrity_name,
            'total_images': len(selected_images),
            'successful_images': successful,
            'success_rate': successful / len(selected_images),
            'avg_confidence': df['face_confidence'].mean(),
            'avg_ssim': df['ssim'].mean(),
            'avg_psnr': df['psnr'].mean(),
            'avg_edge_preservation': df['edge_preservation'].mean(),
            'avg_sharpness_degradation': df['sharpness_degradation'].mean(),
            'quality_score': self.calculate_quality_score(df)
        }
        
        # Save detailed results
        output_file = self.output_dir / f"{celebrity_name}_detailed.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': summary,
                'individual_results': results
            }, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 {celebrity_name} Summary:")
        print(f"  Success Rate: {summary['success_rate']:.1%} ({successful}/{len(selected_images)})")
        print(f"  Quality Score: {summary['quality_score']:.2f}/10")
        print(f"  Avg SSIM: {summary['avg_ssim']:.3f}")
        print(f"  Avg PSNR: {summary['avg_psnr']:.1f} dB")
        print(f"  Avg Edge Preservation: {summary['avg_edge_preservation']:.3f}")
        print(f"  Avg Sharpness Degradation: {summary['avg_sharpness_degradation']:.1f}%")
        print(f"  💾 Detailed results: {output_file}")
        
        return summary

    def calculate_quality_score(self, df):
        """Calculate overall quality score (0-10)"""
        # Normalize metrics to 0-1 scale
        ssim_score = df['ssim'].mean()  # Already 0-1
        psnr_score = min(df['psnr'].mean() / 50, 1.0)  # Normalize PSNR (50dB = perfect)
        edge_score = df['edge_preservation'].mean()  # Already 0-1
        sharpness_score = max(0, 1 - df['sharpness_degradation'].mean() / 100)  # Less degradation = better
        
        # Weighted average (you can adjust weights)
        quality_score = (
            ssim_score * 0.4 +          # 40% weight on structural similarity
            edge_score * 0.3 +          # 30% weight on edge preservation  
            psnr_score * 0.2 +          # 20% weight on PSNR
            sharpness_score * 0.1       # 10% weight on sharpness
        ) * 10  # Scale to 0-10
        
        return quality_score

    def analyze_dataset(self, base_path):
        """Analyze entire celebrity dataset"""
        base_path = Path(base_path)
        races = ['caucasian', 'chinese', 'indian', 'malay']
        
        all_summaries = []
        
        for race in races:
            race_path = base_path / race
            if not race_path.exists():
                continue
                
            print(f"\n🌍 Processing {race.title()} celebrities...")
            
            # Get celebrity folders (exclude _test folders)
            celebrity_folders = [d for d in race_path.iterdir() 
                               if d.is_dir() and not d.name.endswith('_test')]
            
            for celeb_folder in celebrity_folders:
                summary = self.analyze_celebrity_folder(celeb_folder)
                if summary:
                    summary['race'] = race
                    all_summaries.append(summary)
        
        # Save overall summary
        if all_summaries:
            self.save_final_summary(all_summaries)
        
        return all_summaries

    def save_final_summary(self, summaries):
        """Save final analysis summary"""
        df = pd.DataFrame(summaries)
        
        # Overall statistics
        overall_stats = {
            'total_celebrities': len(df),
            'total_images_processed': df['successful_images'].sum(),
            'overall_success_rate': df['success_rate'].mean(),
            'overall_quality_score': df['quality_score'].mean(),
            'best_celebrity': df.loc[df['quality_score'].idxmax(), 'celebrity'],
            'worst_celebrity': df.loc[df['quality_score'].idxmin(), 'celebrity'],
            'by_race': df.groupby('race').agg({
                'quality_score': 'mean',
                'avg_ssim': 'mean',
                'success_rate': 'mean'
            }).round(3).to_dict()
        }
        
        # Save summary
        summary_file = self.output_dir / 'final_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'overall_statistics': overall_stats,
                'celebrity_summaries': summaries
            }, f, indent=2, default=str)
        
        # Save CSV for easy analysis
        csv_file = self.output_dir / 'celebrity_scores.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Total celebrities: {len(df)}")
        print(f"📊 Overall quality score: {overall_stats['overall_quality_score']:.2f}/10")
        print(f"🥇 Best performing: {overall_stats['best_celebrity']}")
        print(f"🥉 Needs improvement: {overall_stats['worst_celebrity']}")
        print(f"💾 Final summary: {summary_file}")
        print(f"📈 CSV data: {csv_file}")


# MAIN EXECUTION
def run_analysis():
    random.seed(42)  # For reproducible results
    
    analyzer = MTCNNQualityAnalyzer('celebrity_mtcnn_analysis')
    
    # Analyze the celebrity dataset
    summaries = analyzer.analyze_dataset('./images/celeb-dataset')
    
    return summaries

if __name__ == "__main__":
    results = run_analysis()