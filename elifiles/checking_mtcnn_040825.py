import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import math
from PIL import Image
import os

class MTCNNVisualizer:
    def __init__(self):
        """Initialize MTCNN detector"""
        # Suppress MTCNN progress bars
        import sys
        from contextlib import redirect_stdout
        import io
        
        with redirect_stdout(io.StringIO()):
            self.detector = MTCNN()
    
    def visualize_face_detection(self, image_path, save_path=None):
        """Visualize MTCNN face detection process step by step"""
        
        # Load and prepare image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(img_rgb)
        
        if not faces:
            print("No faces detected!")
            return None
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MTCNN Face Processing Pipeline - {os.path.basename(image_path)}', fontsize=16)
        
        # 1. Original image
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        
        # 2. Face detection with bounding boxes
        img_with_boxes = img_rgb.copy()
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Add confidence score
            cv2.putText(img_with_boxes, f'Face {i+1}: {confidence:.3f}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        axes[0, 1].imshow(img_with_boxes)
        axes[0, 1].set_title(f'2. Face Detection ({len(faces)} faces found)')
        axes[0, 1].axis('off')
        
        # Process the best face (highest confidence)
        best_face = max(faces, key=lambda x: x['confidence'])
        x, y, w, h = best_face['box']
        keypoints = best_face['keypoints']
        
        # 3. Landmarks visualization
        img_with_landmarks = img_rgb.copy()
        
        # Draw face box
        cv2.rectangle(img_with_landmarks, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Draw landmarks
        landmark_colors = {
            'left_eye': (255, 0, 0),    # Red
            'right_eye': (0, 255, 0),   # Green  
            'nose': (0, 0, 255),        # Blue
            'mouth_left': (255, 255, 0), # Yellow
            'mouth_right': (255, 0, 255) # Magenta
        }
        
        for landmark, point in keypoints.items():
            color = landmark_colors.get(landmark, (255, 255, 255))
            cv2.circle(img_with_landmarks, tuple(map(int, point)), 5, color, -1)
            # Add label
            cv2.putText(img_with_landmarks, landmark, 
                       (int(point[0])+10, int(point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        axes[0, 2].imshow(img_with_landmarks)
        axes[0, 2].set_title('3. Facial Landmarks Detection')
        axes[0, 2].axis('off')
        
        # 4. Face extraction with padding
        padding = max(w, h) // 8  # 12.5% padding
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(img_rgb.shape[1] - x_padded, w + 2*padding)
        h_padded = min(img_rgb.shape[0] - y_padded, h + 2*padding)
        
        face_region = img_rgb[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
        
        axes[1, 0].imshow(face_region)
        axes[1, 0].set_title(f'4. Extracted Face Region\n({w_padded}x{h_padded} with padding)')
        axes[1, 0].axis('off')
        
        # 5. Face alignment using eye landmarks
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Adjust keypoints relative to cropped face
        adjusted_left_eye = (left_eye[0] - x_padded, left_eye[1] - y_padded)
        adjusted_right_eye = (right_eye[0] - x_padded, right_eye[1] - y_padded)
        
        # Calculate center point for rotation
        center_x = (adjusted_left_eye[0] + adjusted_right_eye[0]) / 2
        center_y = (adjusted_left_eye[1] + adjusted_right_eye[1]) / 2
        
        # Apply rotation
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        aligned_face = cv2.warpAffine(face_region, M, (face_region.shape[1], face_region.shape[0]), 
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        axes[1, 1].imshow(aligned_face)
        axes[1, 1].set_title(f'5. Aligned Face\n(Rotated {angle:.1f}° to level eyes)')
        axes[1, 1].axis('off')
        
        # 6. Final normalized face
        normalized_face = cv2.resize(aligned_face, (200, 200), interpolation=cv2.INTER_CUBIC)
        
        # Apply LAB color space normalization
        lab = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2LAB)
        normalized_face_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        axes[1, 2].imshow(normalized_face_lab)
        axes[1, 2].set_title('6. Final Normalized Face\n(200x200, LAB normalized)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        # Return processing details
        return {
            'faces_detected': len(faces),
            'best_confidence': best_face['confidence'],
            'face_box': (x, y, w, h),
            'rotation_angle': angle,
            'keypoints': keypoints,
            'final_face': normalized_face_lab
        }
    
    def compare_multiple_extractions(self, image_paths, save_path=None):
        """Compare face extractions from multiple images"""
        
        extracted_faces = []
        processing_info = []
        
        for img_path in image_paths:
            result = self.extract_face_for_comparison(img_path)
            if result:
                extracted_faces.append(result['face'])
                processing_info.append(result)
        
        if not extracted_faces:
            print("No faces could be extracted from the provided images")
            return
        
        # Create comparison visualization
        n_images = len(extracted_faces)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Face Extraction Comparison', fontsize=16)
        
        # Top row: Original images with detection boxes
        for i, (img_path, info) in enumerate(zip(image_paths, processing_info)):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw detection box
            x, y, w, h = info['face_box']
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(img_rgb, f'Conf: {info["confidence"]:.3f}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            axes[0, i].imshow(img_rgb)
            axes[0, i].set_title(f'Original {i+1}\n{os.path.basename(img_path)}')
            axes[0, i].axis('off')
        
        # Bottom row: Extracted and normalized faces
        for i, (face, info) in enumerate(zip(extracted_faces, processing_info)):
            axes[1, i].imshow(face)
            axes[1, i].set_title(f'Extracted Face {i+1}\nRotated: {info["angle"]:.1f}°')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        plt.show()
        
        return extracted_faces, processing_info
    
    def extract_face_for_comparison(self, image_path):
        """Extract face similar to the validation framework"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector.detect_faces(img_rgb)
            
            if not faces:
                return None
            
            # Get best face
            best_face = max(faces, key=lambda x: x['confidence'])
            
            if 'keypoints' not in best_face:
                return None
            
            keypoints = best_face['keypoints']
            
            if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
                return None
            
            # Extract face with padding
            x, y, w, h = best_face['box']
            padding = max(w, h) // 8
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_rgb.shape[1] - x, w + 2*padding)
            h = min(img_rgb.shape[0] - y, h + 2*padding)
            
            face_region = img_rgb[y:y+h, x:x+w]
            
            # Align face
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dy, dx))
            
            # Adjust keypoints
            adj_left_eye = (left_eye[0] - x, left_eye[1] - y)
            adj_right_eye = (right_eye[0] - x, right_eye[1] - y)
            
            center_x = (adj_left_eye[0] + adj_right_eye[0]) / 2
            center_y = (adj_left_eye[1] + adj_right_eye[1]) / 2
            
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            aligned_face = cv2.warpAffine(face_region, M, (face_region.shape[1], face_region.shape[0]))
            
            # Normalize
            normalized_face = cv2.resize(aligned_face, (200, 200))
            lab = cv2.cvtColor(normalized_face, cv2.COLOR_RGB2LAB)
            final_face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return {
                'face': final_face,
                'confidence': best_face['confidence'],
                'face_box': best_face['box'],
                'angle': angle,
                'keypoints': keypoints
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

# Usage examples
def demo_mtcnn_extraction():
    """Demo function to show MTCNN extraction"""
    
    # Initialize visualizer
    visualizer = MTCNNVisualizer()
    
    # Example 1: Single image analysis
    # Replace with actual image path from your dataset
    image_path = "./images/ronnychieng/download (8).jpeg"  # Update this path
    
    if os.path.exists(image_path):
        print("Analyzing single image...")
        result = visualizer.visualize_face_detection(
            image_path, 
            save_path="mtcnn_processing_steps.png"
        )
        
        if result:
            print(f"Processing completed:")
            print(f"  Faces detected: {result['faces_detected']}")
            print(f"  Best confidence: {result['best_confidence']:.3f}")
            print(f"  Rotation applied: {result['rotation_angle']:.1f}°")
    
    # Example 2: Compare multiple images of same person
    same_person_images = [
        "./images/ronnychieng/download (10).jpeg",  # Update these paths
        "./images/ronnychieng/download (12).jpeg",
        "./images/ronnychieng/download (14).jpeg"
    ]
    
    # Filter existing images
    existing_images = [img for img in same_person_images if os.path.exists(img)]
    
    if len(existing_images) >= 2:
        print(f"\nComparing {len(existing_images)} images of the same person...")
        faces, info = visualizer.compare_multiple_extractions(
            existing_images,
            save_path="face_extraction_comparison.png"
        )
        
        print("Extraction results:")
        for i, inf in enumerate(info):
            print(f"  Image {i+1}: confidence={inf['confidence']:.3f}, rotation={inf['angle']:.1f}°")
    else:
        print("Need at least 2 existing images for comparison demo")

if __name__ == "__main__":
    demo_mtcnn_extraction()