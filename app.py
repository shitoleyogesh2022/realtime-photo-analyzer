import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision.models import ResNet50_Weights
import io
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class AIPhotoAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scene_classifier = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.scene_classifier.eval()
        self.scene_classifier.to(self.device)
        
        self.categories = [f"Category_{i}" for i in range(1000)]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def analyze_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        results = {
            'scene_type': self.detect_scene(image),
            'composition_score': self.analyze_composition(image),
            'lighting_quality': self.analyze_lighting(image),
            'technical_quality': self.assess_technical_quality(image)
        }
        return results

    def detect_scene(self, image):
        try:
            img_tensor = self.transform(image)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.scene_classifier(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            top3_prob, top3_catid = torch.topk(probabilities, 3)
            return [(self.categories[idx], prob.item()) for prob, idx in zip(top3_catid, top3_prob)]
        except Exception as e:
            print(f"Scene detection error: {e}")
            return [("Unknown", 1.0)]

    def analyze_composition(self, image):
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        symmetry_score = 1 - np.mean(np.abs(left_half - right_half)) / 255.0
        
        h_thirds = h // 3
        w_thirds = w // 3
        thirds_score = 0
        for i in range(1, 3):
            for j in range(1, 3):
                region = edges[
                    max(0, i * h_thirds - 20):min(h, i * h_thirds + 20),
                    max(0, j * w_thirds - 20):min(w, j * w_thirds + 20)
                ]
                thirds_score += np.mean(region > 0)
        
        composition_score = edge_density * 0.3 + symmetry_score * 0.3 + thirds_score * 0.4
        
        return {
            'score': float(min(max(composition_score, 0), 1)),
            'edge_density': float(edge_density),
            'symmetry': float(symmetry_score),
            'thirds_alignment': float(thirds_score)
        }

    def analyze_lighting(self, image):
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        brightness = float(np.mean(l_channel) / 255.0)
        contrast = float(np.std(l_channel) / 255.0)
        color_temp = float(np.mean(b_channel) - np.mean(a_channel)) / 255.0
        dynamic_range = float(np.percentile(l_channel, 95) - np.percentile(l_channel, 5)) / 255.0
        
        l_hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
        overexposed = float(np.sum(l_hist[-10:]) / np.sum(l_hist))
        underexposed = float(np.sum(l_hist[:10]) / np.sum(l_hist))
        color_balance = 1.0 - abs(float(np.mean(a_channel) - np.mean(b_channel))) / 255.0
        
        lighting_score = (
            brightness * 0.25 +
            contrast * 0.25 +
            dynamic_range * 0.2 +
            (1 - overexposed) * 0.1 +
            (1 - underexposed) * 0.1 +
            color_balance * 0.1
        )
        
        return {
            'score': float(min(max(lighting_score, 0), 1)),
            'brightness': brightness,
            'contrast': contrast,
            'color_temperature': color_temp,
            'dynamic_range': dynamic_range,
            'overexposed': overexposed,
            'underexposed': underexposed,
            'color_balance': color_balance
        }

    def assess_technical_quality(self, image):
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.mean(np.abs(laplacian)) / 255.0)
        
        noise = float(np.std(gray) / 255.0)
        
        local_var = cv2.blur(gray.astype(float)**2, (5,5)) - cv2.blur(gray.astype(float), (5,5))**2
        detail_score = float(np.mean(local_var) / 255.0)
        
        technical_score = sharpness * 0.4 + (1 - noise) * 0.3 + detail_score * 0.3
        
        return {
            'score': float(min(max(technical_score, 0), 1)),
            'sharpness': sharpness,
            'noise_level': noise,
            'detail_level': detail_score
        }

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.analyzer = AIPhotoAnalyzer()
        self.last_analysis = None
        self.current_frame = None
        self.frame_skip = 30
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.current_frame = img.copy()
        
        height, width = img.shape[:2]
        overlay = img.copy()
        
        for i in range(1, 3):
            cv2.line(overlay, (width*i//3, 0), (width*i//3, height), (255,255,255), 1)
            cv2.line(overlay, (0, height*i//3), (width, height*i//3), (255,255,255), 1)
        
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            try:
                self.last_analysis = self.analyzer.analyze_image(self.current_frame)
            except Exception as e:
                print(f"Analysis error: {e}")
        
        return img

def save_captured_photo(frame, analysis):
    os.makedirs('captured_photos', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"captured_photos/photo_{timestamp}.jpg"
    analysis_path = f"captured_photos/analysis_{timestamp}.txt"
    
    cv2.imwrite(image_path, frame)
    
    with open(analysis_path, 'w') as f:
        f.write("Photography Analysis Report\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Scene Detection:\n")
        for scene, prob in analysis['scene_type']:
            f.write(f"- {scene}: {prob:.2%}\n")
        
        f.write("\nComposition Analysis:\n")
        comp = analysis['composition_score']
        f.write(f"- Overall Score: {comp['score']:.2%}\n")
        f.write(f"- Edge Definition: {comp['edge_density']:.2%}\n")
        f.write(f"- Symmetry: {comp['symmetry']:.2%}\n")
        f.write(f"- Rule of Thirds: {comp['thirds_alignment']:.2%}\n")
        
        f.write("\nLighting Analysis:\n")
        light = analysis['lighting_quality']
        f.write(f"- Overall Score: {light['score']:.2%}\n")
        f.write(f"- Brightness: {light['brightness']:.2%}\n")
        f.write(f"- Contrast: {light['contrast']:.2%}\n")
        f.write(f"- Color Balance: {light['color_balance']:.2%}\n")
        
        f.write("\nTechnical Quality:\n")
        tech = analysis['technical_quality']
        f.write(f"- Overall Score: {tech['score']:.2%}\n")
        f.write(f"- Sharpness: {tech['sharpness']:.2%}\n")
        f.write(f"- Noise Level: {tech['noise_level']:.2%}\n")
        f.write(f"- Detail Level: {tech['detail_level']:.2%}\n")
    
    return image_path, analysis_path

def display_analysis(analysis):
    if not analysis:
        return
    
    st.write("**Scene Detection:**")
    for scene, prob in analysis['scene_type']:
        st.write(f"- {scene}: {prob:.2%}")
    
    st.write("**Composition Analysis:**")
    comp = analysis['composition_score']
    st.write(f"- Overall Score: {comp['score']:.2%}")
    st.write(f"- Edge Definition: {comp['edge_density']:.2%}")
    st.write(f"- Symmetry: {comp['symmetry']:.2%}")
    st.write(f"- Rule of Thirds: {comp['thirds_alignment']:.2%}")
    
    st.write("**Lighting Analysis:**")
    light = analysis['lighting_quality']
    st.write(f"- Overall Score: {light['score']:.2%}")
    st.write(f"- Brightness: {light['brightness']:.2%}")
    st.write(f"- Contrast: {light['contrast']:.2%}")
    st.write(f"- Color Balance: {light['color_balance']:.2%}")
    
    temp = light['color_temperature']
    temp_text = "Neutral" if -0.2 <= temp <= 0.2 else ("Cool" if temp < -0.2 else "Warm")
    st.write(f"- Color Temperature: {temp_text}")

def generate_suggestions(analysis):
    if not analysis:
        return []
    
    suggestions = []
    light = analysis['lighting_quality']
    comp = analysis['composition_score']
    tech = analysis['technical_quality']
    
    if light['brightness'] < 0.4:
        suggestions.append(("lighting", "Scene is too dark. Try increasing exposure or adding more light."))
    elif light['brightness'] > 0.8:
        suggestions.append(("lighting", "Scene is too bright. Consider reducing exposure."))
        
    if comp['score'] < 0.6:
        suggestions.append(("composition", "Try applying the rule of thirds for better composition."))
        
    if tech['sharpness'] < 0.4:
        suggestions.append(("technical", "Image is not sharp. Check focus and camera stability."))
    
    return suggestions

def main():
    st.set_page_config(page_title="Real-time Photography Assistant", layout="wide")
    
    st.title("📸 Real-time Photography Assistant")
    st.write("Use your camera to get instant photography feedback!")

    col1, col2 = st.columns(2)
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="photography_assistant",
            video_transformer_factory=VideoTransformer,
            async_transform=True,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        )

    with col2:
        if st.button("📸 Capture Photo", key="capture"):
            if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.current_frame is not None:
                try:
                    image_path, analysis_path = save_captured_photo(
                        webrtc_ctx.video_transformer.current_frame,
                        webrtc_ctx.video_transformer.last_analysis
                    )
                    
                    st.success(f"Photo captured and saved!")
                    
                    captured_image = cv2.imread(image_path)
                    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
                    st.image(captured_image, caption="Captured Photo", use_container_width=True)
                    
                    with open(image_path, 'rb') as file:
                        st.download_button(
                            label="Download Photo",
                            data=file,
                            file_name=os.path.basename(image_path),
                            mime="image/jpeg"
                        )
                    
                    with open(analysis_path, 'rb') as file:
                        st.download_button(
                            label="Download Analysis Report",
                            data=file,
                            file_name=os.path.basename(analysis_path),
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"Error capturing photo: {str(e)}")
            else:
                st.warning("Camera not ready. Please wait a moment and try again.")

    if webrtc_ctx.video_transformer:
        analysis_placeholder = st.empty()
        suggestions_placeholder = st.empty()
        
        if webrtc_ctx.video_transformer.last_analysis:
            with analysis_placeholder.container():
                display_analysis(webrtc_ctx.video_transformer.last_analysis)
                
                suggestions = generate_suggestions(webrtc_ctx.video_transformer.last_analysis)
                if suggestions:
                    st.write("**Suggestions for Improvement:**")
                    for stype, message in suggestions:
                        if stype == "lighting":
                            st.warning(message)
                        elif stype == "composition":
                            st.info(message)
                        else:
                            st.error(message)

    st.sidebar.title("About")
    st.sidebar.write("""
    This real-time photography assistant helps you:
    - Analyze scene composition
    - Check lighting conditions
    - Monitor technical quality
    - Get instant feedback
    - Capture and save photos with analysis
    
    Use the camera preview to compose your shot, 
    then click 'Capture Photo' to save it with 
    detailed analysis.
    """)

    # Display saved photos section
    if os.path.exists('captured_photos'):
        st.sidebar.markdown("---")
        st.sidebar.title("Recent Captures")
        photo_files = [f for f in os.listdir('captured_photos') if f.startswith('photo_')]
        photo_files.sort(reverse=True)  # Show most recent first
        
        for photo_file in photo_files[:5]:  # Show last 5 photos
            photo_path = os.path.join('captured_photos', photo_file)
            analysis_path = photo_path.replace('photo_', 'analysis_').replace('.jpg', '.txt')
            
            if os.path.exists(photo_path):
                img = Image.open(photo_path)
                st.sidebar.image(img, caption=photo_file, use_container_width=True)
                
                if os.path.exists(analysis_path):
                    with open(analysis_path, 'rb') as f:
                        st.sidebar.download_button(
                            label=f"Download Analysis for {photo_file}",
                            data=f,
                            file_name=os.path.basename(analysis_path),
                            mime="text/plain"
                        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please ensure camera permissions are granted and try refreshing.")
