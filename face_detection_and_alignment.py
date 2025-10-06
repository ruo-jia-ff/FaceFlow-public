import os
import cv2
import yaml
import numpy as np
import onnxruntime as ort
from skimage import transform as trans
from insightface.face_analysis import FaceAnalysis
from utils import preprocess, postprocess
import warnings
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Suppress warnings
ort.set_default_logger_severity(3)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration constants
RETINAFACE_CONFIG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False
}

# Fixed thresholds and parameters
RETINAFACE_CONF_THRESH = 0.5
RETINAFACE_NMS_THRESH = 0.5
ALIGNMENT_SIZE = 224

ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully")
    return config

def init_session(onnx_model_path, device):
    """Initialize ONNX Runtime session"""
    if device.lower() == "cuda":
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    return session

def estimate_norm(lmk, image_size=224):
    """Estimate transformation matrix for face alignment"""
    ratio = image_size / 112.0 if image_size % 112 == 0 else image_size / 128.0
    diff_x = 0 if image_size % 112 == 0 else 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def align_and_crop(img, landmark, image_size=224):
    """Align and crop face using landmarks"""
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

def load_detector(detector_type, device):
    """Load face detector based on type"""
    #print(f"Loading {detector_type} face detector...")
    
    if detector_type == "buffalo":
        detector = FaceAnalysis(detector_onnx_path="models/buffalo_detector.onnx")
        ctx_id = 0 if device.lower() == "cuda" else -1
        detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("Buffalo detector loaded successfully")
        return {"type": "buffalo", "detector": detector}
    
    elif detector_type == "retinaface":
        sess = init_session("models/retinaface_detector.onnx", device)
        print("RetinaFace detector loaded successfully")
        return {"type": "retinaface", "detector": sess}
    
    raise ValueError(f"Unknown detector type: {detector_type}")

def detect_faces(detector_model, img, device):
    """Detect faces in image using specified detector"""
    print("Performing detection...")
    if detector_model["type"] == "buffalo":
        faces = detector_model["detector"].get(img)
        return faces
    
    else:  # retinaface
        sess = detector_model["detector"]
        img_in, scale, resize = preprocess(img, [640, 640], device)
        outputs = sess.run(None, {sess.get_inputs()[0].name: img_in})
        dets = postprocess(
            RETINAFACE_CONFIG, img, outputs, scale, resize,
            RETINAFACE_CONF_THRESH, RETINAFACE_NMS_THRESH, 
            device, [640, 640]
        )
        if dets.shape[0] == 0:
            return []
        return [{"bbox": det[:4], "kps": det[5:].reshape(5, 2), "det_score": det[4]} for det in dets]

def process_single_image(img_path, detector_model, device):
    """Process single image: detect faces, align and crop"""
    print(f"\nProcessing image: {img_path}")
    
    # Validate image path
    if not os.path.exists(img_path):
        print(f"ERROR: Image file not found: {img_path}")
        return
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not read image: {img_path}")
        return
    print(f"Image loaded successfully - Shape: {img.shape}")

    # Create output directory
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = f"output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # Save original image
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    cv2.imwrite(original_path, img)
    print(f"Original image saved: {original_path}")

    # Face detection
    print("\nStarting face detection...")
    detections = detect_faces(detector_model, img, device)
    
    if not detections:
        print("No faces detected in the image")
        return
    
    print(f"Successfully detected {len(detections)} face(s)")
    
    # Process each detected face
    detection_info = []
    
    for idx, det in enumerate(detections):
        
        # Extract bbox and landmarks based on detector type
        if detector_model["type"] == "buffalo":
            bbox = det.bbox
            landmarks = det.kps
            det_score = det.det_score
        else:  # retinaface
            bbox = det["bbox"]
            landmarks = det["kps"]
            det_score = det["det_score"]
                
        # Creating detection image for saving 
        x1, y1, x2, y2 = map(int, bbox)
        cropped_face = img[y1:y2, x1:x2]
        
        # Save detection
        cropped_path = os.path.join(output_dir, f"face_{idx}_cropped.jpg")
        cv2.imwrite(cropped_path, cropped_face)
        print(f"Detection saved: {cropped_path}")
        
        print(f"\nProcessing detected face {idx + 1}/{len(detections)}")
        # Face alignment and cropping
        print("Performing face alignment and cropping...")
        aligned_img = align_and_crop(img, landmarks, ALIGNMENT_SIZE)
        
        # Save aligned face
        aligned_path = os.path.join(output_dir, f"face_{idx}_aligned_{ALIGNMENT_SIZE}x{ALIGNMENT_SIZE}.jpg")
        cv2.imwrite(aligned_path, aligned_img)
        print(f"Aligned face saved: {aligned_path}")
        
        # Store face information
        face_info = {
            "face_idx": idx,
            "bbox": bbox.tolist(),
            "landmarks": landmarks.tolist(),
            "det_score": float(det_score),
            "detection_output": cropped_path,
            "aligned_path": aligned_path
        }
        detection_info.append(face_info)
    
    # Save detection metadata
    print("\nSaving metadata...")
    metadata_path = os.path.join(output_dir, "detection_results.json")
    metadata = {
        "image_path": img_path,
        "image_shape": img.shape,
        "detector_type": detector_model["type"],
        "device": device,
        "alignment_size": ALIGNMENT_SIZE,
        "num_faces_detected": len(detections),
        "faces": detection_info
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
def main():
    """Main processing pipeline"""
    # Load configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print("Please create a config.yaml file with image_path, detector, and device")
        return
    
    config = load_config(config_path)
    
    # Validate configuration
    required_keys = ['image_path', 'detector', 'device']
    for key in required_keys:
        if key not in config:
            print(f"ERROR: Missing required config parameter: {key}")
            return

    # Load detector
    detector_model = load_detector(config['detector'], config['device'])
    
    # Process the image
    process_single_image(config['image_path'], detector_model, config['device'])

if __name__ == "__main__":
    main()