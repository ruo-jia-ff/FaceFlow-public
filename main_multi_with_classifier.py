import os
import cv2
import csv
import base64
import argparse
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from skimage import transform as trans
from insightface.face_analysis import FaceAnalysis
from production_alignAndCrop import production_cropping
from utils import preprocess, postprocess
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import warnings

# ---------------- SUPPRESS WARNINGS ---------------- #
ort.set_default_logger_severity(3)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- CONFIG ---------------- #
DEVICE = "cuda"
RETINAFACE_CONFIG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False
}
RETINAFACE_CONF_THRESH = 0.5
RETINAFACE_NMS_THRESH = 0.5

ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

MASK_CLASSES = ["Mask", "Non-Mask"]
FACE_CLASSES = ["Face-Only", "Non-Face"]

# ---------------- TRANSFORMS ---------------- #
classifier_transform = Compose([
    Resize(112, 112),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

recognition_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ---------------- MODEL HELPERS ---------------- #
def initialize_model(model_type: str) -> str:
    if model_type == "clearface":
        return "models/clear_face_recognition_model.onnx"
    elif model_type == "maskedface":
        return "models/masked_face_recognition_model.onnx"
    raise ValueError(f"Invalid model_type: {model_type}")

def init_session(onnx_model_path: str) -> ort.InferenceSession:
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_model_path, providers=providers)

def get_embedding(session: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = recognition_transform(img_rgb).unsqueeze(0).numpy()
    emb = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0]
    return emb.flatten()

# ---------------- CLASSIFIER HELPERS ---------------- #
def preprocess_classifier(img: np.ndarray):
    augmented = classifier_transform(image=img)
    return augmented['image'].unsqueeze(0).numpy()

def run_classifier(face_img: np.ndarray, session: ort.InferenceSession,
                   face_threshold=0.49, mask_threshold=0.33):
    input_tensor = preprocess_classifier(face_img)
    mask_logits, face_logits = session.run(None, {"input": input_tensor})
    mask_probs = F.softmax(torch.tensor(mask_logits[0]), dim=0).numpy()
    face_probs = F.softmax(torch.tensor(face_logits[0]), dim=0).numpy()

    return {
        "face_prediction": "Face-Only" if face_probs[0] >= face_threshold else "Non-Face",
        "mask_prediction": "Mask" if mask_probs[0] >= mask_threshold else "Non-Mask",
        "face_probs": dict(zip(FACE_CLASSES, face_probs)),
        "mask_probs": dict(zip(MASK_CLASSES, mask_probs))
    }

# ---------------- ALIGNMENT ---------------- #
def estimate_norm(lmk: np.ndarray, image_size=112) -> np.ndarray:
    ratio = image_size / 112.0 if image_size % 112 == 0 else image_size / 128.0
    diff_x = 0 if image_size % 112 == 0 else 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def align_and_crop(img: np.ndarray, landmark: np.ndarray, image_size=224) -> np.ndarray:
    M = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

# ---------------- IMAGE HELPERS ---------------- #
def encode_image_base64(img_path):
    if not img_path or not os.path.exists(img_path):
        return ""
    with Image.open(img_path).convert("RGB") as img:
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_image_paths(input_dir):
    exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(r, f) for r, _, fs in os.walk(input_dir) for f in fs if f.lower().endswith(exts)]

# ---------------- DETECTOR WRAPPERS ---------------- #
def load_detector(detector_type):
    if detector_type == "buffalo":
        detector = FaceAnalysis(detector_onnx_path="models/buffalo_detector.onnx")
        detector.prepare(ctx_id=0, det_size=(640, 640))
        return {"type": "buffalo", "detector": detector}
    elif detector_type == "retinaface":
        sess = init_session("models/retinaface_detector.onnx")
        return {"type": "retinaface", "detector": sess}
    raise ValueError(f"Unknown detector type: {detector_type}")

def detect_faces(detector_model, img: np.ndarray):
    if detector_model["type"] == "buffalo":
        return detector_model["detector"].get(img)
    else:  # retinaface
        sess = detector_model["detector"]
        img_in, scale, resize = preprocess(img, [640, 640], DEVICE)
        outputs = sess.run(None, {sess.get_inputs()[0].name: img_in})
        dets = postprocess(RETINAFACE_CONFIG, img, outputs, scale, resize,
                           RETINAFACE_CONF_THRESH, RETINAFACE_NMS_THRESH, DEVICE, [640, 640])
        if dets.shape[0] == 0:
            return []
        return [{"bbox": det[:4], "kps": det[5:].reshape(5, 2), "det_score": det[4]} for det in dets]

# ---------------- PROCESS IMAGE ---------------- #
def process_image(img_path, classifier_session, recog_sessions, detector_model, save_root):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return []

    detections = detect_faces(detector_model, img)
    if not detections:
        print(f"❌ No faces found in {img_path}")
        return []

    save_dir = os.path.join(save_root, os.path.splitext(os.path.basename(img_path))[0])

    os.makedirs(save_dir, exist_ok=True)
    results = []

    for idx, det in enumerate(detections):
        bbox, landmarks = det["bbox"], det["kps"]
        aligned_img = align_and_crop(img, landmarks, 224)
        production_cropped = production_cropping(img, bbox)

        classifier_result = run_classifier(production_cropped, classifier_session)
        aligned_path_112 = os.path.join(save_dir, f"face_{idx}_aligned_112.jpg")
        cv2.imwrite(aligned_path_112, production_cropped)

        if classifier_result["face_prediction"] == "Non-Face":
            print(f"⚠️ Detection {idx} classified as Non-Face. Skipping recognition.")
            results.append({**classifier_result, "image_path": img_path, "face_idx": idx,
                            "aligned_path": None, "aligned_path_112": aligned_path_112})
            continue

        # Recognition
        recog_type = "maskedface" if classifier_result["mask_prediction"] == "Mask" else "clearface"
        print("Recognition Model Selected: ", recog_type)
        if recog_type not in recog_sessions:
            recog_sessions[recog_type] = init_session(initialize_model(recog_type))
        emb = get_embedding(recog_sessions[recog_type], aligned_img)

        aligned_path = os.path.join(save_dir, f"face_{idx}_aligned.jpg")
        cv2.imwrite(aligned_path, aligned_img)
        np.savetxt(os.path.join(save_dir, f"face_{idx}_embedding.txt"), emb, fmt="%.8f")

        results.append({**classifier_result, "image_path": img_path, "face_idx": idx,
                        "aligned_path": aligned_path, "aligned_path_112": aligned_path_112})

    return results

# ---------------- MAIN ---------------- #
def main(args):
    classifier_session = init_session("models/mask_classifier_with_face_branch_with_extra_convLayer_BGR_chk51_epoch3.onnx")
    recog_sessions = {}
    detector_model = load_detector(args.detector)

    image_paths = get_image_paths(args.input_dir)
    if not image_paths:
        print("No images found in directory.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    html_report = os.path.join(args.output_dir, "report.html")

    with open(summary_csv, "w", newline="") as f, open(html_report, "w", encoding="utf-8") as html:
        writer = csv.writer(f)
        writer.writerow(["image_path", "face_idx", "mask_prediction", "mask_Prob_Mask", "mask_Prob_Non-Mask",
                         "face_prediction", "face_Prob_Face", "face_Prob_Non-Face", "aligned_path", "aligned_path_112"])

        html.write("<html><head><title>Face Classifier Report</title></head><body><h2>Results</h2><table border='1'>"
                   "<tr><th>Image</th><th>Face #</th><th>Mask</th><th>Face</th></tr>")

        for img_path in image_paths:
            results = process_image(img_path, classifier_session, recog_sessions, detector_model, args.output_dir)
            for r in results:
                writer.writerow([
                    r.get("image_path"), r.get("face_idx"),
                    r.get("mask_prediction", ""), r.get("mask_probs", {}).get("Mask", ""),
                    r.get("mask_probs", {}).get("Non-Mask", ""), r.get("face_prediction", ""),
                    r.get("face_probs", {}).get("Face-Only", ""), r.get("face_probs", {}).get("Non-Face", ""),
                    r.get("aligned_path", ""), r.get("aligned_path_112", "")
                ])

                aligned_b64_224 = encode_image_base64(r.get("aligned_path"))
                aligned_b64_112 = encode_image_base64(r.get("aligned_path_112"))
                html.write(f"<tr><td>"
                           f"{'<b>Aligned 224x224:</b><br><img src=\"data:image/jpeg;base64,'+aligned_b64_224+'\" height=\"100\"/><br>' if aligned_b64_224 else ''}"
                           f"{'<b>Classifier 112x112:</b><br><img src=\"data:image/jpeg;base64,'+aligned_b64_112+'\" height=\"100\"/><br>' if aligned_b64_112 else ''}"
                           f"</td><td>{r.get('face_idx')}</td>"
                           f"<td>{r.get('mask_prediction','')}</td><td>{r.get('face_prediction','')}</td></tr>")

        html.write("</table></body></html>")

    print(f"✅ Processing complete. Summary CSV: {summary_csv}, HTML report: {html_report}")

# ---------------- ENTRY POINT ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directory-based multi-face embedding generator with HTML report")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder")
    parser.add_argument("--detector", type=str, choices=["buffalo","retinaface"], default="buffalo")
    args = parser.parse_args()
    print(args)
    main(args)
