import os
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from skimage import transform as trans
from insightface.face_analysis import FaceAnalysis
from production_alignAndCrop import production_cropping
from utils import preprocess, postprocess   # keep your existing functions


# ---------------- CONFIG ---------------- #
DEVICE = "cuda"  # or "cpu"

RETINAFACE_CONFIG = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}
RETINAFACE_CONF_THRESH = 0.5
RETINAFACE_NMS_THRESH = 0.5

ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


# ---------------- MODEL INIT ---------------- #
def initialize_model(model_type: str) -> str:
    if model_type == "clearface":
        return "models/clear_face_recognition_model.onnx"
    elif model_type == "maskedface":
        return "models/masked_face_recognition_model.onnx"
    else:
        raise ValueError("Invalid model_type. Choose 'clearface' or 'maskedface'.")


def init_session(onnx_model_path: str) -> ort.InferenceSession:
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_model_path, providers=providers)


# ---------------- PREPROCESSING ---------------- #
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_embedding(session: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = get_transform()(img).unsqueeze(0).numpy()
    embedding = session.run([session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: tensor})[0]
    return embedding.flatten()


# ---------------- FACE DETECTION ---------------- #
def buffalo_inference(img: np.ndarray):
    det = FaceAnalysis(detector_onnx_path="models/buffalo_detector.onnx")
    det.prepare(ctx_id=0, det_size=(640, 640))
    results = det.get(img=img)

    if not results:
        print("⚠️ No face detected with Buffalo.")
        return []

    # return [res["kps"] for res in results]   # multiple 5-point landmarks
    return results


def retinaface_inference(img_raw: np.ndarray):
    session = init_session("models/retinaface_detector.onnx")
    img, scale, resize = preprocess(img_raw, [640, 640], DEVICE)
    outputs = session.run(None, {session.get_inputs()[0].name: img})

    dets = postprocess(
        RETINAFACE_CONFIG, img, outputs, scale, resize,
        RETINAFACE_CONF_THRESH, RETINAFACE_NMS_THRESH,
        DEVICE, input_size=[640, 640]
    )

    if dets.shape[0] == 0:
        print("⚠️ No face detected with RetinaFace.")
        return []

    # return [d[5:].reshape(5, 2) for d in dets]  # list of landmarks
    return dets


# ---------------- ALIGNMENT ---------------- #
def estimate_norm(lmk: np.ndarray, image_size=112, mode="arcface") -> np.ndarray:
    assert lmk.shape == (5, 2)
    ratio = image_size / 112.0 if image_size % 112 == 0 else image_size / 128.0
    diff_x = 0 if image_size % 112 == 0 else 8.0 * ratio
    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]


def align_and_crop(img: np.ndarray, landmark: np.ndarray, image_size=224) -> np.ndarray:
    M = estimate_norm(landmark, image_size, mode="arcface")
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


# ---------------- MAIN PIPELINE ---------------- #
def main(args):
    recog_model = initialize_model(args.model_type)
    recog_sess = init_session(recog_model)

    img = cv2.imread(args.img_path)
    if img is None:
        raise FileNotFoundError(f"❌ Could not read image: {args.img_path}")

    # detection model
    if args.detector == "buffalo":
        detection_list = buffalo_inference(img)
    else:
        detection_list = retinaface_inference(img)

    if detection_list is None or len(detection_list) == 0:
        print("❌ No faces found.")
        return

    save_dir = os.path.splitext(args.img_path)[0] + "_faces"
    os.makedirs(save_dir, exist_ok=True)

    for idx, detection in enumerate(detection_list):
        # Landmark
        if args.detector == "buffalo":
            bbox, landmarks = detection['bbox'], detection['kps']
        else:
            bbox, landmarks = detection[:4], detection[5:].reshape(5,2)
        aligned_img = align_and_crop(img, landmarks, image_size=224)
        ## PRODUCTION ALIGN AND CROP FOR CLASSIFIEER
        production_cropped = production_cropping(img, bbox) 

        # Save aligned face
        aligned_path = os.path.join(save_dir, f"face_{idx}_aligned.jpg")
        cv2.imwrite(aligned_path, aligned_img)

        # Save aligned face @ 112
        aligned_path_112 = os.path.join(save_dir, f"face_{idx}_aligned_112.jpg")
        cv2.imwrite(aligned_path_112, production_cropped)

        # Extract embedding
        emb = get_embedding(recog_sess, aligned_img)
        np.savetxt(os.path.join(save_dir, f"face_{idx}_embedding.txt"), emb, fmt="%.8f")

        print(f"✅ Face {idx}: embedding shape = {emb.shape}, saved to {aligned_path}")


# ---------------- ENTRY POINT ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-face embedding generator with alignment")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--detector", type=str, choices=["buffalo", "retinaface"],
                        default="buffalo", help="Face detector to use")
    parser.add_argument("--model_type", type=str, choices=["clearface", "maskedface"],
                        required=True, help="Recognition model type")
    args = parser.parse_args()
    main(args)
