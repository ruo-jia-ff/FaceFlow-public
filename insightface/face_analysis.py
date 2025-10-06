import os.path as osp
from .common import Face
from .retinaface import RetinaFace


class FaceAnalysis:
    def __init__(self, detector_onnx_path: str, **kwargs):
        """
        Initialize FaceAnalysis with a local RetinaFace ONNX model.
        """
        if not osp.exists(detector_onnx_path):
            raise FileNotFoundError(f"Detector model not found: {detector_onnx_path}")

        print(f"Loading detector: {detector_onnx_path}")
        self.det_model = RetinaFace(model_file=detector_onnx_path)
        self.det_model.prepare(ctx_id=0, **kwargs)

    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=(640, 640)):
        """Configure detection threshold and input size."""
        self.det_model.prepare(ctx_id, det_thresh=det_thresh, input_size=det_size)

    def get(self, img, max_num=0, det_metric='default'):
        """Run face detection on an image and return Face objects."""
        bboxes, kpss = self.det_model.detect(img,
                                             input_size=self.det_model.input_size,
                                             max_num=max_num,
                                             metric=det_metric)
        if bboxes.shape[0] == 0:
            return []

        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            faces.append(face)
        return faces

    def draw_on(self, img, faces):
        """Draw detected faces on the image."""
        import cv2
        dimg = img.copy()
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                for l in range(kps.shape[0]):
                    color = (0, 255, 0) if l in (0, 3) else (0, 0, 255)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        return dimg
