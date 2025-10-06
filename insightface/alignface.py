import os
import cv2
import time
import numpy as np
from face_analysis import FaceAnalysis
from skimage import transform as trans


arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

'''
READ THE IMAGES FROM INPUT DIRECTORY
'''

img_dir = r"D:\Test_resources\probes"
# print(img_dir)
img_list = []
for dirs, folds, files in os.walk(img_dir):
    # print(dirs)
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            img_list.append(os.path.join(dirs,file))
###########################################################

'''
FETCH AND PREPARE DETECTION MODEL FROM INSIGHTFACE MODEL ZOO
'''
det = FaceAnalysis(detector_onnx_path="det_10g.onnx")
# det = FaceAnalysis(detector_onnx_path="FaceDetector.onnx")
det.prepare(ctx_id=0, det_size=(640, 640))
############################################################

'''
ALIGNS AND CROP IMAGES

CHECK OUTPUT PATH ACCORDINGLY. CURRENTLY IF INPUT IMAGES ARE AS "/d/Test_resources/probes/black_female/1/",
THE OUTPUT FOLDER ARE CREATED AS "/d/Test_resources/wakefern_ourAlignMatrix/black_female/1/", AND IMAGES WILL 
BE SAVED WITH SAME NAME.
'''

missed_img = []
start = time.time()
for im_path in img_list:
    output_path = im_path.replace("\\probes\\", "\\probes_newCroppedFiles_retinaface\\")
    # output_path = im_path.replace("/probes/", "/wakefern_DetectorTime/")
    img = cv2.imread(im_path)
    # print("Image shape: ",img)
    h, w = img.shape[0], img.shape[1]
    if img is None:
        print("unable to read image")
        continue
    img_copy = img.copy()
    res = det.get(img=img)

    if not res:
        print(f"No detection for {im_path}")
        missed_img.append(im_path)
        continue
    else:
        aligned_img = norm_crop(
                                img = img_copy,
                                landmark=res[0]['kps'],  #  #np.array(res)
                                image_size=224
                                )
        
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        print(output_path)
        cv2.imwrite(output_path, aligned_img)

#########################################################

'''
SAVE THE MISSED IMAGES LIST
'''
with open(r"D:\additionalWork\detectAlignCropRecog_pipeline\missed_images.txt", "w") as f:
    f.write(",".join(missed_img))
print(f"Average time per image (total {len(img_list)} images): {(time.time()-start)/len(img_list)}" )