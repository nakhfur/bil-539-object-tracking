import os
import glob
import time
import math
import cv2
import xml.etree.ElementTree as ET
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_ROOT = os.path.join(PROJECT_ROOT, "images")
ANNOS_ROOT  = os.path.join(PROJECT_ROOT, "annotations")

SEQUENCE_NAME = "MVI_20011"  

IMAGES_DIR = os.path.join(IMAGES_ROOT, SEQUENCE_NAME)
XML_PATH   = os.path.join(ANNOS_ROOT, SEQUENCE_NAME + ".xml")


def load_frame_paths(images_dir):
    paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    paths.sort()
    return paths


def load_annotations(xml_path, target_id=1):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for frame in root.iter('frame'):
        box_for_this_frame = None
        target_list = frame.find('target_list')
        if target_list is not None:
            for target in target_list.iter('target'):
                if int(target.attrib.get('id', -1)) == target_id:
                    box = target.find('box')
                    if box is not None:
                        x = float(box.attrib['left'])
                        y = float(box.attrib['top'])
                        w = float(box.attrib['width'])
                        h = float(box.attrib['height'])
                        box_for_this_frame = (x, y, w, h)
                    break

        boxes.append(box_for_this_frame)

    return boxes


def box_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return None

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    iou = interArea / float(areaA + areaB - interArea)
    return iou


def box_center(box):
    if box is None:
        return None
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx, cy


def center_location_error(box_pred, box_gt):
    if box_pred is None or box_gt is None:
        return None

    cx_p, cy_p = box_center(box_pred)
    cx_g, cy_g = box_center(box_gt)

    return math.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2)


if __name__ == "__main__":
    print("Images dir:", IMAGES_DIR)
    print("XML path :", XML_PATH)

    frame_paths = load_frame_paths(IMAGES_DIR)
    print("Toplam frame sayisi:", len(frame_paths))
    print("Ilk 5 frame:")
    for p in frame_paths[:5]:
        print("  ", os.path.basename(p))

    
    gt_boxes = load_annotations(XML_PATH, target_id=3)
    print("Toplam GT frame sayisi:", len(gt_boxes))

    assert len(frame_paths) == len(gt_boxes)


    start_idx = None
    for i, b in enumerate(gt_boxes):
        if b is not None:
            start_idx = i
            break

    if start_idx is None:
        raise RuntimeError("Hiç bbox bulunamadı, XML/target_id kontrol et.")

    print("Takip baslangic frame index:", start_idx)

    first_frame = cv2.imread(frame_paths[start_idx])
    x, y, w, h = gt_boxes[start_idx]
    x, y, w, h = int(x), int(y), int(w), int(h)

    template = first_frame[y:y+h, x:x+w]
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    
    iou_list = []
    cle_list = []

    t0 = time.time()

    for idx, img_path in enumerate(frame_paths):
        frame = cv2.imread(img_path)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        px, py = max_loc  
        ph, pw = template_gray.shape
        pred_box = (px, py, pw, ph)

        gt_box = gt_boxes[idx]

        iou = box_iou(pred_box, gt_box) if gt_box is not None else None
        cle = center_location_error(pred_box, gt_box) if gt_box is not None else None

        if iou is not None:
            iou_list.append(iou)
        if cle is not None:
            cle_list.append(cle)

        # Görselleştirme
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 255, 0), 2)  # prediction (yeşil)
        if gt_box is not None:
            gx, gy, gw, gh = map(int, gt_box)
            cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 2)  # GT (kırmızı)

        cv2.putText(frame, f"Frame: {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Baseline Template Matching", frame)

        key = cv2.waitKey(1)  
        if key == 27:  # ESC ile çık
            break

    total_time = time.time() - t0
    num_frames = len(iou_list)  
    fps = num_frames / total_time if total_time > 0 else 0.0

    cv2.destroyAllWindows()

    
    mean_iou = np.mean(iou_list) if len(iou_list) > 0 else 0.0
    mean_cle = np.mean(cle_list) if len(cle_list) > 0 else 0.0

    print(f"Ortalama IoU: {mean_iou:.4f}")
    print(f"Ortalama CLE: {mean_cle:.2f} piksel")
    print(f"FPS (yaklasik): {fps:.2f}")
