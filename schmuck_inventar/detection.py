from ultralytics import YOLO
from schmuck_inventar.utils import download_and_unzip
import glob
import shutil
import torch
import cv2
import os

class YoloImageDetector:
    def __init__(self, resources_path, chunk_size=50,
                 weights_url='https://faubox.rrze.uni-erlangen.de/dl/fi9iK4rseupfrrTeXWQUGP/weights.zip'):
        self._prepare_resources(resources_path, weights_url)
        self.model = YOLO(os.path.join(resources_path, 'yolov8.pt'))
        self.chunk_size = chunk_size
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f'Detection running on {self.device}')

    def _prepare_resources(self, resources_path, weights_url):
        if os.path.exists(os.path.join(resources_path, 'yolov8.pt')):
            return
        print(f'Downloading YOLO weights to {os.path.abspath(resources_path)}...')
        download_and_unzip(weights_url, resources_path)

    def _batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:ndx+n]

    def _move_crops(self, yolo_name, out_dir):
        yolo_output = os.path.join(out_dir, yolo_name)
        crop_dir = os.path.join(out_dir, 'images')
        if not os.path.isdir(crop_dir):
            os.makedirs(crop_dir)
        for file in glob.glob(os.path.join(yolo_output, '**', '*.jpg'), recursive=True):
            fn = os.path.basename(file)
            shutil.move(file, os.path.join(crop_dir, fn))
        shutil.rmtree(yolo_output)
        print(f'Detected images moved to \033[1m{crop_dir}\033[0m')

    def parse_directory(self, input_dir, crop_dir='tmp', output_base_dir='output'):
        image_exts = ['.jpg', '.jpeg']
        images_to_process = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if os.path.splitext(fn)[1] in image_exts]
        n_chunks = len(images_to_process) // self.chunk_size + 1
        i = 1
        for img_chunk in self._batch(images_to_process, self.chunk_size):
            print(f'Detecting images in chunk {i}/{n_chunks}..')
            self.model.predict(img_chunk, save_crop=True, device=self.device, name=crop_dir, project=output_base_dir)
            self._move_crops(crop_dir, output_base_dir)
            i += 1

    def detect(self,image):
        results = self.model.predict(image, device=self.device, max_det=1)
        return results

    def crop_and_save(self, detections, out_dir, name):
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        for i, result in enumerate(detections):
            if result.boxes is None or len(result.boxes) == 0:
                continue # no image found
            # Sort boxes by confidence in descending order
            x1,y1,x2,y2 = sorted(result.boxes, key=lambda box: box.conf, reverse=True)[0].xyxy[0].flatten().int().tolist()
            crop = result.orig_img[y1:y2,x1:x2]
            cv2.imwrite(os.path.join(out_dir, name),crop)


class DummyDetector:
    def __init__(self, chunk_size=50):
        self.chunk_size = chunk_size
        print(f'Dummy detector doing nothing, OCR only.')

    def parse_directory(self, input_dir, crop_dir='tmp', output_base_dir='output'):
        pass

    def detect(self, image):
        return []

    def crop_and_save(self, detections, out_dir, name):
        pass