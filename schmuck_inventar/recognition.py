from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
from schmuck_inventar.utils import download_and_unzip
import platform
import numpy as np
from PIL import Image
import yaml
import random

@dataclass
class OCRResult:
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @staticmethod
    def from_ocrmac_result(result):
        """Construct an OCRResult instance from an ocrmac result."""
        box = result[2]
        return OCRResult(
            text=result[0],
            confidence=result[1],
            x1=box[0],
            y1=1 - box[1],  # top left corner is (0,0) in opencv, but (0,1) in ocrmac
            x2=box[0] + box[2],
            y2=(1 - box[1]) + box[3]
        )

    @staticmethod
    def from_pero_result(result, imwidth, imheight):
        """Construct an OCRResult instance from a pero ocr result."""
        def to_rel_coordinates(polygon, imwidth, imheight):
            """Convert absolute coordinates to relative coordinates."""
            return np.array([[point[0] / imwidth, point[1] / imheight] for point in polygon])
        polygon = to_rel_coordinates(result.polygon,imwidth,imheight)
        return OCRResult(
            text=result.transcription,
            confidence=result.transcription_confidence,
            x1=np.min(polygon[:, 0]),
            y1=np.min(polygon[:, 1]),
            x2=np.max(polygon[:, 0]),
            y2=np.max(polygon[:, 1])
        )

class CardRecognizer(ABC):
    """Abstract base class for different implementations. 
    Takes images of cards along with the path to  layout config yaml as input and returns a dictionary of recognized text per region."""
    def __init__(self, layout_config):
        with open(layout_config, 'r') as file:
            self.layout_dict = yaml.safe_load(file)['regions']

    def _assign_region(self, ocr_result, layout_dict, iou_threshold=0.51):
        """Assigns a region name to the OCR result based on the layout dictionary.
        Assigns a region if more than iou_threshold% of the OCR result area is within the region."""
        def calculate_area(box):
            """Calculate the area of a bounding box."""
            return (box['x2'] - box['x1']) * (box['y2'] - box['y1'])

        def calculate_intersection_area(box1, box2):
            """Calculate the intersection area of two bounding boxes."""
            x_left = max(box1['x1'], box2['x1'])
            y_top = max(box1['y1'], box2['y1'])
            x_right = min(box1['x2'], box2['x2'])
            y_bottom = min(box1['y2'], box2['y2'])

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            return (x_right - x_left) * (y_bottom - y_top)

        for region_name, coordinates in layout_dict.items():
            region_dict = {
            'x1': coordinates[0],
            'y1': coordinates[1],
            'x2': coordinates[2],
            'y2': coordinates[3]
            }
            result_dict = {
            'x1': ocr_result.x1,
            'y1': ocr_result.y1,
            'x2': ocr_result.x2, 
            'y2': ocr_result.y2
            }

            # Calculate the intersection area and the OCR result area
            intersection_area = calculate_intersection_area(region_dict, result_dict)
            result_area = calculate_area(result_dict)

            # Check if more than 80% of the OCR result area is within the region
            if intersection_area / result_area > iou_threshold:
                return region_name

        return None
        

    def _correct_image_orientation(self, image: Image) -> Image:
        """Corrects the orientation of the image based on EXIF data."""
        try:
            exif_data = image._getexif()
            if exif_data is not None:
                orientation = exif_data.get(0x0112)  # EXIF tag for orientation
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except AttributeError:
            print("Image does not have EXIF data.")
        return image

    def recognize(self, image: Image, filename: str) -> dict[str, str]:
        """Recognizes text in the image and assigns it to regions defined in the layout configuration.
        returns a dictionary with region names as keys and recognized text as values."""

        image = self._correct_image_orientation(image)

        try:
            ocr_results = self._do_ocr(image)
        except Exception as e:
            print(f"Warning: OCR failed for {filename}: {e}")
            ocr_results = []
        
        assigned_texts = {"source_file": filename}

        for ocr_result in ocr_results:
            region_name = self._assign_region(ocr_result, self.layout_dict)
            if not region_name:
                print(f"Warning: No region assigned for OCR result: {ocr_result.text}")
                continue
            if region_name in assigned_texts:
                assigned_texts[region_name] += ' ' + ocr_result.text
            else:
                assigned_texts[region_name] = ocr_result.text

        return assigned_texts
        

    @abstractmethod
    def _do_ocr(self, image: Image) -> list[OCRResult]:
        """This method should be implemented by subclasses to perform the actual recognition."""
        raise NotImplementedError("Subclasses must implement this method.")


class MacOSCardRecognizer(CardRecognizer):
    def __init__(self, layout_config):
        if platform.system() != 'Darwin':
            raise ImportError(
                "MacOSCardRecognizer requires macOS and the 'ocrmac' package. "
                "Please run this on a Mac with 'ocrmac' installed."
            )
        try:
            from ocrmac import ocrmac
        except ImportError as e:
            raise ImportError(
                "The 'ocrmac' package is required for MacOSCardRecognizer. "
                "Please install it using 'pip install ocrmac'."
            ) from e
        super().__init__(layout_config)

    def _do_ocr(self, image):
        ocrmac_results = ocrmac.OCR(image).recognize()
        results = []
        for ocrmac_result in ocrmac_results:
            results.append(OCRResult.from_ocrmac_result(ocrmac_result))
        return results

class DummyCardRecognizer(CardRecognizer):
    """Just to be able to develop this on non-Mac systems. Might be extended with a real implementation later."""
    def __init__(self, layout_config):
        import json, os
        examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'resources','example_output.json')
        with open(examples_path, 'r') as f:
            self.example_output = json.load(f)
        super().__init__(layout_config)


    def _do_ocr(self, image):
        """Dummy implementation that returns a fixed dictionary."""
        examples = random.choice(self.example_output)
        results = []
        for example in examples:
            results.append(OCRResult.from_ocrmac_result(example))
        return results

class PeroCardRecognizer(CardRecognizer):
    """Recognition using pero ocr (https://github.com/DCGM/pero-ocr)."""
    def __init__(self, layout_config, app_dir):
        try:
            from pero_ocr.user_scripts.parse_folder import PageParser
            from configparser import ConfigParser
        except ImportError as e:
            raise ImportError(
                "The 'pero_ocr' package is required for PeroCardRecognizer. "
                "Please install it using 'pip install schmuck-inventar[pero]'"
            ) from e
        resources_path = os.path.join(app_dir, "pero_ocr_resources")
        self._prepare_resources(resources_path)
        config = ConfigParser()
        config.read(os.path.join(resources_path, 'config_cpu.ini'))
        current_dir = os.getcwd()
        os.chdir(resources_path)
        self._page_parser = PageParser(config)
        os.chdir(current_dir)
        super().__init__(layout_config)

    def _prepare_resources(self, resources_path):
        """Download and prepare the resources needed for Pero OCR."""
        if os.path.exists(resources_path):
            print(f"Using existing pero ocr resources at {resources_path}")
            return
        print(f"Downloading pero ocr resources to {resources_path}")
        url = "https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ/download/pero_eu_cz_print_newspapers_2022-09-26.zip"
        download_and_unzip(url, resources_path)
            

    def _do_ocr(self, image):
        from pero_ocr.core.layout import PageLayout
        np_image = np.array(image.convert('RGB'))  # Convert PIL image to numpy array, ensure 3 channels
        page_layout = PageLayout(id=0,page_size=(image.width, image.height))
        parsed_results = []
        pero_results = self._page_parser.process_page(np_image,page_layout)
        for result_region in pero_results.regions: 
            for line in result_region.lines:
                parsed_results.append(OCRResult.from_pero_result(line,image.width,image.height))
        return parsed_results

class MistralOCRRecognizer(CardRecognizer):
    """Recognition using Mistral OCR"""
    pass