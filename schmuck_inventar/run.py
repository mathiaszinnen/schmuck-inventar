import argparse
import os
import sys
from schmuck_inventar.detection import Detector, YoloImageDetector, DummyDetector
from schmuck_inventar.recognition import CardRecognizer, DummyCardRecognizer, MacOSCardRecognizer, PeroCardRecognizer, MistralOCRRecognizer
from schmuck_inventar.postprocessor import PostProcessor, SchmuckPostProcessor, BenchmarkingPostProcessor
import platform
import appdirs
from PIL import Image
import yaml
import csv
from tqdm import tqdm

def pipeline(input_dir, output_dir, layout_config, detector: Detector, recognizer: CardRecognizer, postprocessor: PostProcessor):
    print(f"Processing files in directory: {input_dir}")
    
    # Load layout configuration
    with open(layout_config, 'r') as config_file:
        config_file = yaml.safe_load(config_file)
        layout_keys = config_file['regions'].keys() 

    results_csv_raw = os.path.join(output_dir, 'results_raw.csv')
    os.makedirs(output_dir, exist_ok=True)

    with open(results_csv_raw, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=['source_file'] + list(layout_keys))
        csv_writer.writeheader()

        for filename in tqdm(os.listdir(input_dir)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                continue
            file_path = os.path.join(input_dir, filename)
            image = Image.open(file_path)
            detections = detector.detect(image)
            detector.crop_and_save(detections, os.path.join(output_dir, 'images'), filename)
            results = recognizer.recognize(image, filename)

            # Write raw results to CSV
            row = {'source_file': filename}
            row.update({key: results.get(key, '') for key in layout_keys})
            csv_writer.writerow(row)
        print(f"Raw extraction results written to {results_csv_raw}")
    
    final_csv_output = os.path.join(output_dir, 'results.csv')
    postprocessor.postprocess(results_csv_raw, final_csv_output)


def instantiate_recognizer(engine, layout_config, app_dir):
    if engine == 'pero':
        print("Using PeroCardRecognizer, ensure you have 'pero-ocr' installed.")
        return PeroCardRecognizer(layout_config=layout_config, app_dir=app_dir)
    elif engine == 'ocrmac':
        if not platform.system() == 'Darwin':
            raise ImportError(
                "MacOSCardRecognizer requires macOS and the 'ocrmac' package. "
                "Please run this on a Mac with 'ocrmac' installed."
            )
        print("Using MacOSCardRecognizer, ensure you are running this on a Mac with 'ocrmac' installed.")
        return MacOSCardRecognizer(layout_config=layout_config)
    elif engine == 'mistral':
        return MistralOCRRecognizer(layout_config=layout_config)
    elif engine == 'dummy':
        # Dummy recognizer for rapid development
        return DummyCardRecognizer(layout_config=layout_config)
    else: # automatic selection based on platform
        if platform.system() == 'Darwin':
            print("Using MacOSCardRecognizer, ensure you are running this on a Mac with 'ocrmac' installed.")
            return MacOSCardRecognizer(layout_config=layout_config)
        else:
            print("Using PeroCardRecognizer, ensure you have 'pero-ocr' installed.")
            return PeroCardRecognizer(layout_config=layout_config, app_dir=app_dir)


def instantiate_detector(eval_mode, app_dir):
    if eval_mode:
        print("Using DummyDetector for evaluation mode.")
        return DummyDetector()
    else:
        print("Using YoloImageDetector for production mode.")
        return YoloImageDetector(resources_path=os.path.join(app_dir, "detection"))

def instantiate_postprocessor(eval_mode):
    if eval_mode:
        print("Using BenchmarkingPostProcessor for evaluation mode.")
        return BenchmarkingPostProcessor()
    else:
        print("Using SchmuckPostProcessor for production mode.")
        return SchmuckPostProcessor()

def main():
    parser = argparse.ArgumentParser(description="Process input directory for Schmuck Inventar.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory containing files to process."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "output"),
        help="Path to the output directory. Defaults to './output' in the current working directory."
    )
    parser.add_argument(
        '--layout_config',
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'regions.yaml'),
        required=False,
        help="Path to the layout configuration file (YAML). Defaults to 'config/regions.yaml' relative to the project root."
    )
    parser.add_argument(
        '--ocr_engine',
        type=str,
        choices=['auto','ocrmac', 'pero', 'dummy','mistral'],
        default='auto',
        help="Recognition engine to use: 'ocrmac', 'pero', or 'dummy'. Default is 'auto', " \
        "which resolves to ocrmac when running on a mac system and pero else."
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help="If set, runs the pipeline in evaluation mode."
    )
    args = parser.parse_args()

    app_dir = appdirs.user_data_dir("schmuck_inventar")
    recognizer = instantiate_recognizer(args.ocr_engine, args.layout_config, app_dir)
    detector = instantiate_detector(args.eval, app_dir)
    postprocessor = instantiate_postprocessor(args.eval)

    input_dir = args.input_dir
    output_dir = args.output_dir

    pipeline(input_dir, output_dir, args.layout_config, detector, recognizer, postprocessor)

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()