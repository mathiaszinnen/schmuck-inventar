import argparse
import os
import sys
from schmuck_inventar.detection import YoloImageDetector
from schmuck_inventar.recognition import DummyCardRecognizer, MacOSCardRecognizer
from schmuck_inventar.postprocessor import SchmuckPostProcessor
import platform
import appdirs
from PIL import Image
import yaml
import csv
from tqdm import tqdm

def pipeline(input_dir, output_dir, layout_config):
    print(f"Processing files in directory: {input_dir}")

    app_dir = appdirs.user_data_dir("schmuck_inventar")
    detector = YoloImageDetector(resources_path=os.path.join(app_dir,"detection"))
    if platform.system() == 'Darwin':
        recognizer = MacOSCardRecognizer(layout_config=layout_config)
    else:
        recognizer = DummyCardRecognizer(layout_config=layout_config)
        print("Using dummy recognizer, as this is not a Mac system.")
    
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
    postprocessor = SchmuckPostProcessor(results_csv_raw, final_csv_output)
    postprocessor.postprocess()

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
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    pipeline(input_dir, output_dir, args.layout_config)

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()