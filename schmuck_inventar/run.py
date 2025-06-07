import argparse
import os
import sys
from schmuck_inventar.detection import YoloImageDetector
from schmuck_inventar.recognition import DummyCardRecognizer, MacOSCardRecognizer
import platform
import appdirs
import cv2
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

    for filename in tqdm(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        image = cv2.imread(file_path)
        detections = detector.detect(image)
        detector.crop_and_save(detections, os.path.join(output_dir,'images'), filename)
        results = recognizer.recognize(image,filename)
        print('debug')
        print(results)


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
        default=os.path.join(os.getcwd(), 'config', 'regions.yaml'),
        required=False,
        help="Path to the layout configuration file (YAML). Defaults to 'config/regions.yaml' in the current working directory."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    pipeline(input_dir, output_dir, args.layout_config)

    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing files in directory: {input_dir}")
    # Add your processing logic here

if __name__ == "__main__":
    main()