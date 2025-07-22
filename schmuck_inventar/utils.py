import requests
import json
import shutil
import zipfile
import io
import os
from tqdm import tqdm
import base64
from PIL import Image

def download_and_unzip(url, extract_to):
    response = requests.get(url, stream=True)  # Set stream to True to retrieve the content in chunks
    total_size_in_bytes = int(response.headers.get('content-length', 0))  # Get the total size of the file
    block_size = 1024  # Block size to read in loop (1 Kibibyte)
    os.makedirs(extract_to, exist_ok=True)

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)  # Setup progress bar
    with io.BytesIO() as file_stream:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file_stream.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        file_stream.seek(0)  # Go to the beginning of the file-like stream
        with zipfile.ZipFile(file_stream) as zip_file:
            for member in zip_file.infolist():
                if member.is_dir():
                    continue
                target_path = os.path.join(extract_to, os.path.basename(member.filename))
                with zip_file.open(member) as source_file, open(target_path, 'wb') as target_file:
                    shutil.copyfileobj(source_file, target_file)

def pil_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def dict_from_markdown(md_string: str) -> dict[str,str]:
    md_lines = md_string.split('\n')

    pass

def test():
    output_path = '/home/mathias/development/schmuck-inventar/output.jsonl'
    with open(output_path) as f:
        output_lines = list(f)
    output = json.loads(output_lines[0])
    md_string = output['response']['body']['pages'][0]['markdown']
    result = dict_from_markdown(md_string)
    print(md_string)

test()