import argparse
from PIL import Image
import json
import os
import random
import time
from schmuck_inventar.utils import pil_image_to_base64
from io import BytesIO

import httpx
from mistralai import File, Mistral
from dotenv import load_dotenv


def create_client():
    """
    Create a Mistral client using the API key from environment variables.
    Returns:
        Mistral: An instance of the Mistral client.
    """
    load_dotenv()
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def generate_random_string(start, end):
    """
    Generate a random string of variable length.

    Args:
        start (int): Minimum length of the string.
        end (int): Maximum length of the string.

    Returns:
        str: A randomly generated string.
    """
    length = random.randrange(start, end)
    return ' '.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

def print_stats(batch_job):
    """
    Print the statistics of the batch job.

    Args:
        batch_job: The batch job object containing job statistics.
    """
    print(f"Total requests: {batch_job.total_requests}")
    print(f"Failed requests: {batch_job.failed_requests}")
    print(f"Successful requests: {batch_job.succeeded_requests}")
    print(
        f"Percent done: {round((batch_job.succeeded_requests + batch_job.failed_requests) / batch_job.total_requests, 4) * 100}")


def create_input_file(client, num_samples):
    from pydantic import BaseModel
    from mistralai.extra import response_format_from_pydantic_model
    """
    Create an input file for the batch job.

    Args:
        client (Mistral): The Mistral client instance.
        num_samples (int): Number of samples to generate.

    Returns:
        File: The uploaded input file object.
    """
    buffer = BytesIO()
    image_path = '/home/mathias/data/schmuck/test/SCH1063.jpeg'
    image = Image.open(image_path)
    image_base64 = pil_image_to_base64(image) 

    class StructuredOutput(BaseModel):
        Gegenstand: str
        Inventarnummer: str

    properties = {
                    "Gegenstand": {
                        "title": "Gegenstand",
                        "type": "string"
                    },
                    "Inventarnummer": {
                        "title": "Inventarnummer",
                        "type": "string"
                    },
                    "NonExistentField": {
                        "title": "NonExistentField",
                        "type": "string"
                    }
    }

    schema = {
                "properties": properties,
                "title": "Document Annotation",
                "type": "object",
                "additionalProperties": False
                
    }

    json_schema = {
            "schema": schema,
            "name": "document_annotation",
            "strict": True
    }

    document_annotation_format = {
        "type": "json_schema",
        "json_schema": json_schema
        }
    
    document_annotation_format_str = json.dumps(document_annotation_format, indent=2)

    response_format = response_format_from_pydantic_model(StructuredOutput)
    ocr_request = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_base64}",
        },
        "document_annotation_format": document_annotation_format,
        # "include_image_base64": True
    }
    for idx in range(num_samples):
        request = {
            "custom_id": str(idx),
            "body": ocr_request
        }
        buffer.write(json.dumps(request).encode("utf-8"))
        buffer.write("\n".encode("utf-8"))
    return client.files.upload(file=File(file_name="file.jsonl", content=buffer.getvalue()), purpose="batch")


def run_batch_job(client, input_file, model):
    """
    Run a batch job using the provided input file and model.

    Args:
        client (Mistral): The Mistral client instance.
        input_file (File): The input file object.
        model (str): The model to use for the batch job.

    Returns:
        BatchJob: The completed batch job object.
    """
    batch_job = client.batch.jobs.create(
        input_files=[input_file.id],
        model=model,
        endpoint="/v1/ocr",
        metadata={"job_type": "testing"}
    )

    while batch_job.status in ["QUEUED", "RUNNING"]:
        batch_job = client.batch.jobs.get(job_id=batch_job.id)
        print_stats(batch_job)
        time.sleep(1)

    print(f"Batch job {batch_job.id} completed with status: {batch_job.status}")
    return batch_job


def download_file(client, file_id, output_path):
    """
    Download a file from the Mistral server.

    Args:
        client (Mistral): The Mistral client instance.
        file_id (str): The ID of the file to download.
        output_path (str): The path where the file will be saved.
    """
    if file_id is not None:
        print(f"Downloading file to {output_path}")
        output_file = client.files.download(file_id=file_id)
        with open(output_path, "w") as f:
            for chunk in output_file.stream:
                f.write(chunk.decode("utf-8"))
        print(f"Downloaded file to {output_path}")


def main(num_samples, success_path, error_path, model):
    """
    Main function to run the batch job.

    Args:
        num_samples (int): Number of samples to process.
        success_path (str): Path to save successful outputs.
        error_path (str): Path to save error outputs.
        model (str): Model name to use.
    """
    client = create_client()
    input_file = create_input_file(client, num_samples)
    print(f"Created input file {input_file}")

    batch_job = run_batch_job(client, input_file, model)
    print(f"Job duration: {batch_job.completed_at - batch_job.created_at} seconds")
    download_file(client, batch_job.error_file, error_path)
    download_file(client, batch_job.output_file, success_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mistral AI batch job")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--success_path", type=str, default="output.jsonl", help="Path to save successful outputs")
    parser.add_argument("--error_path", type=str, default="error.jsonl", help="Path to save error outputs")
    parser.add_argument("--model", type=str, default="codestral-latest", help="Model name to use")

    args = parser.parse_args()

    # main(args.num_samples, args.success_path, args.error_path, args.model)
    main(2,args.success_path, args.error_path, 'mistral-ocr-latest')  # For testing purposes, set num_samples to 1