
import os
import tarfile
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, set_start_method
import json
import io
import threading
import fcntl
from prompts import VISUAL_CHAIN_OF_THOUGHT

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

torch.distributed.is_available()


VISUAL_CHAIN_OF_THOUGHT2 = """Your task is to examine the visual information carefully and describe your process of understanding and solving the problem.

Observation: Begin by describing what you see in the image. Break down the visual elements into understandable segments. For instance, if it's a picture of a street, identify the key components like cars, buildings, people, street signs, etc. If it's a graph, start by outlining its type, the axes, and the data it presents.

Detailed Reasoning: Delve deeper into your analysis. This is where the chain of thought becomes critical. If you're looking at a scene, consider the relationships between elements. Why might that person be running? What does the traffic signal indicate? For graphs or data-driven images, analyze trends, outliers, and correlations. Explain your thought process in a step-by-step manner.

Visual References: As you explain, make visual references. Draw arrows, circles, or use highlights in the image to pinpoint exactly what you're discussing. These annotations should accompany your verbal reasoning, adding clarity to your explanations."""

device = torch.set_default_device("cuda")
torch.manual_seed(1234)
CHUNK_SIZE = 20 # Number of images to extract at a time
tar_file_number = 1 #Change this manually to match corresponding dataset

# Path to the tar file
tar_file_path = f"./openimages/train_{tar_file_number}.tar.gz"
#File to store the list of processed images
responses_file = f"openimage-responses-{tar_file_number}.jsonl"

# Instantiate the QwenVLMultiModal model
#model = QwenVLMultiModal(
#    model_name="Qwen/Qwen-VL-Chat",
#    device="cuda"
#)
def distribute_chunks(members, chunk_size, world_size):
    # split members into chunks
    chunks = [members[i:i + chunk_size] for i in range(0, len(members), chunk_size)]
    # Dsitribute chunks across GPUs
    gpu_chunks = [[] for _ in range(world_size)]
    for i, chunk in enumerate(chunks):
        gpu_chunks[i % world_size].append(chunk)
    return gpu_chunks

def setup_model_and_tokenizer(device):

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", fast_attention2=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True
        ).eval()
    return model, tokenizer
processed_images = []
unprocessed_images = []
# Load the list of already processed iamges from the responses file
if os.path.exists(responses_file):
    with open(responses_file, "r") as file:
        for line in file:
            try:
                entry = json.loads(line)
                image_file_name = entry["image_path"].split('/')[-1]
                if 'response' in entry:
                    processed_images.append(image_file_name)
                else:
                    unprocessed_images.append(image_file_name)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")
def get_next_id(responses_file):
    max_id = 0
    if os.path.exists(responses_file):
        with open(responses_file, "r") as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    entry_id = entry.get("id", 0)
                    if entry_id > max_id:
                        max_id = entry_id
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")
                    continue
    return max_id + 1

def get_tar_members(tar_file_path, valid_extensions=(".png", ".jpg", ".jpeg")):
    """
    Extracts members from a tar file that match the valid extensions.

    Parameters:
    - tar_file_path: Path to the tar file.
    - valid_extensions: Tuple of file extensions to include.

    Returns:
    - List of tarfile.TarInfo objects for files matching the valid extensions.
    """
    with tarfile.open(tar_file_path, "r:gz") as tar:
        # Filter members by checking if the file extension is in valid_extensions
        members = [m for m in tar.getmembers() if m.name.lower().endswith(valid_extensions) and m.name not in processed_images]
    return members

def add_image(image_path, next_id, tar_file_number):
    data = {
        "image_path": image_path,
        "id": next_id,
        "dataset": tar_file_number
    }
    if os.path.exists(responses_file):
        with open(responses_file, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data,file,indent=4)
    else:
        with open(responses_file, "w") as file:
            json.dump([data], file, indent=4)


def write_outputs_from_queue(output_queue, responses_file):
    with open(responses_file, "a") as f:
        while True:
            output = output_queue.get()
            if output == "DONE":
                break
            f.write(json.dumps(output) + "\n")

def process_images(rank,chunks, output_queue, world_size, tar_file_path):
    # Process the extracted files
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model, tokenizer = setup_model_and_tokenizer(device)
    with tarfile.open(tar_file_path, "r:gz") as tar:
        for chunk in chunks[rank]: # Distribute files to process based on rank
            with tempfile.TemporaryDirectory() as tmpdirname:
                for member in chunk:
                    tar.extract(member, path=tmpdirname)
                    extracted_file_path = os.path.join(tmpdirname, member.name)
                    print(f"processing {extracted_file_path}")
                    extracted_file_name = os.path.basename(extracted_file_path)
                    path_parts = extracted_file_path.split('/')
                    tar_file_number = path_parts[-2].split('_')[-1]
                    image_file = f'train_{tar_file_number}/{extracted_file_name}'
                    print(f'image_file {image_file}')
                    with torch.cuda.device(device):
                        query_data = [
                    {"image": extracted_file_path},
                    {"text": VISUAL_CHAIN_OF_THOUGHT2}
            ]
                        query = tokenizer.from_list_format(query_data)
                        caption_response = model.chat(tokenizer, query=query, history=None)
                        print(caption_response)
                        next_id = get_next_id(responses_file)
                        output_queue.put({"id": next_id, "response": caption_response, "image_path": image_file, "dataset": tar_file_path})
    #save the response
    if rank == 0:
        for _ in range(world_size):
            output_queue.put("DONE")
    dist.destroy_process_group
                


def main():
    if os.path.exists(responses_file):
        with open(responses_file, "r") as file:
            for line in file:
                try:
                    entry = json.loads(line)
                    if 'response' in entry:
                        processed_images.append(image_file_name)
                    else:
                        unprocessed_images.append(image_file_name)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")
    set_start_method("spawn")
    world_size = torch.cuda.device_count()
    output_queue = Queue()
    members = get_tar_members(tar_file_path)
    gpu_chunks = distribute_chunks(members, CHUNK_SIZE, world_size)
    writer_process = Process(
            target=write_outputs_from_queue, args=(output_queue, responses_file)
    )
    writer_process.start()
    # extracted_information = match extracted_files with the object for responses["image_path"] in responses_file
    mp.spawn(
            process_images,
            args=(gpu_chunks, output_queue, world_size, tar_file_path),
            nprocs=world_size,
           join=True,
    )
    output_queue.put("DONE")
    writer_process.join()

if __name__ == "__main__":
    main()

