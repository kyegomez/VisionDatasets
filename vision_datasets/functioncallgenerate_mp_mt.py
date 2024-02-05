import torch
import json
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, set_start_method
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os

os.environ["MASTER_ADDR"] = (
    "localhost"  # Use the appropriate master address
)
os.environ["MASTER_PORT"] = "29500"  # Use an appropriate port number

torch.distributed.is_available()
# File to store the responses
functions_file = "functions.json"
features_file = "responses.json"

device = torch.set_default_device("cuda")
# Create a lock object


# Initialization of the model and tokenizer should be done inside the function that runs on each process to ensure they are correctly mapped to the respective device (GPU).
def setup_model_and_tokenizer(device):
    model_name_or_path = (
        "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        fast_attention2=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model.to(device)
    return tokenizer, model


def generate_response(tokenizer, model, features):
    prompt = features
    system_message = """When presented with features described by a visual language model, synthesize a function call and generate its output. The function call should be structured to capture specific attributes of the image as detailed by the visual description. Start the function call with the <fn_call> tag and then provide the expected output in JSON format.

[INSTRUCTION]
1. Based on the visual description, create a structured function call under the <fn_call> tag.
2. Generate the expected output of the function call as if it were executed.

[EXAMPLE]
Visual Description: 'A red and white bus with an advertisement on the back, driving through the city streets.'

Synthesized Function Call and Output:

[FUNCTION CALL]
 {{
  'type': 'object',
  'properties': {{
    'bus_colors': {{
      'type': 'array',
      'description': 'The colors of the bus in the image.',
      'items': {{
        'type': 'string',
        'enum': ['red', 'white']
      }}
    }},
    'bus_features': {{
      'type': 'string',
      'description': 'The features seen on the back of the bus.',
      'enum': ['advertisement']
    }},
    'bus_location': {{
      'type': 'string',
      'description': 'The location of the bus.',
      'enum': ['driving']
    }}
  }}
}}

[EXPECTED OUTPUT]
{{
  'bus_colors': ['red', 'white'],
  'bus_features': 'advertisement',
  'bus_location': 'driving'
}}
"""
    prompt_template = f"""<|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant"""
    input_ids = tokenizer(
        prompt_template, return_tensors="pt"
    ).input_ids.cuda()
    outputs = model.generate(
        input_ids,
        temperature=0.7,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Decode the generated tokens to a string
    full_response = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

    # Use regex to find everything after "assistant"
    match = re.search(r"assistant\s*(.*)", full_response, re.DOTALL)
    if match:
        response = match.group(
            1
        )  # Extract everything after "assistant"
    else:
        response = "No response found after 'assistant'."
    print(response)
    return response


def expand_qa(rank, features_with_ids, output_queue, world_size):
    torch.cuda.set_device(
        rank
    )  # Set the current device to the specified GPU
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    tokenizer, model = setup_model_and_tokenizer(device)
    print(f"Process {rank} is running on GPU {rank}")
    for feature in features_with_ids:
        if rank == (feature["id"] % world_size):
            response = generate_response(
                tokenizer, model, feature["response"]
            )
            output_queue.put(
                {"id": feature["id"], "response": response}
            )
    # Save the response
    if rank == 0:
        for _ in range(world_size):
            output_queue.put("DONE")
    dist.destroy_process_group
    return response


def load_processed_items(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            processed_data = json.load(file)
            # Assuming each item has a unique identifier 'id'
            return set(item["id"] for item in processed_data)
    return set()


# Function to save responses in a JSON file
def save_response(data):
    if os.path.exists(functions_file):
        with open(functions_file, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    else:
        with open(functions_file, "w") as file:
            json.dump([data], file, indent=4)


def process_responses(file_path, output_file_path):
    processed_items = load_processed_items(output_file_path)
    with open(file_path, "r") as file:
        data = json.load(file)
        for item in data:
            # Check if the item has already been processed
            if item["id"] in processed_items:
                continue
            features = item.get("response", "")
            output = expand_qa(features)
            item["new_response"] = (
                output  # Add the new response to the original object
            )
            save_response(item)
    return data


def load_features(features_path, responses_path):
    # Load the features
    with open(features_path, "r") as f:
        features = json.load(f)
    processed_ids = set()

    # Load the processed responses
    try:
        with open(responses_path, "r") as f:
            responses = json.load(f)
        # Extract the IDs of processed features
        processed_ids = {response["id"] for response in responses}
    except FileNotFoundError:
        # If the responses file doesn't exist, no IDs have been processed
        processed_ids = set()

    # Filter out features that have already been processed
    new_features = [
        feature
        for feature in features
        if feature["id"] not in processed_ids
    ]

    return new_features


def write_outputs_from_queue(output_queue, functions_path):
    with open(functions_path, "a") as f:
        while True:
            output = (
                output_queue.get()
            )  # This will block until an item is available
            if output == "DONE":
                break  # Break the loop if a "DONE" signal is received
            f.write(json.dumps(output) + "\n")


def main():
    set_start_method(
        "spawn"
    )  # Recommended for PyTorch multiprocessing
    world_size = torch.cuda.device_count()
    unprocessed_features = load_features(
        features_file, functions_file
    )
    output_queue = Queue()
    writer_process = Process(
        target=write_outputs_from_queue,
        args=(output_queue, functions_file),
    )
    writer_process.start()
    features_with_ids = [
        {"id": feature["id"], "response": feature["response"]}
        for feature in unprocessed_features
    ]
    # Ensure features list is properly divided among processes or handled per your logic
    # This example assumes a simplistic division, which may need adjustment
    mp.spawn(
        expand_qa,
        args=(features_with_ids, output_queue, world_size),
        nprocs=world_size,
        join=True,
    )

    # Signal the writer process that processing is done
    output_queue.put("DONE")
    writer_process.join()


if __name__ == "__main__":
    main()
