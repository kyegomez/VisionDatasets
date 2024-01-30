import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from prompt import VISUAL_CHAIN_OF_THOUGHT
import torch
import json

torch.manual_seed(1234)


# Path to the images folder on the G drive
image_folder_path = "G:\images\combined"
# File to store the list of processed images
processed_images_file = "processed_images.txt"
# Load the list of already processed images
if os.path.exists(processed_images_file):
    with open(processed_images_file, "r") as file:
        processed_images = file.read().splitlines()
else:
    processed_images = []


# File to store the responses
responses_file = "responses.json"


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True
)

# Instantiate the QwenVLMultiModal model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True,
    bf16=True,
).eval()


# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-VL-Chat", trust_remote_code=True
)
shortprompt = "Begin by describing what you see in the image. Extract as many features as you can in english. Make sure you are as descriptive as possible in english."


# Function to process an images and generate QA pairs
def generate_qa_for_image(image_path):
    # Run the model to generate QA pairs
    query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": VISUAL_CHAIN_OF_THOUGHT},
    ])
    response, history = model.chat(
        tokenizer, query=query, history=None
    )
    return response, history


# Function to get the next available ID from the responses file
def get_next_id(responses_file):
    if os.path.exists(responses_file):
        with open(responses_file, "r") as file:
            data = json.load(file)
        return max(entry["id"] for entry in data) + 1 if data else 0
    else:
        return 0


# Function to map processed images to their IDs and save as JSON
def save_image_id_map(processed_images, responses_file, output_file):
    with open(responses_file, "r") as file:
        response_data = json.load(file)
    image_id_map = {
        entry["image_path"].split("\\")[-1]: entry["id"]
        for entry in response_data
    }
    image_json_data = [
        {"image_name": img, "id": image_id_map.get(img)}
        for img in processed_images
        if img in image_id_map
    ]
    with open(output_file, "w") as file:
        json.dump(image_json_data, file, indent=4)


# Function to save the list of processed images
def save_processed_images(processed_images):
    with open(processed_images_file, "w") as file:
        for image_name in processed_images:
            file.write(image_name + "\n")


# Function to save responses in a JSON file
def save_response(image_path, response, next_id):
    data = {
        "image_path": image_path,
        "response": response,
        "id": next_id,
    }
    if os.path.exists(responses_file):
        with open(responses_file, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    else:
        with open(responses_file, "w") as file:
            json.dump([data], file, indent=4)


next_id = get_next_id(responses_file)
# Loop through each image in the directory
for image_file in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_file)

    # Check if the file is an image and not already processed
    if (
        image_path.lower().endswith((".png", ".jpg", ".jpeg"))
        and image_file not in processed_images
    ):
        try:
            qa_pairs, history = generate_qa_for_image(image_path)
            print(f"QA pairs for {image_file}:")
            print(qa_pairs)
            # Save the response and the image path
            save_response(image_path, qa_pairs, next_id)
            # Add the image to the processed list and save
            next_id += 1
            processed_images.append(image_file)
            save_processed_images(processed_images)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Please note, the code assumes the 'model' object has a method that takes
# an image and a prompt as input and returns QA pairs.
