import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
torch.manual_seed(1234)

    

# Path to the images folder on the G drive
image_folder_path = 'G:/images/syntheticcorrosion/test/Corrosion'
# File to store the list of processed images
processed_images_file = 'processed_images.txt'
# Load the list of already processed images
if os.path.exists(processed_images_file):
    with open(processed_images_file, 'r') as file:
        processed_images = file.read().splitlines()
else:
    processed_images = []
    

# File to store the responses
responses_file = 'responses.json'


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Instantiate the QwenVLMultiModal model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()


# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Function to process an images and generate QA pairs
def generate_qa_for_image(image_path):
    # Run the model to generate QA pairs
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': "Begin by describing what you see in the image. Extract as many features as you can in english."},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response, history

# Function to save the list of processed images
def save_processed_images(processed_images):
    with open(processed_images_file, 'w') as file:
        for image_name in processed_images:
            file.write(image_name + '\n')

# Function to save responses in a JSON file
def save_response(image_path, response):
    data = {'image_path': image_path, 'response': response}
    if os.path.exists(responses_file):
        with open(responses_file, 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    else:
        with open(responses_file, 'w') as file:
            json.dump([data], file, indent=4)


# Loop through each image in the directory
for image_file in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_file)
    
    # Check if the file is an image and not already processed
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')) and image_file not in processed_images:
        try:
            qa_pairs, history = generate_qa_for_image(image_path)
            print(f"QA pairs for {image_file}:")
            print(qa_pairs)
            # Save the response and the image path
            save_response(image_path, qa_pairs)
            # Add the image to the processed list and save
            processed_images.append(image_file)
            save_processed_images(processed_images)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Please note, the code assumes the 'model' object has a method that takes
# an image and a prompt as input and returns QA pairs.