import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re
import os

device = torch.set_default_device("cuda")
# Create a lock object
model_name_or_path = (
    "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"
)
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    fast_attention2=True,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map=("auto"),
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# File to store the responses
functions_file = "functions.json"


def expand_qa(features):
    prompt = f"""{features}"""
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
    model.to(device)
    outputs = model.generate(
        input_ids,
        temperature=0.7,
        max_new_tokens=512,
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


# Process the responses.json file
updated_data = process_responses("responses.json", "functions.json")


print("Data saved in functions.json")
