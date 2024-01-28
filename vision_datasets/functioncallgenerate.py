import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import re

model_name_or_path = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto", 
                                             torch_dtype=torch.float16)

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
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    outputs = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    # Decode the generated tokens to a string
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Use regex to find everything after "assistant"
    match = re.search(r'assistant\s*(.*)', full_response, re.DOTALL)
    if match:
        response = match.group(1)  # Extract everything after "assistant"
    else:
        response = "No response found after 'assistant'."

    print(response)
    return response

def process_responses(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = []
    for item in data:
        features = item.get("response", "")
        output = expand_qa(features)
        item["new_response"] = output  # Add the new response to the original object

    return data

# Process the responses.json file
updated_data = process_responses("responses.json")

# Save the updated data to functions.json
with open("functions.json", 'w') as file:
    json.dump(updated_data, file, indent=4)


print("Data saved in functions.json")