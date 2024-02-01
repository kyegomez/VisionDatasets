import json
import re

def extract_expected_output_as_string(input_file, output_file):
    # Read the JSON data
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for obj in data:
        user_text = obj.get("user", "")
        # Use regex to find the expected output section, capturing until the next double quote
        # Assuming the entire expected output is in one line
        match = re.search(r'\[EXPECTED OUTPUT\](.*?)$', user_text, re.DOTALL)
        if match:
            # Extract the matched expected output text
            expected_output_str = match.group(1).strip()
            # Assign the raw string of expected output to the 'output' key
            obj['output'] = expected_output_str

    # Write the updated data to a new file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

# Example usage
extract_expected_output_as_string('sshfunctions.json', 'sshfunctions_updated.json')