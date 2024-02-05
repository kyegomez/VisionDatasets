import json
import re

def extract_expected_output_as_string_jsonl(input_file, output_file):
    updated_lines = []  # To hold updated JSON objects as strings

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)  # Parse each line as a JSON object

            user_text = obj.get("response", "")
            # Use regex to find the expected output section, capturing until the next double quote
            # Assuming the entire expected output is in one line
            match = re.search(r'\[EXPECTED OUTPUT\](.*?)$', user_text, re.DOTALL)
            if match:
                # Extract the matched expected output text
                expected_output_str = match.group(1).strip()
                # Assign the raw string of expected output to the 'output' key
                obj['output'] = expected_output_str

            # Convert the updated object back to a JSON string and store it
            updated_lines.append(json.dumps(obj))

    # Write the updated lines back to a new .jsonl file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in updated_lines:
            outfile.write(line + '\n')  # Write each updated JSON object as a separate line

# Example usage for .jsonl files
extract_expected_output_as_string_jsonl('newcombined_functions.jsonl', 'newcobimed_functionsoutput.jsonl')