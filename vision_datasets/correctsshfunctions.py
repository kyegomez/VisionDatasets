import json

def correct_json_file(input_file, output_file):
    # Initialize an empty list to store JSON objects
    corrected_data = []

    # Open the file and read line by line
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Convert each line to a JSON object and append to the list
            try:
                json_object = json.loads(line)
                corrected_data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from line: {line}")
                print(f"Error message: {e}")

    # Write the corrected data to a new file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(corrected_data, outfile, indent=4)

# Example usage
correct_json_file('sshfunctions.json', 'corrected_sshfunctions.json')