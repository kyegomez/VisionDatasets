import json

def remove_duplicates(json_file_path, output_file_path):
    try:
        # Load the data from the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Use a dictionary to remove duplicates, assuming data is a list of dictionaries
        unique_data = {}
        for item in data:
            # Use image_path as the unique key
            image_path = item['image_path']
            unique_data[image_path] = item

        # Convert the unique_data back to a list
        unique_data_list = list(unique_data.values())

        # Write the unique data to a new JSON file
        with open(output_file_path, 'w') as file:
            json.dump(unique_data_list, file, indent=4)

        print(f"Successfully removed duplicates and saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
json_file_path = 'functioncalls.json'
output_file_path = 'unique_functioncalls.json'
remove_duplicates(json_file_path, output_file_path)