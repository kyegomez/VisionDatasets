import json


def merge_data(sshfunctions_file, responses_file, output_file):
    # Load sshfunctions data
    with open(sshfunctions_file, "r", encoding="utf-8") as file:
        sshfunctions_data = json.load(file)

    # Load responses data
    with open(responses_file, "r", encoding="utf-8") as file:
        responses_data = json.load(file)

    # Convert responses_data into a dict for faster lookups
    responses_dict = {item["id"]: item for item in responses_data}

    # Merge data
    for item in sshfunctions_data:
        # Assuming 'id' is the field to match on
        response_item = responses_dict.get(item["id"])
        if response_item:
            # Add 'image_path' and 'response' to the sshfunctions item
            item["image_path"] = response_item.get(
                "image_path", "No image path found"
            )
            item["response"] = response_item.get(
                "response", "No response found"
            )

    # Write the updated data to a new file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(sshfunctions_data, outfile, indent=4)


# Example usage
merge_data(
    "sshfunctions.json", "responses.json", "updated_sshfunctions.json"
)
