import json


def adjust_ids(file_name):
    # Load the file
    with open(file_name, "r") as f:
        data = json.load(f)

    # Adjust the IDs
    for entry in data:
        if "id" in entry:
            entry["id"] -= 1

    # Save the modified data back to the file
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)


# Call the function with your file name
adjust_ids("final_combined_responses.json")
