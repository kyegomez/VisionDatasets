import json
import os


def create_image_json(response_file, images_file, output_file):
    # Load the response data
    with open(response_file, "r") as f:
        response_data = json.load(f)

    # Create a mapping of image names to IDs
    image_id_map = {
        os.path.basename(entry["image_path"]): entry["id"]
        for entry in response_data
    }

    # Read the image file names from processed_images.txt
    with open(images_file, "r") as f:
        image_names = [line.strip() for line in f]

    # Create a list to store the new JSON structure
    image_json_data = []

    # Map the image names to their IDs
    for image_name in image_names:
        if image_name in image_id_map:
            image_json_data.append(
                {
                    "image_name": image_name,
                    "id": image_id_map[image_name],
                }
            )

    # Write the data to a new JSON file
    with open(output_file, "w") as f:
        json.dump(image_json_data, f, indent=4)


# Call the function with your file names
create_image_json(
    "final_combined_responses.json",
    "processed_images.txt",
    "image_id_mapping.json",
)
