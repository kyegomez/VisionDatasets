from typing import Any, Callable, Iterator
from datasets import load_dataset
from PIL import Image
import json


class ImageDatasetIterator:
    """
    An iterator class for processing images in a dataset.

    Args:
        dataset_name (str): The name of the dataset.
        model (Callable): The model function to process the images.

    Attributes:
        dataset: The dataset to iterate over.
        model: The model function to process the images.
        index (int): The current index of the iterator.

    Methods:
        __iter__(): Returns the iterator object.
        __next__(): Returns the next image in the dataset.
        process_images(): Processes the images in the dataset using the model function.
        process_images_to_json(json_file: str): Processes the images and writes the output to a JSON file.
        run(json_file: str): Runs the image processing and writes the output to a JSON file.
        create_and_run(dataset_name: str, model: Callable[[Image.Image], Any], json_file: str): Creates an instance of ImageDatasetIterator and runs the image processing.

    """

    def __init__(
        self,
        dataset_name: str,
        model: Callable,
        json_file: str = "vision_datasets.json",
    ):
        self.dataset_name = dataset_name
        self.model = model
        self.json_file = json_file

        self.dataset = load_dataset(dataset_name)
        self.index = 0

    def __iter__(self) -> Iterator[Image.Image]:
        return self

    def __next__(self) -> Image.Image:
        if self.index >= len(self.dataset):
            raise StopIteration
        image_data = self.dataset[self.index]["image"]
        image = Image.fromarray(image_data)
        self.index += 1
        return image

    def process_images(self, task: str) -> Iterator[Any]:
        """
        Processes the images in the dataset using the model function.

        Yields:
            Any: The output of the model function for each image.

        """
        self.index = 0  # Reset index to start from the beginning
        for image in self:
            output = self.model(task, image)  # Pass image into the model
            yield output

    def process_images_to_json(self) -> None:
        """
        Processes the images and writes the output to a JSON file.

        Args:
            json_file (str): The path to the JSON file.

        """
        with open(self.json_file, "w") as f:
            for output in self.process_images():
                json.dump(output, f)
                f.write("\n")  # Write each output on a new line

    def run(self) -> None:
        """
        Runs the image processing and writes the output to a JSON file.

        Args:
            json_file (str): The path to the JSON file.

        """
        self.process_images_to_json(self.json_file)

    @classmethod
    def create_and_run(
        self,
        cls,
        dataset_name: str,
        model: Callable[[Image.Image], Any],
    ) -> "ImageDatasetIterator":
        """
        Creates an instance of ImageDatasetIterator and runs the image processing.

        Args:
            dataset_name (str): The name of the dataset.
            model (Callable): The model function to process the images.
            json_file (str): The path to the JSON file.

        Returns:
            ImageDatasetIterator: The created instance of ImageDatasetIterator.

        """
        processor = cls(dataset_name, model)
        processor.run(self.json_file)
        return processor
