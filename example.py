from vision_datasets.iter_over_dataset import ImageDatasetIterator
from swarms import QwenVLMultiModal

model = QwenVLMultiModal(
    system_prompt="You, as the model, are presented with a visual problem. This could be an image containing various elements that you need to analyze, a graph that requires interpretation, or a visual puzzle. Your task is to examine the visual information carefully and describe your process of understanding and solving the problem.",
)

iterator = ImageDatasetIterator("coco", model)


# Run the iterator
iterator.run()
