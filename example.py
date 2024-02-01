from vision_datasets.iter_over_dataset import ImageDatasetIterator
from swarms import QwenVLMultiModal

model = QwenVLMultiModal(
<<<<<<< HEAD
    system_prompt="You, as the model, are presented with a visual problem. This could be an image containing various elements that you need to analyze, a graph that requires interpretation, or a visual puzzle. Your task is to examine the visual information carefully and describe your process of understanding and solving the problem.",
=======
    system_prompt=(
        "You, as the model, are presented with a visual problem. This"
        " could be an image containing various elements that you need"
        " to analyze, a graph that requires interpretation, or a"
        " visual puzzle. Your task is to examine the visual"
        " information carefully and describe your process of"
        " understanding and solving the problem."
    ),
>>>>>>> a7d64d7 ([FEAT][create_base_retry_decorator])
)

iterator = ImageDatasetIterator("coco", model)


# Run the iterator
iterator.run()
