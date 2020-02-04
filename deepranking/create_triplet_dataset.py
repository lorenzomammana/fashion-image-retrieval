from fashion_dataset import FashionDataset
import files as files
import os
import shutil

dataset = FashionDataset(files.small_images_directory)

new_path_root = files.small_images_classes_directory
if os.path.exists(new_path_root):
    shutil.rmtree(new_path_root)

os.mkdir(new_path_root)

for label in dataset.label_filters:
    new_path = new_path_root / label
    if os.path.exists(new_path):
        shutil.rmtree(new_path)

    os.mkdir(new_path)

    img_label = dataset.data[dataset.data['articleType'] == label]['x']
    img_label.apply(lambda x: shutil.copy(files.small_images_directory / x,
                                          new_path / x))


