import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ColorizeData(Dataset):
    def __init__(self, id_list):
        self.input_transform = T.Compose([T.ToPILImage(),
                                          T.Resize(size=(256, 256), interpolation=Image.LANCZOS),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize(0.5, 0.5)
                                          ])
        self.target_transform = T.Compose([T.ToPILImage(),
                                           T.Resize(size=(256, 256), interpolation=Image.LANCZOS),
                                           T.ToTensor(),
                                           T.Normalize(0.5, 0.5)
                                           ])
        self.id_list = id_list

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, index: int):
        img_name = self.id_list[index]
        image = np.asarray(Image.open(img_name))
        try:
            input_image = self.input_transform(image)
            target_image = self.target_transform(image)
        except:
            return None
        return input_image, target_image

dataset_path = '../../data/landscape_images'
ratio = 0.1


class ColorizeDataset:
    def __init__(self):
        all_img_ids = [os.path.join(dataset_path, img_id) for img_id in os.listdir(dataset_path)]
        random.shuffle(all_img_ids)

        train_size = int(len(all_img_ids) * (1 - ratio))
        train_ids = all_img_ids[:train_size]
        val_ids = all_img_ids[train_size:]

        self.train_dataset = ColorizeData(train_ids)
        self.val_dataset = ColorizeData(val_ids)


def collate_fn(batch):
    # use the customized collate_fn to filter out bad inputs
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = ColorizeDataset()
    train_loader = DataLoader(dataset=dataset.train_dataset                              ,
                              batch_size=5,
                              shuffle=True,
                              collate_fn=collate_fn)
    data = next(iter(train_loader))
    print(data[0].size())
    print(data[1].size())