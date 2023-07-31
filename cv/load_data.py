from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.transforms import RandomRotation
import torchvision.transforms as transforms

TINY_IMAGENET_DATASET = 'Maysee/tiny-imagenet'
POKEMON_DATASET = "keremberke/pokemon-classification"
OXFORD_FLOWER_DATASET = "nelorth/oxford-flowers"


batch_size = 128
rotate = RandomRotation(degrees=(0, 90))
to_tensor = transforms.ToTensor()

def transform(examples):
    examples["image"] = [rotate(image.convert("RGB")) for image in examples["image"]]
    examples["image"] = [to_tensor(image) for image in examples["image"]]

    return examples

def load_imagenet_dataset():
    # image_data: Maysee/tiny-imagenet for tiny imagenet dataset
    tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    number_label = max(tiny_imagenet_train["label"]) + 1
    tiny_imagenet_train.set_transform(transform)
    train_dataset = torch.utils.data.DataLoader(tiny_imagenet_train, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', split='valid')
    tiny_imagenet_test.set_transform(transform)
    valid_dataset = torch.utils.data.DataLoader(tiny_imagenet_test, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    return train_dataset, valid_dataset, number_label

def load_imagenet_dataset():
    # image_data: Maysee/tiny-imagenet for tiny imagenet dataset
    tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    number_label = max(tiny_imagenet_train["label"]) + 1
    tiny_imagenet_train.set_transform(transform)
    train_dataset = torch.utils.data.DataLoader(tiny_imagenet_train, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', split='valid')
    tiny_imagenet_test.set_transform(transform)
    valid_dataset = torch.utils.data.DataLoader(tiny_imagenet_test, batch_size=batch_size,
                                shuffle=True, num_workers=0)

    return train_dataset, valid_dataset, number_label
