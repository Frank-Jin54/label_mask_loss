from datasets import load_dataset
def load_imagenet_dataset():
    # image_data: Maysee/tiny-imagenet for tiny imagenet dataset
    tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
    tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', split='test')
    return tiny_imagenet_train, tiny_imagenet_test
