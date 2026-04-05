from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # fill later

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load image + label here
        raise NotImplementedError
