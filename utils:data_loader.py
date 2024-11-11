import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import boto3
import logging

logger = logging.getLogger()

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, bucket_name, transform=None):
        self.dataframe = dataframe
        self.bucket_name = bucket_name
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_paths = [self.dataframe.iloc[idx].L_CC_file, self.dataframe.iloc[idx].R_CC_file,
                     self.dataframe.iloc[idx].L_MLO_file, self.dataframe.iloc[idx].R_MLO_file]
        images = []
        for img_path in img_paths:
            key = img_path.strip()
            try:
                obj = s3.get_object(Bucket=self.bucket_name, Key=key)
                img = Image.open(BytesIO(obj['Body'].read())).convert('RGB')  # Load image directly as RGB using PIL
                if self.transform:
                    img = self.transform(img)  # Apply transformation
                images.append(img)
            except s3.exceptions.NoSuchKey as e:
                logger.warning(f"Warning: Key {key} does not exist. Skipping this image.")
                continue

        # If not enough images are available, skip this sample
        if len(images) < 4:
            logger.warning(f"Skipping index {idx} due to insufficient images.")
            return None

        label = self.dataframe.iloc[idx].target
        return tuple(images), label  # Return as tuple

# Modify DataLoader to handle None values gracefully
def collate_fn(batch):
    # Filter out None values
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)
