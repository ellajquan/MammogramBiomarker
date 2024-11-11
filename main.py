import torch
import json
import os
from torch.utils.data import DataLoader
from models.breast_cancer_model import BreastCancerModel
from utils.data_loader import BreastCancerDataset, collate_fn
from utils.s3_utils import load_from_s3, save_checkpoint_s3
from utils.training_utils import train_model, load_checkpoint
from utils.attention_visualizer import visualize_attention
import transforms

# Load configuration from config.json
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Load configuration values
bucket_name = config["s3_bucket_name"]
train_labels_path = config["train_labels_path"]
test_labels_path = config["test_labels_path"]
train_prefix = config["output_dir_train"]
test_prefix = config["output_dir_test"]
img_rows, img_cols = config["img_rows"], config["img_cols"]
channels = config["in_channel"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
all_epochs = config["all_epochs"]

# Initialize S3 client
s3 = boto3.client('s3')

# Load train labels from S3
train_df = load_from_s3(bucket_name, train_labels_path, read_func=pd.read_excel)

# Dataset and DataLoader setup
transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = BreastCancerDataset(train_df, bucket_name, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BreastCancerModel(num_classes=num_classes).to(device)

# Define optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Start training
train_model(model, train_loader, optimizer, criterion, num_epochs=all_epochs, device=device)
