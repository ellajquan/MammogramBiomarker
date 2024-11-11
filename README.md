# MammogramBiomarker
breast_cancer_model/
│
├── main.py                  # Main script for training and evaluation
├── config.json              # Configuration file for the project
├── README.md                # Project overview
├── requirements.txt         # Required Python packages
│
├── models/
│   ├── __init__.py          # Makes the models folder a package
│   ├── breast_cancer_model.py  # Contains model definitions, including ChannelAttention, CSA, SelfAttention, etc.
│
├── utils/
│   ├── __init__.py          # Makes the utils folder a package
│   ├── data_loader.py       # Custom dataset and DataLoader related utilities
│   ├── s3_utils.py          # S3-related helper functions (e.g., load_from_s3, upload_to_s3)
│   ├── attention_visualizer.py  # Functions to visualize attention maps
│   ├── training_utils.py    # Helper functions for training loop, early stopping, checkpoints
│
└── checkpoints/             # (Optional) Directory to store local checkpoints 
