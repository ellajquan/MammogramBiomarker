# Imaging Biomarkers of Indolent Early-Stage Breast Cancer Using Deep Learning Techniques
# Background
Breast cancer affects 1 in 8 women, with risk peaking among women in their 70s. Although most older women diagnosed with localized breast cancer have a high five-year survival rate of 95%, overdiagnosis and overtreatment are significant issues. Current treatments often involve surgery, radiation, or hormone-blocking medications, which might be unnecessary for some low-risk cases. Deep learning, particularly convolutional neural networks, has shown promise in mammographic analysis and risk assessment, making it a potential solution to address overtreatment.

# Objective
The objective of my research is to identify older women with early-stage breast cancer who are candidates for active surveillance rather than immediate treatment. Specifically, we aim to use mammograms to identify women who are likely to have low-risk breast cancer, such as the Luminal A subtype, shown to have slow proliferation rate and excellent prognosis.

# Study Design
Our approach is to develop and validate an AI-based algorithm utilizing convolutional neural networks, a deep learning framework optimized for image analysis. The study involves supervised learning using labeled mammograms and external validation on an independent cohort for generalization. We hypothesize that this model can accurately and reliably identify low-risk breast cancer cases, paving the way for active surveillance strategies.

# Data Structure
```
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
```
