## Retrieves an Excel file from the S3 bucket, split is done at the patient level to ensure that no patient appears in both sets. Create separate DataFrames for training and testing data.
## Extracts patient IDs and subtypes.
## Processes each view (like CC or MLO) for each patient.
## Converts the DICOM to PNG using the convert_dcm_to_png_from_s3() function.
## Collects metadata for each converted PNG (including paths and patient information).
import os
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from io import BytesIO
import json
import boto3
from sklearn.model_selection import train_test_split
import boto3

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'ella-dlbiomarkers'
prefix = 'CMMD-D2/'  # or adjust to your specific path

# Initialize variables for pagination
continuation_token = None
file_count = 0

while True:
    # Fetch objects from the bucket
    if continuation_token:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
    else:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    # List all object keys in the current response
    for obj in response.get('Contents', []):
        print(obj['Key'])
        file_count += 1

    # Check if there are more objects to fetch
    if response.get('IsTruncated'):  # Indicates if more pages are available
        continuation_token = response['NextContinuationToken']
    else:
        break

print(f"Number of files: {file_count}")

# Load configuration file
with open('config.json') as config_file:
    config = json.load(config_file)
# Configuration settings
bucket_name = config["s3_bucket_name"]
file_path_excel = config["file_path_excel"]
data_percentage = config.get("data_percentage", 1)  # Default to 100% if not set
train_test_split_percentage = config.get("train_test_split_percentage", 0.8)  # Default to 80% train, 20% test
output_excel_name_train = config.get("output_excel_name_train", "train_labels.xlsx")
output_excel_name_test = config.get("output_excel_name_test", "test_labels.xlsx")
views = config.get("views", ["CC", "MLO"])

# Load metadata from S3 directly
obj = s3.get_object(Bucket=bucket_name, Key=file_path_excel)
metadata = pd.read_excel(BytesIO(obj['Body'].read()))

# Sample a subset of the data based on data_percentage
metadata = metadata.sample(frac=data_percentage, random_state=42).reset_index(drop=True)

# Split at patient level to avoid data leakage
unique_patient_ids = metadata['patient_id'].unique()
train_ids, test_ids = train_test_split(unique_patient_ids, train_size=train_test_split_percentage, random_state=42)

# Create train and test metadata based on patient IDs
train_metadata = metadata[metadata['patient_id'].isin(train_ids)].reset_index(drop=True)
test_metadata = metadata[metadata['patient_id'].isin(test_ids)].reset_index(drop=True)

# Map view types to exam IDs
view_map = {
    'L_CC': '1-1',
    'L_MLO': '1-2',
    'R_CC': '1-3',
    'R_MLO': '1-4'
}

train_metadata
len(_ids)
# Pagination and counting PNG files
paginator = s3.get_paginator('list_objects_v2')
operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}
page_iterator = paginator.paginate(**operation_parameters)

png_count = 0

for page in page_iterator:
    if 'Contents' in page:
        for obj in page['Contents']:
            if obj['Key'].endswith('.png'):
                png_count += 1

print(f"Total number of PNG files in {prefix}: {png_count}")

# Initialize S3 client
s3 = boto3.client('s3')

# Bucket and prefix information
bucket_name = 'ella-dlbiomarkers'
prefix = 'CMMD-D2/train/PNGs/'

# Load train metadata
expected_ids = set(train_metadata['patient_id'].apply(lambda x: f"{str(x).zfill(4)}"))

# Pagination to list all files in S3
paginator = s3.get_paginator('list_objects_v2')
operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}
page_iterator = paginator.paginate(**operation_parameters)

# Collect all PNG files in S3
s3_files = set()
for page in page_iterator:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            patient_id = key.split('/')[-1].split('_')[0]
            s3_files.add(patient_id)

# Compare expected and actual files
missing_ids = expected_ids - s3_files
extra_ids = s3_files - expected_ids

print(f"Missing patient IDs in S3: {len(missing_ids)}")
print(f"Extra patient IDs in S3: {len(extra_ids)}")
print(f"Missing patient IDs in S3: {missing_ids}")
print(f"Extra patient IDs in S3: {extra_ids}")


# Initialize S3 client
s3 = boto3.client('s3')

# Bucket and prefix information
bucket_name = 'ella-dlbiomarkers'
prefix = 'CMMD-D2/test/PNGs/'

# Load train metadata
expected_ids = set(test_metadata['patient_id'].apply(lambda x: f"{str(x).zfill(4)}"))

# Pagination to list all files in S3
paginator = s3.get_paginator('list_objects_v2')
operation_parameters = {'Bucket': bucket_name, 'Prefix': prefix}
page_iterator = paginator.paginate(**operation_parameters)

# Collect all PNG files in S3
s3_files = set()
for page in page_iterator:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            patient_id = key.split('/')[-1].split('_')[0]
            s3_files.add(patient_id)

# Compare expected and actual files
missing_ids = expected_ids - s3_files
extra_ids = s3_files - expected_ids

print(f"Missing patient IDs in S3: {len(missing_ids)}")
print(f"Extra patient IDs in S3: {len(extra_ids)}")
print(f"Missing patient IDs in S3: {missing_ids}")
print(f"Extra patient IDs in S3: {extra_ids}")

def convert_dcm_to_png_s3(dcm_s3_key, s3_output_key):
    """Convert a DICOM file from S3 to PNG and upload to S3."""
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=dcm_s3_key)
        dcm = pydicom.dcmread(BytesIO(obj['Body'].read()))
        pixel_array = dcm.pixel_array
        with BytesIO() as png_buffer:
            plt.imsave(png_buffer, pixel_array, cmap='gray', format='png')
            png_buffer.seek(0)
            s3.put_object(Bucket=bucket_name, Key=s3_output_key, Body=png_buffer)
    except s3.exceptions.NoSuchKey:
        print(f"File not found: {dcm_s3_key}")
    except Exception as e:
        print(f"Error converting {dcm_s3_key}: {e}")

def process_metadata(metadata, output_prefix):
    """Process metadata and save PNGs and metadata entries to S3."""
    processed_data = []
    for _, row in metadata.iterrows():
        patient_id = row['patient_id']
        subtype = row['subtype']
        
        for view in views:
            exam_id = view_map.get(view)
            if not exam_id:
                continue
            
            # Construct S3 key for the DICOM file with patient ID and exam ID
            dcm_s3_key = f"CMMD-D2/{patient_id}_{exam_id}.dcm"
            s3_output_png_key = f"{output_prefix}/PNGs/{patient_id}_{view}.png"  # Include view for clarity
            
            # Convert DICOM to PNG and upload to S3
            convert_dcm_to_png_s3(dcm_s3_key, s3_output_png_key)
            
            # Append entry with laterality included in view name
            processed_data.append({
                'patient_id': patient_id,
                f'{view}_file': s3_output_png_key,
                'subtype': subtype
            })
    
    return processed_data

views = ["L_CC", "L_MLO", "R_CC", "R_MLO"]


# Function to convert missing patient IDs to PNG using process_metadata
def process_missing_ids(missing_ids, metadata_df, output_dir):
    missing_metadata = metadata_df[metadata_df['patient_id'].isin(missing_ids)]
    process_metadata(missing_metadata, output_dir)
process_missing_ids(missing_ids, train_metadata, 'CMMD-D2/train')

train_ids = set(train_metadata['patient_id'].apply(lambda x: f"D2-{str(x).zfill(4)}").values)

# List objects in the specified S3 prefix
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Iterate through each object in the S3 bucket and filter based on train IDs
for obj in response.get('Contents', []):
    key = obj['Key']
    patient_id = key.split('/')[-1].split('_')[0]  # Extract patient ID from the file name

    if patient_id not in train_ids:
        try:
            # Delete the object if it is not in train IDs
            s3.delete_object(Bucket=bucket_name, Key=key)
            print(f"Deleted {key} as it is not in train IDs")
        except ClientError as e:
            print(f"Error deleting {key}: {e}")
    else:
        print(f"Kept {key}")

# Function to convert missing patient IDs to PNG using process_metadata
def process_missing_ids(missing_ids, metadata_df, output_dir):
    missing_metadata = metadata_df[metadata_df['patient_id'].isin(missing_ids)]
    process_metadata(missing_metadata, output_dir)

# Function to delete extra patient PNGs from S3
def delete_extra_ids(extra_ids, bucket, prefix):
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                patient_id = key.split('/')[-1].split('_')[0]  # Extract patient ID from the file name
                
                if patient_id in extra_ids:
                    try:
                        # Delete the object if it is in extra IDs
                        s3.delete_object(Bucket=bucket, Key=key)
                        print(f"Deleted {key} as it is an extra ID")
                    except ClientError as e:
                        print(f"Error deleting {key}: {e}")

# Convert missing patient IDs to PNGs
process_missing_ids(missing_ids, train_metadata, "CMMD-D2/train")

# Delete extra patient PNGs from S3
delete_extra_ids(extra_ids, bucket_name, prefix)

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Initialize S3 client
s3 = boto3.client('s3')

# Bucket and prefix information
bucket_name = 'ella-dlbiomarkers'
prefix = 'CMMD-D2/train/PNGs/'

# List objects in the specified S3 prefix
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Iterate through each object in the S3 bucket and filter based on train IDs
for obj in response.get('Contents', []):
    key = obj['Key']
    patient_id = key.split('/')[-1].split('_')[0]  # Extract patient ID from the file name

    if patient_id not in missing_ids:
        try:
            # Delete the object if it is not in train IDs
            s3.delete_object(Bucket=bucket_name, Key=key)
            print(f"Deleted {key} as it is not in train IDs")
        except ClientError as e:
            print(f"Error deleting {key}: {e}")
    else:
        print(f"Kept {key}")
import boto3
import pandas as pd

print(train_metadata)

print(len(test_metadata))
print(len(test_ids))
def convert_dcm_to_png_s3(dcm_s3_key, s3_output_key):
    """Convert a DICOM file from S3 to PNG and upload to S3."""
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=dcm_s3_key)
        dcm = pydicom.dcmread(BytesIO(obj['Body'].read()))
        pixel_array = dcm.pixel_array
        with BytesIO() as png_buffer:
            plt.imsave(png_buffer, pixel_array, cmap='gray', format='png')
            png_buffer.seek(0)
            s3.put_object(Bucket=bucket_name, Key=s3_output_key, Body=png_buffer)
    except s3.exceptions.NoSuchKey:
        print(f"File not found: {dcm_s3_key}")
    except Exception as e:
        print(f"Error converting {dcm_s3_key}: {e}")

def process_metadata(metadata, output_prefix):
    """Process metadata and save PNGs and metadata entries to S3."""
    processed_data = []
    for _, row in metadata.iterrows():
        patient_id = row['patient_id']
        subtype = row['subtype']
        
        for view in views:
            exam_id = view_map.get(view)
            if not exam_id:
                continue
            
            # Construct S3 key for the DICOM file with patient ID and exam ID
            dcm_s3_key = f"CMMD-D2/{patient_id}_{exam_id}.dcm"
            s3_output_png_key = f"{output_prefix}/PNGs/{patient_id}_{view}.png"  # Include view for clarity
            
            # Convert DICOM to PNG and upload to S3
            convert_dcm_to_png_s3(dcm_s3_key, s3_output_png_key)
            
            # Append entry with laterality included in view name
            processed_data.append({
                'patient_id': patient_id,
                f'{view}_file': s3_output_png_key,
                'subtype': subtype
            })
    
    return processed_data

views = ["L_CC", "L_MLO", "R_CC", "R_MLO"]


train_data = process_metadata(train_metadata, "CMMD-D2/train")
# Number of items in the training metadata
train_count = len(train_metadata)  # or train_metadata_df.shape[0]

# Number of items in the testing metadata
test_count = len(test_metadata)  # or test_metadata_df.shape[0]

# Print the counts
print(f"Number of items in train_metadata_df: {train_count}")
print(f"Number of items in test_metadata_df: {test_count}")

train_metadata
test_metadata
type(test_metadata)
# Assume train_metadata is the original DataFrame with column: patient_id
expected_ids = set(test_metadata['patient_id'].apply(lambda x: f"{str(x).zfill(4)}"))

# Load the existing Excel file from S3
bucket_name = "ella-dlbiomarkers"
s3 = boto3.client('s3')
excel_buffer = BytesIO()
s3.download_fileobj(Bucket=bucket_name, Key="CMMD-D2/train_labels_old.xlsx", Fileobj=excel_buffer)

# Read the Excel file into a DataFrame
train_labels_df = pd.read_excel(excel_buffer)

# Filter the DataFrame to keep only rows with patient_id in expected_ids
filtered_df = train_labels_df[train_labels_df['patient_id'].isin(expected_ids)]

# Save the filtered DataFrame to Excel and upload to S3
filtered_excel_buffer = BytesIO()
filtered_df.to_excel(filtered_excel_buffer, index=False)

try:
    s3.put_object(Bucket=bucket_name, Key="CMMD-D2/test_labels_filtered.xlsx", Body=filtered_excel_buffer.getvalue())
    print("Filtered file successfully uploaded to S3.")
except ClientError as e:
    print(f"An error occurred: {e}")
    if e.response['Error']['Code'] == 'AccessDenied':
        print("Access denied. Please check your S3 bucket permissions.")



# Pivot the DataFrame to have columns for each image view
pivoted_df = test_metadata.pivot(index='patient_id', columns=['laterality', 'view'], values='file_path').reset_index()

# Rename the columns to match the desired format
pivoted_df.columns = [
    'patient_id', 'L_CC_file', 'L_MLO_file', 'R_CC_file', 'R_MLO_file'
]

# Add the 'subtype' and 'target' columns
pivoted_df = pivoted_df.merge(test_metadata[['patient_id', 'subtype']].drop_duplicates(), on='patient_id', how='left')
pivoted_df['target'] = pivoted_df['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)
 
pivoted_df


# Save train metadata with 'target' column to S3 as Excel
train_excel_buffer = BytesIO()
pivoted_df.to_excel(train_excel_buffer, index=False)
s3.put_object(Bucket=bucket_name, Key="CMMD-D2/test_labels.xlsx", Body=train_excel_buffer.getvalue())

# Convert processed data to DataFrame and pivot
df_test = pd.DataFrame(test_metadata)

# Keep all original columns and add 'target' column
df_test['target'] = df_test['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)

# Save train metadata with 'target' column to S3 as Excel
train_excel_buffer = BytesIO()
df_test.to_excel(train_excel_buffer, index=False)
s3.put_object(Bucket=bucket_name, Key="CMMD-D2/test_labels.xlsx", Body=train_excel_buffer.getvalue())

# Process test metadata and save to S3
test_data = process_metadata(test_metadata, "CMMD-D2/test")
df_test = pd.DataFrame(test_data)
df_test_pivot = df_test.pivot_table(index='patient_id', values=[f'{view}_file' for view in views if f'{view}_file' in df_test.columns], aggfunc='first').reset_index()
df_test_pivot['subtype'] = test_metadata.groupby('patient_id')['subtype'].first().values
df_test_pivot['target'] = df_test_pivot['subtype'].apply(lambda x: 1 if x == "Luminal A" else 0)

# Save test metadata to S3
excel_buffer = io.BytesIO()
df_test_pivot.to_excel(excel_buffer, index=False)
s3.put_object(Bucket=s3_bucket_name, Key="CMMD-D2/test_labels.xlsx", Body=excel_buffer.getvalue())

print(f"DICOM files from S3 converted and saved in {output_dir_train} for training and {output_dir_test} for testing.")
aws s3 --recursive mv s3://ella-dlbiomarkers/CMMD-D2/train/PNGs/ s3://ella-dlbiomarkers/CMMD-D2/train/exPNGs/
