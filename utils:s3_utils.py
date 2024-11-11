Python 3.12.0 (v3.12.0:0fb18b02c8, Oct  2 2023, 09:45:56) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import boto3
... from io import BytesIO
... 
... s3 = boto3.client('s3')
... 
... def load_from_s3(bucket, key, read_func):
...     obj = s3.get_object(Bucket=bucket, Key=key)
...     return read_func(BytesIO(obj['Body'].read()))
... 
... def save_checkpoint_s3(state, bucket_name, s3_key):
...     buffer = BytesIO()
...     torch.save(state, buffer)
...     buffer.seek(0)
...     s3.upload_fileobj(buffer, bucket_name, s3_key)
...     print(f"Checkpoint saved to s3://{bucket_name}/{s3_key}")
