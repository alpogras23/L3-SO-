from azure.storage.blob import BlobServiceClient

account_name = 'l3storage001'
account_key = ''
blob_service_client = BlobServiceClient(account_url=f'https://{account_name}.blob.core.windows.net', credential=account_key)
container_client = blob_service_client.get_container_client('amos22-data')
blobs = list(container_client.list_blobs())
for blob in blobs[:10]:
    print(f'{blob.name} - {blob.size} bytes')