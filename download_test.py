from azure.storage.blob import BlobServiceClient
import os

account_name = 'l3storage001'
account_key = ''
blob_service_client = BlobServiceClient(account_url=f'https://{account_name}.blob.core.windows.net', credential=account_key)
container_client = blob_service_client.get_container_client('amos22-data')

# Tüm blobları listele
blobs = container_client.list_blobs()
blob_list = []
for blob in blobs:
    blob_list.append(blob.name)
    print(f"{blob.name} - {blob.size} bytes")

print(f"\nToplam {len(blob_list)} dosya bulundu.")

# Tüm dosyaları indir
os.makedirs('data/amos22', exist_ok=True)
downloaded_count = 0

for blob_name in blob_list:
    try:
        blob_client = container_client.get_blob_client(blob_name)
        local_path = os.path.join('data/amos22', blob_name.replace('/', '_'))

        print(f"İndiriliyor: {blob_name} -> {local_path}")
        with open(local_path, 'wb') as f:
            f.write(blob_client.download_blob().readall())

        downloaded_count += 1
        print(f"Başarıyla indirildi: {blob_name}")

    except Exception as e:
        print(f"Hata - {blob_name}: {e}")

print(f"\nİndirme tamamlandı. {downloaded_count}/{len(blob_list)} dosya indirildi.")