'''
Adapted from:
https://dev.to/sethmlarson/python-data-streaming-to-google-cloud-storage-with-resumable-uploads-458h
'''

from google.auth.transport.requests import AuthorizedSession
from google.resumable_media import requests, common
from google.cloud import storage
import io, os
from tqdm import tqdm
import hashlib
import functools


os.environ['GDAL_DATA'] = '/anaconda3/envs/gis/share/gdal'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/terenceconlon/Google Authentication/qsel-columbia-37d91db96bb8.json'

class GCSObjectStreamUpload(object):
    def __init__(
            self,
            client: storage.Client,
            bucket_name: str,
            blob_name: str,
            chunk_size: int = 256 * 1024
    ):
        self._client = client
        self._bucket = self._client.bucket(bucket_name)
        self._blob = self._bucket.blob(blob_name)

        self._buffer = b''
        self._buffer_size = 0
        self._chunk_size = chunk_size
        self._read = 0

        self._transport = AuthorizedSession(
            credentials=self._client._credentials
        )
        self._request = None  # type: requests.ResumableUpload

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.stop()

    def start(self):
        url = (
            f'https://www.googleapis.com/upload/storage/v1/b/'
            f'{self._bucket.name}/o?uploadType=resumable'
        )
        self._request = requests.ResumableUpload(
            upload_url=url, chunk_size=self._chunk_size
        )
        self._request.initiate(
            transport=self._transport,
            content_type='application/octet-stream',
            stream=self,
            stream_final=False,
            metadata={'name': self._blob.name},
        )

    def stop(self):
        self._request.transmit_next_chunk(self._transport)

    def write(self, data: bytes) -> int:
        data_len = len(data)
        self._buffer_size += data_len
        self._buffer += data
        del data
        # pbar = tqdm(total=data_len)

        while self._buffer_size >= self._chunk_size:
            try:
                self._request.transmit_next_chunk(self._transport)
                # pbar.update(self._chunk_size)
            except common.InvalidResponse:
                self._request.recover(self._transport)
        return data_len

    def read(self, chunk_size: int) -> bytes:
        # I'm not good with efficient no-copy buffering so if this is
        # wrong or there's a better way to do this let me know! :-)
        to_read = min(chunk_size, self._buffer_size)
        memview = memoryview(self._buffer)
        self._buffer = memview[to_read:].tobytes()
        self._read += to_read
        self._buffer_size -= to_read
        return memview[:to_read].tobytes()

    def tell(self) -> int:
        return self._read


# if __name__ == '__main__':
#
#     bucket_name = 'sentinel_imagery_useast'
#     source_file_name = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles/T37PCM/stacks/' \
#                        'T37PCM_stack_EVI_10m_infilled.tif'
#
#     # source_file_name = '/Users/terenceconlon/Documents/Columbia - Spring 2020/personal/lebron.jpg'
#
#     # source_file_name = glob.glob('/Volumes/sel_external/sentinel_imagery/reprojected_tiles/{}/stacks/'
#     #                              '{}_*_100m_infilled.tif'.format(tile, tile))
#
#     destination_blob_name = os.path.join('ethiopia', source_file_name.split('/')[-1])
#
#     client = storage.Client()
#     print('Sending file: {}'.format(source_file_name))
#     with GCSObjectStreamUpload(client=client, bucket_name=bucket_name,
#                                blob_name=destination_blob_name) as s:
#
#         size = os.stat(source_file_name).st_size
#         print('Size of upload: {} MB'.format(size/1e6))
#         with open(source_file_name, 'rb') as infile:
#             pbar = tqdm(total=size)
#             for f in infile:
#                 data = io.BytesIO(f).read()
#                 pbar.update(len(data))
#                 s.write(data)



