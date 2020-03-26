import os
import sys
import io
import time
import datetime
from azure.storage.blob import BlockBlobService


def get_container_sas_token(block_blob_client, container_name, blob_permissions):
    container_sas_token = \
        block_blob_client.generate_container_shared_access_signature(
            container_name,
            permission=blob_permissions,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        )
    return container_sas_token


def get_blob_sas_url(block_blob_client, container_name, blob_name, blob_permissions):
    sas_token = get_container_sas_token(
        block_blob_client, container_name, blob_permissions
    )
    sas_url = block_blob_client.make_blob_url(
        container_name, blob_name, sas_token=sas_token
    )
    return sas_url
