from flask import Flask, request, jsonify
from azure.storage.blob import BlockBlobService
from azure.storage.blob import BlobPermissions
import url_recommend
import blob_process
import pandas as pd
import os
import json

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello():
    return '<h1>Welcome to the recommend api !</h1>'


@app.route('/recommend', methods=['POST'])
def index():
    req_body = request.get_json()
    user = req_body.get('user')
    url = req_body.get('url')
    title = req_body.get('title')
    descr = req_body.get('descr')

    # Get the source blob
    block_blob_service = BlockBlobService(
        connection_string=os.environ['AzureWebJobsStorage']
    )
    blobs_name = []
    for blob in block_blob_service.list_blobs('history-clean'):
        blobs_name.append(blob.name)
    targetBlob = blobs_name[len(blobs_name) - 1]
    sas_url = blob_process.get_blob_sas_url(
        block_blob_service, 'history-clean', targetBlob, BlobPermissions.READ)

    df = url_recommend.open_df(sas_url)
    df = url_recommend.add(df, user, url, title, descr)
    df['descr_clean'] = df['descr'].apply(url_recommend.clean_text)

    cosine_similarities = url_recommend.get_cosine_similarity(df)

    df.set_index('url', inplace=True)
    answer = url_recommend.recommend(df, url, cosine_similarities)
    info = {
        "answer": answer
    }

    return jsonify(info)


# run the server
if __name__ == '__main__':
    app.run(debug=True)
