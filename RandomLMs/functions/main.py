"""
# _* coding: utf8 *_

filename: main.py

@author: sounishnath
createdAt: 2023-06-04 01:08:34
"""

import logging
import os
import typing
from google.cloud import storage 
import tempfile

from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import (
    FakeEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenerateEmbeddingFunction")


def _to_log_text_payload(data: typing.Any):
    return {"textPayload": f"{data}"}

loader=None
documents=None

with tempfile.TemporaryDirectory() as tempdir:
    storage_client=storage.Client()
    bucket=storage_client.get_bucket("workspace-bucket-01")
    blob=bucket.get_blob("./1706.03762.pdf")

    filepath=f"{tempdir}/1706.03762.pdf"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    blob.download_to_filename(filepath)
    loader = PDFPlumberLoader(filepath)
    documents = loader.load()
    logger.info({"Total documents": len(documents), "sample": documents[0]})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=768, chunk_overlap=32, length_function=len
)
documents = text_splitter.split_documents(documents)
logger.info(
    {
        "Total documents": len(documents),
        "sample": documents[0],
        "splitting": "Recursive",
    }
)

# embedding_model = FakeEmbeddings(size=768)
embedding_model = embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(documents, embedding=embedding_model)


def docs_llm_service(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    global db
    global embedding_model
    request_json = request.get_json()
    if request.args and "message" in request.args:
        query = request.args.get("query")
        query = "Who are the authors of transformer paper?"
        docs_and_scores = db.similarity_search_with_score(query)

        logger.info(
            {
                "query": query,
                "foundAnswers": len(docs_and_scores),
                "top_match": docs_and_scores[0][0].page_content,
                "topScore": docs_and_scores[0][1],
            }
        )

        return {
            "query": query,
            "foundAnswers": len(docs_and_scores),
            "top_match": docs_and_scores[0][0].page_content,
            "topScore": docs_and_scores[0][1],
        }

    else:
        return f"Hello World!"


"""
curl -X GET \
https://asia-south1-sounish-cloud-workstation.cloudfunctions.net/docs-qa-service?query="who are the authors of transformer architecture?"
-H 'Content-Type: "application/json"'
-H 'Authorization: Bearer $(gcloud auth print-identity-token)'
"""