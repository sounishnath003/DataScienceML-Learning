# * coding utf-8 *
# @author: @github/sounishnath003
# createdAt: 11-08-2024

"""
Text embedding models are not the Generative AI Embedding Models
** Uses Classical Bert Models Last Hidden States as Embedding of the sentences
** Note: As BERT Tokenzers supports maximum of 512 (context dimension)
** You have to split the document chunks <= 512
** In this case Tokenizer Input ids Max length considered as "384"

** Tested on :

1) India's Freedom (Article Generated by Google Gemini)
2) Terraform FAQ and Hands on (Article Generated by Google Gemini)
"""

import os
import torch
import logging
import warnings
from typing import List
from langchain.pydantic_v1 import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModel
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever


# AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
# AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

def log(message, *args):
    logging.info(message, *args)


class DistilbertModelEmbeddings(Embeddings):

    def __init__(self):
        self._max_length = 384
        self._tokenzier = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._model = AutoModel.from_pretrained("distilbert-base-uncased")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_outputs = []
        for text in texts:
            embeddings_outputs.append(self.embed_query(text))
        return embeddings_outputs

    def embed_query(self, text: str) -> List[float]:
        tokenizer_out = self._tokenzier(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        model_logits = None
        with torch.no_grad():
            model_logits = self._model.forward(**tokenizer_out)

        embeddings = self._mean_pooling(model_logits, tokenizer_out["attention_mask"])
        return embeddings

    def _mean_pooling(self, model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        return embeddings[0].float().numpy().tolist()


def format_docs(docs):
    return "\n\n".join([page.page_content for page in docs]).strip()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings("ignore")

    # sample_docx = os.path.join(os.getcwd(), "data-repo", "terraform.docx")
    # log("file to be read {%s}", sample_docx)
    # loader = Docx2txtLoader(sample_docx)
    # documents = loader.load()

    loader = TextLoader(file_path="./data-repo/texts/india.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n\n\n",
        ],
        chunk_size=300,
        chunk_overlap=100,
        keep_separator=False,
    )
    documents = text_splitter.split_documents(documents)
    log("total.documents: {%s}", len(documents))

    embedding_model = DistilbertModelEmbeddings()
    vector_store = Chroma(
        embedding_function=embedding_model, persist_directory=".chromadb"
    )
    vector_store.add_documents(documents)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # llm_service = Ollama(verbose=True, model="llama2:latest")
    # retriever = MultiQueryRetriever.from_llm(
    #     retriever=vector_store.as_retriever(
    #         search_type="similarity_score_threshold",
    #         search_kwargs={"score_threshold": 0.5},
    #     ),
    #         llm=llm_service,
    # )

    sample_queries = [
        "What was the significance of the Quit India Movement in the Indian independence struggle?",
        "What were the contributions of women to the Indian independence movement?",
        "Who has established Indian National Army?",
        "How did the Indian National Army (INA) impact the course of the Indian independence struggle?",
        "What was the influence of Netaji Subhas Bose?",
        "how big is black hole?",
    ]
    sample_query = "Tell me some Concepts of terraform?"  # sample_queries[2]

    log("query to be asked: {%s}", sample_query)

    outputs = retriever.invoke(sample_query)
    log(10 * "=============")
    log("possible_answer: {%s}", format_docs(outputs))
    log(10 * "=============")

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use 10 (ten) sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    Context:
    {context}

    Question: {question}
    Note: Make sure question is aligning to the Content which is provided above!!. IF not Just say "Go fuck yourself"

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm_service
        | StrOutputParser()
    )

    outputs = rag_chain.invoke(sample_query)
    log(10 * "=============")
    log("possible_answer: {%s}", outputs)
    log(10 * "=============")
