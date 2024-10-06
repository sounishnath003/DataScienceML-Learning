import torch
import json
from pprint import pformat
import warnings
from typing import List
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel


def log(*data):
    print("[INFO]:", ":".join(map(str, data)))


@dataclass
class TokenizedInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def tokenize(model: PreTrainedTokenizer, sentence: str, max_length: int = 64):
    with torch.no_grad():
        tokenized = model.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return tokenized


def get_query_embeddings(tokenizer:PreTrainedTokenizer, model:AutoModel, query:str):
    with torch.no_grad():
        query_tokens = tokenize(tokenizer_model, query)
        query_input_ids = query_tokens["input_ids"]
        query_attn_mask = query_tokens["attention_mask"]
        query_embeddings =model.forward(
            input_ids=query_tokens["input_ids"],
            attention_mask=query_tokens["attention_mask"],
        ).last_hidden_state

        # exapand the attn_mask to match the dimension of embeddings
        query_attn_mask = query_attn_mask.unsqueeze(-1).expand(query_embeddings.size())
        log("modified.attn.mask.size", query_attn_mask.size())

        # mask_embddings
        mask_query_embeddings = torch.mul(query_embeddings, query_attn_mask)
        log("masked.query_embeddings", mask_query_embeddings.size())

        # sum the query_embeddings along the sequence dimension
        summed = torch.sum(mask_query_embeddings, dim=1)
        log("summed", summed.size())

        # calculate the mean
        query_mean_pooled = torch.div(summed, torch.clamp(query_attn_mask.sum(1), min=1e-9))
        log("mean.pooled", mean_pooled)
        log("mean.pooled", mean_pooled.size())

        return query_mean_pooled
    


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tokenizer_model = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")

    sentences = [
        "Sounish is a highly motivated software engineer leveraging cloud, data, and AI to solve complex challenges.",
        "Sounish also, skilled in a variety of programming languages, libraries, and frameworks, including Python, JavaScript, Go, Java, PyTorch, Angular, Node.js, and more. He also has experience with a variety of tools and platforms, such as Git, Docker, and Kubernetes.",
        "Ram is a principle software engineer, but works only Finance domain. he changed his career objective. not a SWE anymore."
    ]

    log("sentences", sentences[:])

    sentence_tokens = {
        "sentence": sentences[0],
        "tokens": tokenize(tokenizer_model, sentences[0]),
    }
    log("tokens output", sentence_tokens)

    tokenized_inputs: List[TokenizedInputs] = []

    for sentence in sentences:
        tok_out = tokenize(tokenizer_model, sentence)
        tokenized_inputs.append(
            TokenizedInputs(
                input_ids=tok_out["input_ids"][0],
                attention_mask=tok_out["attention_mask"][0],
            )
        )

    input_ids_tokens = torch.stack([ti.input_ids for ti in tokenized_inputs])
    attn_mask_tokens = torch.stack([ti.attention_mask for ti in tokenized_inputs])

    log("input.ids.size", input_ids_tokens.size())
    log("attn.mask.size", attn_mask_tokens.size())

    with torch.no_grad():
        model_out = model.forward(
            input_ids=input_ids_tokens, attention_mask=attn_mask_tokens
        )
        embeddings = model_out.last_hidden_state
        log("embeddings.size", embeddings.size())

    # exapand the attn_mask to match the dimension of embeddings
    attn_mask_tokens = attn_mask_tokens.unsqueeze(-1).expand(embeddings.size())
    log("modified.attn.mask.size", attn_mask_tokens.size())

    # mask_embddings
    mask_embeddings = torch.mul(embeddings, attn_mask_tokens)
    log("masked.embeddings", mask_embeddings.size())

    # sum the embeddings along the sequence dimension
    summed = torch.sum(mask_embeddings, dim=1)
    log("summed", summed.size())

    # calculate the mean
    mean_pooled = torch.div(summed, torch.clamp(attn_mask_tokens.sum(1), min=1e-9))
    log("mean.pooled", mean_pooled)
    log("mean.pooled", mean_pooled.size())

    # calculate the cosine similarity
    cosine_similar_chk = torch.nn.CosineSimilarity(dim=1)

    # query embeddings generator
    query = "What are tools and technologies Sounish works on?"
    query_meanpool_embeddings=get_query_embeddings(tokenizer_model, model, query)
    log("query.meanpool.embeddings", query_meanpool_embeddings.size())

    # similarity calculations
    similarity_scores = cosine_similar_chk.forward(
        query_meanpool_embeddings.squeeze(0), mean_pooled
    )
    log("similarity_scores", similarity_scores)

    answers = []
    for index, score in enumerate(similarity_scores):
        answers.append(
            
                {
                    "sentence": sentences[index],
                    "score": score.item(),
                }
            )
    
    
    log("answers", json.dumps({
        "query": query,
        "similarSearchs": answers,
        "topK": 3,
    }, indent=4))
