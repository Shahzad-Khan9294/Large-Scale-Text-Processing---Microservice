# Large-Scale-Text-Processing-Microservice
## High-Level Purpose
### This FastAPI app exposes an embedding microservice:
- It accepts a list of texts.
- Runs them through a LangChain-based pipeline (pipeline_run) that handles embeddings, caching, and optional FAISS vector storage.
-  Returns the embeddings (plus timing) as JSON.

## Requiremnets:
- einops
- fastapi
- uvicorn
- torch
- pydantic
- aioredis
- transformers
- huggingface_hub[hf_xet]

# Model Used
## Snowflake/snowflake-arctic-embed-m-long
It’s part of Snowflake’s Arctic-Embed family of text embedding models, hosted on Hugging Face and released under the Apache-2.0 open-source license.
## Scale & Specs

- Parameters: ~137 million
- Embedding Dimension: 768
- MTEB Retrieval Score (NDCG@10): 54.83
  (Very close to the 110M-param “m” version at 54.90)
## If you're integrating it programmatically, you have multiple options:
- Using sentence-transformers
- Using Hugging Face transformers

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-long")
embeddings = model.encode(["your text here"], prompt_name="query")  # for queries
````
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-long")
model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-m-long", add_pooling_layer=False)
model.eval()
```
