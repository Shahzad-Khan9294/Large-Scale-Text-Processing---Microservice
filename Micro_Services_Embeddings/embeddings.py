import torch
from typing import List
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SnowflakeEmbeddingModel:
    def __init__(self, model_name: str = 'Snowflake/snowflake-arctic-embed-m-long', batch_size: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            trust_remote_code=True,
            safe_serialization=True,
            rotary_scaling_factor=2
        )
        self.model.eval()
        self.batch_size = batch_size  # <== add this

    def embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=768)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs[0][:, 0]
            normalized = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(normalized.tolist())  # Collect each batch

        return all_embeddings
