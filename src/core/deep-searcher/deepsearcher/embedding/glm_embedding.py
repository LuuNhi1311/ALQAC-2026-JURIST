import os
from typing import List

from deepsearcher.embedding.base import BaseEmbedding

GLM_MODEL_DIM_MAP = {
    "embedding-3": 2048,
}


class GLMEmbedding(BaseEmbedding):
    """
    https://platform.openai.com/docs/guides/embeddings/use-cases
    """

    def __init__(self, model: str = "embedding-3", **kwargs):
        """

        Args:
            model_name (`str`):
        """
        from zhipuai import ZhipuAI

        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("GLM_API_KEY")
        if "model_name" in kwargs and (not model or model == "embedding-3"):
            model = kwargs.pop("model_name")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("GLM_BASE_URL", default="https://open.bigmodel.cn/api/paas/v4/")

        self.model = model
        self.client = ZhipuAI(api_key=api_key, base_url=base_url, **kwargs)

    def embed_query(self, text: str) -> List[float]:
        # text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(input=texts, model=self.model)
        res = [r.embedding for r in res.data]
        return res

    @property
    def dimension(self) -> int:
        return GLM_MODEL_DIM_MAP[self.model]
