# Gemma로 Chatbot 만들기

오픈 모델인 [google/gemma-7b](https://huggingface.co/google/gemma-7b)을 이용한 Chatbot을 만들고자 합니다.

[Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/)와 같이 Gemini의 기술을 활용한 가볍고(lightweight), 최신의(state-of-the-art) 오픈 모델입니다. [2B와 7B의 모델](https://www.kaggle.com/models/google/gemma)로 사전학습되었고, [instruction-tuned](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)을 이용하고 있습니다. [256k token 크기](https://www.promptingguide.ai/models/gemma)을 제공합니다.

[Gemma 설정](https://ai.google.dev/gemma/docs/setup?hl=ko)에 따라 테스트를 해볼 수 있습니다. 

## Prompt

[Zero-shot Prompting with System Prompt](https://www.promptingguide.ai/models/gemma)는 아래와 같습니다.

```text
<start_of_turn>user
Answer the following question in a concise and informative manner:
 
Explain why the sky is blue<end_of_turn>
<start_of_turn>model
```

[Gemma Notebook](https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/get_started.ipynb)의 구현코드는 아래와 같습니다.

```python

import os

os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch".

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

gemma_lm.compile(sampler="top_k")
gemma_lm.generate("What is the meaning of life?", max_length=64)
```

[Gemma meets LangChain - summarize kaggle writeups](https://www.kaggle.com/code/toshik/gemma-meets-langchain-summarize-kaggle-writeups)

```python
import os

os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

import keras
import keras_nlp

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_2b_en")

print(gemma_lm.generate("hi, how are you doing?", max_length=256))

# Define the custom model for LangChain
from typing import Any, Optional, List, Mapping
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

class GemmaLC(LLM):

    model: Any = None
    n: int = None

    def __init__(self, keras_model, n):
        super(GemmaLC, self).__init__()
        self.model = keras_model
        self.n = n

    @property
    def _llm_type(self) -> str:
        return "Gemma"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:

        generated = self.model.generate(prompt, max_length=self.n)

        # post-processing to extract the result of summarization
        split_string = generated.split("CONCISE SUMMARY:", 1)
        if len(split_string) > 1:            
            return split_string[1].lstrip('\n')
        else:
            return generated.lstrip('\n')

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

gemma_lc = GemmaLC(gemma_lm, 1024)

from langchain_core.prompts import PromptTemplate

prompt_template = """Write a concise summary of the following kaggle solution writeup:


"{text}"


CONCISE SUMMARY:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(
    gemma_lc, chain_type="map_reduce",
    map_prompt=PROMPT,
    combine_prompt=PROMPT
)

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def summarize(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    paged_docs = [Document(page_content=t) for t in texts]
    return chain.invoke(paged_docs)
```
