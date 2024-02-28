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
