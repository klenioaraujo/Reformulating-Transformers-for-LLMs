---
language: en
license: mit
---

# GPT-2 Medium

## Model Details

**Model Description:** GPT-2 Medium is the **355M parameter** version of GPT-2, a transformer-based language model created and released by OpenAI. The model is a pretrained model on English language using a causal language modeling (CLM) objective. 

- **Developed by:** OpenAI, see [associated research paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [GitHub repo](https://github.com/openai/gpt-2) for model developers.
- **Model Type:** Transformer-based language model
- **Language(s):** English
- **License:** [Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE)
- **Related Models:** [GPT2](https://huggingface.co/gpt2), [GPT2-Large](https://huggingface.co/gpt2-large) and [GPT2-XL](https://huggingface.co/gpt2-xl)
- **Resources for more information:**
  - [Research Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - [OpenAI Blog Post](https://openai.com/blog/better-language-models/)
  - [GitHub Repo](https://github.com/openai/gpt-2)
  - [OpenAI Model Card for GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md)
  - Test the full generation capabilities here: https://transformer.huggingface.co/doc/gpt2-large

## How to Get Started with the Model 

Use the code below to get started with the model. You can use this model directly with a pipeline for text generation. Since the generation relies on some randomness, we
set a seed for reproducibility:

```python
>>> from transformers import pipeline, set_seed
>>> generator = pipeline('text-generation', model='gpt2-medium')
>>> set_seed(42)
>>> generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

[{'generated_text': "Hello, I'm a language model, I'm a language. I'm a compiler, I'm a parser, I'm a server process. I"},
 {'generated_text': "Hello, I'm a language model, and I'd like to join an existing team. What can I do to get started?\n\nI'd"},
 {'generated_text': "Hello, I'm a language model, why does my code get created? Can't I just copy it? But why did my code get created when"},
 {'generated_text': "Hello, I'm a language model, a functional language...\n\nI'm a functional language. Is it hard? A little, yes. But"},
 {'generated_text': "Hello, I'm a language model, not an object model.\n\nIn a nutshell, I need to give me objects from which I can get"}]
```

Here is how to use this model to get the features of a given text in PyTorch:

```python
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import GPT2Tokenizer, TFGPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = TFGPT2Model.from_pretrained('gpt2-medium')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Uses

#### Direct Use

In their [model card about GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md), OpenAI wrote: 

> The primary intended users of these models are AI researchers and practitioners.
> 
> We primarily imagine these language models will be used by researchers to better understand the behaviors, capabilities, biases, and constraints of large-scale generative language models.

#### Downstream Use

In their [model card about GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md), OpenAI wrote: 

> Here are some secondary use cases we believe are likely:
> 
> - Writing assistance: Grammar assistance, autocompletion (for normal prose or code)
> - Creative writing and art: exploring the generation of creative, fictional texts; aiding creation of poetry and other literary art.
> - Entertainment: Creation of games, chat bots, and amusing generations.

#### Misuse and Out-of-scope Use

In their [model card about GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md), OpenAI wrote: 

> Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases that require the generated text to be true.
> 
> Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race, and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of caution around use cases that are sensitive to biases around human attributes.

## Risks, Limitations and Biases

**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propogate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). 

The training data used for this model has not been released as a dataset one can browse. We know it contains a lot of unfiltered content from the internet, which is far from neutral. Predictions generated by the model can include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups. For example:

```python
>>> from transformers import pipeline, set_seed
>>> generator = pipeline('text-generation', model='gpt2-medium')
>>> set_seed(42)
>>> generator("The man worked as a", max_length=10, num_return_sequences=5)

[{'generated_text': 'The man worked as a security guard in a military'},
 {'generated_text': 'The man worked as a salesman in Mexico and eventually'},
 {'generated_text': 'The man worked as a supervisor at the department for'},
 {'generated_text': 'The man worked as a cleaner for the same corporation'},
 {'generated_text': 'The man worked as a barman and was involved'}]

>>> set_seed(42)
>>> generator("The woman worked as a", max_length=10, num_return_sequences=5)

[{'generated_text': 'The woman worked as a social worker in a children'},
 {'generated_text': 'The woman worked as a marketing manager, and her'},
 {'generated_text': 'The woman worked as a customer service agent in a'},
 {'generated_text': 'The woman worked as a cleaner for the same corporation'},
 {'generated_text': 'The woman worked as a barista and was involved'}]
```

This bias will also affect all fine-tuned versions of this model. Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## Training

#### Training Data

The OpenAI team wanted to train this model on a corpus as large as possible. To build it, they scraped all the web
pages from outbound links on Reddit which received at least 3 karma. Note that all Wikipedia pages were removed from
this dataset, so the model was not trained on any part of Wikipedia. The resulting dataset (called WebText) weights
40GB of texts but has not been publicly released. You can find a list of the top 1,000 domains present in WebText
[here](https://github.com/openai/gpt-2/blob/master/domains.txt).

#### Training Procedure

The model is pretrained on a very large corpus of English data in a self-supervised fashion. This
means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots
of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely,
it was trained to guess the next word in sentences.

More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence,
shifted one token (word or piece of word) to the right. The model uses internally a mask-mechanism to make sure the
predictions for the token `i` only uses the inputs from `1` to `i` but not the future tokens.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks.

The texts are tokenized using a byte-level version of Byte Pair Encoding (BPE) (for unicode characters) and a
vocabulary size of 50,257. The inputs are sequences of 1024 consecutive tokens.

## Evaluation

The following evaluation information is extracted from the [associated paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

#### Testing Data, Factors and Metrics

The model authors write in the [associated paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that:

> Since our model operates on a byte level and does not require lossy pre-processing or tokenization, we can evaluate it on any language model benchmark. Results on language modeling datasets are commonly reported in a quantity which is a scaled or ex- ponentiated version of the average negative log probability per canonical prediction unit - usually a character, a byte, or a word. We evaluate the same quantity by computing the log-probability of a dataset according to a WebText LM and dividing by the number of canonical units. For many of these datasets, WebText LMs would be tested significantly out- of-distribution, having to predict aggressively standardized text, tokenization artifacts such as disconnected punctuation and contractions, shuffled sentences, and even the string <UNK> which is extremely rare in WebText - occurring only 26 times in 40 billion bytes. We report our main results...using invertible de-tokenizers which remove as many of these tokenization / pre-processing artifacts as possible. Since these de-tokenizers are invertible, we can still calculate the log probability of a dataset and they can be thought of as a simple form of domain adaptation. 

#### Results

The model achieves the following results without any fine-tuning (zero-shot):

| Dataset  | LAMBADA | LAMBADA | CBT-CN | CBT-NE | WikiText2 | PTB    | enwiki8 | text8  | WikiText103 | 1BW   |
|:--------:|:-------:|:-------:|:------:|:------:|:---------:|:------:|:-------:|:------:|:-----------:|:-----:|
| (metric) | (PPL)   | (ACC)   | (ACC)  | (ACC)  | (PPL)     | (PPL)  | (BPB)   | (BPC)  | (PPL)       | (PPL) |
|          | 15.60   | 55.48   | 92.35  | 87.1   | 22.76     | 47.33  | 1.01    | 1.06   | 26.37       | 55.72 |

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Unknown
- **Hours used:** Unknown
- **Cloud Provider:** Unknown
- **Compute Region:** Unknown
- **Carbon Emitted:** Unknown

## Technical Specifications

See the [associated paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) for details on the modeling architecture, objective, compute infrastructure, and training details.

## Citation Information

```bibtex
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya and others},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}
```

## Model Card Authors

This model card was written by the Hugging Face team.