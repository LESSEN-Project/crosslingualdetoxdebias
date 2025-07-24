# Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation
This is the repository for [Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](https://aclanthology.org/2025.findings-acl.145/).

Authors: Vera Neplenbroek, Arianna Bisazza, Raquel Fernández.

![Introduction](intro_figure.png)

## Paper abstract
Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. Our results show that finetuning on curated non-harmful text is more effective for mitigating bias, and finetuning on direct preference optimization (DPO) datasets is more effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods.

## Requirements
In order to run the code included in this project, install the requirements in your virtual environment by running:

```
pip install -r requirements.txt
```
This project was developed using Python 3.12.

This repository contains snippets of code from:
- https://github.com/manon-reusens/multilingual_bias
- https://github.com/for-ai/language-confusion (compute_metrics.py has been taken from this repo, which was released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt). Line 111 has been added to this file.)
- https://github.com/BatsResearch/cross-lingual-detox

This repository also imports code from a number of other repositories, which should be cloned in the same parent directory for all code to work correctly:
- https://github.com/McGill-NLP/bias-bench
- https://github.com/slds-lmu/stereotypes-multi
- https://github.com/Veranep/MBBQ

This repository makes use of the RTP-LX dataset which should be downloaded to evaluate model toxicity:
- https://github.com/microsoft/RTP-LX

This repository depends on the Language Model Evaluation Harness to obtain Global-MMLU results:
- https://github.com/EleutherAI/lm-evaluation-harness

## Using this repository
- `bias_eval.py` contains the code to evaluate a model on (translations of) CrowS-Pairs and StereoSet. Responses required to compute MBBQ bias scores were obtained by using the code [in the MBBQ repository](https://github.com/Veranep/MBBQ).

  Example usage:
  ```
  python bias_eval.py -model Meta-Llama-3.1-8B-Instruct -dataset crowspairs
  ```
- `bilingual_sent_sim.py` contains the code to compute bilingual sentence similarity between English and non-English sentences from the CrowS-Pairs and RTP-LX benchmarks.

  Example usage:
  ```
  python bilingual_sent_sim.py -model Meta-Llama-3.1-8B-Instruct -dataset crowspairs
  ```
- `compute_metrics.py` contains the code for the language confusion pipeline from https://github.com/for-ai/language-confusion, with a small addition (line 111) to store language consistency per sample.
- `generation_eval.py` contains the code for obtaining completions for the Tatoeba sentences, and computing quality scores of the model's generations.

  Example usage:
  ```
  python generation_eval.py -model Meta-Llama-3.1-8B-Instruct
  ```
- `process_tatoeba.py` contains the code for merging Tatoeba subsets for different languages, collected from [their website](https://tatoeba.org/en).

  Example usage:
  ```
  python process_tatoeba.py
  ```
- `qlora.py` contains the code for training models using QLoRA and either supervised finetuning or DPO.

  Example usage:
  ```
  python qlora.py -model Meta-Llama-3.1-8B-Instruct -token <HF_TOKEN> -seed 42 -dataset panda
  ```
- `subword_overlap.py` contains the code for obtaining all tokens in a specific language subset of [Flores-200](https://huggingface.co/datasets/Muennighoff/flores200), using one model's tokenizer.

  Example usage:
  ```
  python subword_overlap.py -model Meta-Llama-3.1-8B-Instruct
  ```
- `toxicity_eval.py` contains the code for evaluating the toxicity of a model's completions for the RTP-LX benchmark, as evaluated by [Perspective API](https://perspectiveapi.com/).

  Example usage:
  ```
  python toxicity_eval.py -model Meta-Llama-3.1-8B-Instruct -key <PERSPECTIVE_API_KEY>
  ```
- `visualize_results.ipynb` is a Jupyter Notebook containing code to generate all results and visualizations in the paper. Note that this notebook is depended on toxicity, bias, question-answering and language generation quality scores obtained by running the other files in this project or its requirements.

## Citation
If you use the code in this repository, please cite the following paper:
```bibtex
@inproceedings{neplenbroek-etal-2025-cross,
    title = "Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual {LLM}s: An Extensive Investigation",
    author = "Neplenbroek, Vera  and
      Bisazza, Arianna  and
      Fern{\'a}ndez, Raquel",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.145/",
    pages = "2805--2830",
    ISBN = "979-8-89176-256-5",
    abstract = "Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model{'}s bias and toxicity, but also on its ability to produce fluent and diverse text. We reduce biases by finetuning on curated non-harmful text, but find only direct preference optimization to be effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model{'}s pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods."
}
