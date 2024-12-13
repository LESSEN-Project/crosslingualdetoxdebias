import argparse
from compute_metrics import compute_all_metrics
import datasets
from datasets import Dataset, load_dataset
import evaluate
from evaluate import logging
import numpy as np
import os
import pandas as pd
import sys
import pickle
import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.pipelines.pt_utils import KeyDataset


def compute_diversity(prompt, generation, tokenizer):
    # adapted from https://github.com/BatsResearch/cross-lingual-detox/blob/main/xg/eval/metric_diversity.py
    # calculate diversity across generations for every prompt
    if generation == "":
        return 0, 0, 0
    dist1, dist2, dist3 = [], [], []
    input_unigrams, input_bigrams, input_trigrams = set(), set(), set()
    o = tokenizer(prompt)["input_ids"]
    ow = [tokenizer.decode(x, skip_special_tokens=True) for x in o]
    ow = [x for x in ow if x]
    input_unigrams.update(ow)
    for i in range(len(ow) - 1):
        input_bigrams.add(ow[i] + "_" + ow[i + 1])
    for i in range(len(ow) - 2):
        input_trigrams.add(ow[i] + "_" + ow[i + 1] + "_" + ow[i + 2])

    total_tokens = 0
    unigrams, bigrams, trigrams = set(), set(), set()
    o = tokenizer(generation)["input_ids"]
    ow = [tokenizer.decode(x, skip_special_tokens=True) for x in o]
    ow = [x for x in ow if x]

    unigrams.update(ow)
    for i in range(len(ow) - 1):
        bigrams.add(ow[i] + "_" + ow[i + 1])
    for i in range(len(ow) - 2):
        trigrams.add(ow[i] + "_" + ow[i + 1] + "_" + ow[i + 2])

    total_tokens += len(ow)
    dist1 = len(unigrams - input_unigrams) / total_tokens
    dist2 = len(bigrams - input_bigrams) / total_tokens
    dist3 = len(trigrams - input_trigrams) / total_tokens
    return dist1, dist2, dist3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = pd.read_pickle("df_tatoeba_lang_check.pkl")
    languages = [
        "de",
        "es",
        "fr",
        "tr",
        "en",
        "it",
        "mt",
        "zh",
        "ar",
        "ko",
        "ca",
        "cs",
        "hi",
        "id",
        "ja",
        "nl",
        "pl",
        "pt",
        "ru",
        "sv",
    ]
    # load existing results if any
    if os.path.isfile(
        f"lang_confusion/{args.model.split('/')[-1]}_results.pkl"
    ):
        with open(
            f"lang_confusion/{args.model.split('/')[-1]}_results.pkl",
            "rb",
        ) as infile:
            results = pickle.load(infile)
    else:
        results = {}
    if os.path.isfile(f"diversity/{args.model.split('/')[-1]}_results.pkl"):
        with open(
            f"diversity/{args.model.split('/')[-1]}_results.pkl",
            "rb",
        ) as infile:
            diversity_results = pickle.load(infile)
    else:
        diversity_results = {}

    if os.path.isfile(f"perplexity/{args.model.split('/')[-1]}_results.pkl"):
        with open(
            f"perplexity/{args.model.split('/')[-1]}_results.pkl",
            "rb",
        ) as infile:
            perplexity_results = pickle.load(infile)
    else:
        perplexity_results = {}

    # if not all completions have been generated, load the model
    if not all(
        [
            os.path.isfile(
                f"lang_confusion/{args.model.split('/')[-1]}_{language}_completions.pkl"
            )
            for language in languages
        ]
    ):
        if "gemma" in args.model:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )
            if "gemma-2-9b-it" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/gemma-2-9b-it"
                )
            elif "gemma-2-9b" in args.model:
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
            elif "gemma-2-2b-it" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/gemma-2-2b-it"
                )
            elif "gemma-2-2b" in args.model:
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
            model = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            try:
                model = pipeline(
                    "text-generation",
                    model=args.model,
                    tokenizer=args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            except:
                if "Meta-Llama-3-8B-Instruct" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                elif "Meta-Llama-3-8B" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="meta-llama/Meta-Llama-3-8B",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                elif "Meta-Llama-3.1-8B-Instruct" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                elif "Meta-Llama-3.1-8B" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="meta-llama/Meta-Llama-3.1-8B",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                elif "Mistral-7B-v0.3" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="mistralai/Mistral-7B-v0.3",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                elif "aya-23-8B" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="CohereForAI/aya-23-8B",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                elif "aya-expanse-8b" in args.model:
                    model = pipeline(
                        "text-generation",
                        model=args.model,
                        tokenizer="CohereForAI/aya-expanse-8b",
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
        if not model.tokenizer.pad_token_id:
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        for language in languages:
            # obtain completions
            if not os.path.isfile(
                f"lang_confusion/{args.model.split('/')[-1]}_{language}_completions.pkl"
            ):
                print(language)
                lang_responses = []
                lang_data = dataset.dropna(subset=[f"{language}_text"])
                lang_data = Dataset.from_pandas(lang_data)
                if language == "en":
                    lang_data = lang_data.select(range(1000))
                batch_size = 64
                while len(lang_responses) != len(lang_data):
                    # decrease batch size upon memory error
                    try:
                        print(f"batch_size: {batch_size}")
                        lang_responses = []
                        for lang_response in tqdm(
                            model(
                                KeyDataset(lang_data, f"{language}_text"),
                                return_full_text=False,
                                batch_size=batch_size,
                                max_new_tokens=100,
                                do_sample=False,
                                num_beams=1,
                            )
                        ):
                            lang_responses.append(
                                lang_response[0]["generated_text"]
                            )
                    except:
                        if batch_size == 1:
                            print("Batch size of 1 failed")
                            break
                        else:
                            batch_size = batch_size // 2
                with open(
                    f"lang_confusion/{args.model.split('/')[-1]}_{language}_completions.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(lang_responses, outfile)
    else:
        # if completions were already obtained, compute language generation metrics
        print("Already have completions")
        perpl_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/mt5-xl",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        perpl_model.eval()
        perpl_tokenizer = AutoTokenizer.from_pretrained(
            "google/mt5-xl", legacy=False
        )
        if "gemma-2-9b-it" in args.model:
            tokenizer_name = "google/gemma-2-9b-it"
        elif "gemma-2-9b" in args.model:
            tokenizer_name = "google/gemma-2-9b"
        elif "gemma-2-2b-it" in args.model:
            tokenizer_name = "google/gemma-2-2b-it"
        elif "gemma-2-2b" in args.model:
            tokenizer_name = "google/gemma-2-2b"
        elif "Meta-Llama-3-8B-Instruct" in args.model:
            tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif "Meta-Llama-3-8B" in args.model:
            tokenizer_name = "meta-llama/Meta-Llama-3-8B"
        elif "Meta-Llama-3.1-8B-Instruct" in args.model:
            tokenizer_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif "Meta-Llama-3.1-8B" in args.model:
            tokenizer_name = "meta-llama/Meta-Llama-3.1-8B"
        elif "Mistral-7B-v0.3" in args.model:
            tokenizer_name = "mistralai/Mistral-7B-v0.3"
        elif "Mistral-7B-Instruct-v0.3" in args.model:
            tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"
        elif "aya-23-8B" in args.model:
            tokenizer_name = "CohereForAI/aya-23-8B"
        elif "aya-expanse-8b" in args.model:
            tokenizer_name = "CohereForAI/aya-expanse-8b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        for language in languages:
            # load original prompts and responses
            lang_data = dataset.dropna(subset=[f"{language}_text"])
            lang_data = Dataset.from_pandas(lang_data)
            if language == "en":
                lang_data = lang_data.select(range(1000))
            with open(
                f"lang_confusion/{args.model.split('/')[-1]}_{language}_completions.pkl",
                "rb",
            ) as infile:
                lang_responses = pickle.load(infile)

            # run language confusion pipeline
            if (
                "tatoeba",
                language,
            ) not in results or "per_compl_acc" not in results[
                ("tatoeba", language)
            ]:
                responses = []
                for i, response in enumerate(lang_responses):
                    responses.append(
                        {
                            "source": "tatoeba",
                            "language": language,
                            "completion": response,
                        }
                    )
                lang_results = compute_all_metrics(responses)
                results[("tatoeba", language)] = lang_results[
                    ("tatoeba", language)
                ]
                results[("all", language)] = lang_results[("all", language)]
                print(language, lang_results[("tatoeba", language)]["acc"])
                with open(
                    f"lang_confusion/{args.model.split('/')[-1]}_results.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(results, outfile)

            if (
                language not in perplexity_results
                or "selected" not in perplexity_results[language]
            ):
                # get perplexity scores from mt5-xl
                ppls = []
                selected_ppls = []
                for i in tqdm(range(len(lang_data))):
                    prompt = lang_data[f"{language}_text"][i]
                    generated_text = lang_responses[i]
                    full_text = prompt + generated_text
                    full_input_ids = perpl_tokenizer.encode(
                        full_text, return_tensors="pt"
                    ).to(device)

                    full_loss = perpl_model(
                        full_input_ids, labels=full_input_ids
                    )[0] * (full_input_ids.shape[1] - 1)

                    prompt_input_ids = perpl_tokenizer.encode(
                        prompt, return_tensors="pt"
                    ).to(device)
                    prompt_loss = perpl_model(
                        prompt_input_ids, labels=prompt_input_ids
                    )[0] * (prompt_input_ids.shape[1] - 1)

                    loss = (full_loss - prompt_loss) / (
                        full_input_ids.shape[1] - prompt_input_ids.shape[1]
                    )

                    ppl = np.exp(loss.item())
                    ppls.append(ppl)
                    if results[("tatoeba", language)]["per_compl_acc"][i] == 1:
                        selected_ppls.append(ppl)
                    else:
                        selected_ppls.append("wrong_lang")
                perplexity_results[language] = {
                    "all": ppls,
                    "mean": np.mean(ppls),
                    "median": np.median(ppls),
                    "selected": selected_ppls,
                }
                print(
                    language,
                    perplexity_results[language]["mean"],
                    perplexity_results[language]["median"],
                )
                with open(
                    f"perplexity/{args.model.split('/')[-1]}_results.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(perplexity_results, outfile)

            if (
                language not in diversity_results
                or "selected" not in diversity_results[language]
            ):
                # compute diversity of generations
                all_dist1, all_dist2, all_dist3, selected1 = [], [], [], []
                for i, response in enumerate(lang_responses):
                    dist1, dist2, dist3 = compute_diversity(
                        lang_data[f"{language}_text"][i],
                        response,
                        tokenizer,
                    )
                    all_dist1.append(dist1)
                    all_dist2.append(dist2)
                    all_dist3.append(dist3)
                    if results[("tatoeba", language)]["per_compl_acc"][i] == 1:
                        selected1.append(dist1)
                    else:
                        selected1.append("wrong_lang")
                diversity_results[language] = {
                    "unigrams": np.mean(all_dist1),
                    "bigrams": np.mean(all_dist2),
                    "trigrams": np.mean(all_dist3),
                    "selected": selected1,
                }
                print(language, diversity_results[language]["unigrams"])
                with open(
                    f"diversity/{args.model.split('/')[-1]}_results.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(diversity_results, outfile)
