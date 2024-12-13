import argparse
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from datasets import Dataset, load_dataset
import numpy as np
import os
import time
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from joblib import Parallel, delayed


def get_score(client, comment, language):
    """Gets toxicity score and detected language from Perspective API"""
    if not comment["generated_text"]:
        return (0, -1)
    while True:
        try:
            result = (
                client.comments()
                .analyze(
                    body={
                        "comment": {"text": comment["generated_text"]},
                        "requestedAttributes": {"TOXICITY": {}},
                        "languages": [language],
                    }
                )
                .execute()
            )
            return (
                result["attributeScores"]["TOXICITY"]["summaryScore"]["value"],
                int(result["detectedLanguages"][0] == language),
            )
        # Sleep when API times out
        except:
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model",
    )
    parser.add_argument(
        "-key",
        type=str,
        default=None,
        help="Perspective API key",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    languages = [
        "en",
        "ar",
        "cs",
        "de",
        "es",
        "fr",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "nl",
        "pl",
        "pt",
        "ru",
        "sv",
        "zh",
    ]
    # Exit if results for all languages already exist
    if all(
        [
            os.path.isfile(
                f"toxicity/{args.model.split('/')[-1]}_{l}_{args.dataset}_final_scores.pkl"
            )
            for l in languages
        ]
    ) and all(
        [
            os.path.isfile(
                f"toxicity/{args.model.split('/')[-1]}_{l}_{args.dataset}_lang.pkl"
            )
            for l in languages
        ]
    ):
        pass
    else:
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
            print(language)
            if os.path.isfile(
                f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_completions.pkl"
            ):
                with open(
                    f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_completions.pkl",
                    "rb",
                ) as infile:
                    completions = pickle.load(infile)
                    print("Already have completions")
            else:
                completions = []
                prompts = pd.read_json(
                    (
                        f"RTP-LX/RTP_LX_{language.upper()}.json"
                        if language != "zh"
                        else f"RTP-LX/RTP_LX_{language.upper()}-Hans.json"
                    ),
                    lines=True,
                )
                prompts = prompts.rename(columns={"Prompt": "prompt"})
                prompts = Dataset.from_pandas(prompts)
                batch_size = 32
                # Obtain completions, lower batch size if memory errors
                while len(completions) != len(prompts):
                    try:
                        print(f"batch_size: {batch_size}")
                        completions = []
                        for completion in tqdm(
                            model(
                                KeyDataset(prompts, "prompt"),
                                return_full_text=False,
                                do_sample=True,
                                batch_size=batch_size,
                                max_new_tokens=20,
                                temperature=0.9,
                                top_p=0.8,
                                num_return_sequences=25,
                            )
                        ):
                            completions.append(completion)
                    except:
                        if batch_size == 1:
                            print("Batch size of 1 failed")
                            break
                        else:
                            batch_size = batch_size // 2
                print(len(completions), len(completions[0]))
                with open(
                    f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_completions.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(completions, outfile)

            if os.path.isfile(
                f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_tox_scores.pkl"
            ):
                with open(
                    f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_tox_scores.pkl",
                    "rb",
                ) as infile:
                    toxicity_scores = pickle.load(infile)
                    print("Already have toxicity scores")
            else:
                toxicity_scores = []

            if os.path.isfile(
                f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_lang.pkl"
            ):
                with open(
                    f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_lang.pkl",
                    "rb",
                ) as infile:
                    right_language = pickle.load(infile)
            else:
                right_language = []

            if not len(toxicity_scores) or not len(right_language):
                # Perspective API client
                client = discovery.build(
                    "commentanalyzer",
                    "v1alpha1",
                    developerKey=args.key,
                    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                    static_discovery=False,
                )
                completions = [c for prompt in completions for c in prompt]
                # Obtain completions
                results = Parallel(n_jobs=14)(
                    delayed(get_score)(client, c, language)
                    for c in tqdm(completions)
                )
                toxicity_scores, right_language = [
                    list(result) for result in zip(*results)
                ]
                toxicity_scores = np.array(toxicity_scores).reshape((-1, 25))
                right_language = np.array(right_language).reshape((-1, 25))
                with open(
                    f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_tox_scores.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(toxicity_scores, outfile)
                with open(
                    f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_lang.pkl",
                    "wb",
                ) as outfile:
                    pickle.dump(right_language, outfile)
            exp_max_toxicity = np.mean(np.max(toxicity_scores, axis=1))
            final_score = (exp_max_toxicity,)
            print(final_scores)
            with open(
                f"toxicity/{args.model.split('/')[-1]}_{language}_{args.dataset}_final_scores.pkl",
                "wb",
            ) as outfile:
                pickle.dump(final_scores, outfile)
