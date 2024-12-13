import argparse
import sys

sys.path.append("../stereotypes-multi/code")
sys.path.append("../bias-bench/bias_bench/benchmark/crows")

import torch
import numpy as np
import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel
from crows import CrowSPairsRunner
from intrasentence_inference import generative_inference
from evaluation import ScoreStorage, add_example_calculate_score
import pickle


def get_joint_sent_prob(
    model, tokenizer, sent, device, initial_token_probabilities
):
    # from https://github.com/manon-reusens/multilingual_bias/blob/main/bias_bench/benchmark/crows/crows.py
    tokens = tokenizer.encode(sent)
    tokens_tensor = torch.tensor(tokens).to(device).unsqueeze(0)
    output = model(tokens_tensor)[0].softmax(dim=-1)
    joint_sentence_probability = [
        initial_token_probabilities[0, 0, tokens[0]].item()
    ]
    for idx in range(1, len(tokens)):
        joint_sentence_probability.append(
            output[0, idx - 1, tokens[idx]].item()
        )
    score = np.sum([np.log2(i) for i in joint_sentence_probability])
    score /= len(joint_sentence_probability)
    score = np.power(2, score)
    return score


def dict_func():
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=str,
        default=None,
        help="Model",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        default=None,
        help="Dataset",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Exit if results for all languages already exist
    if args.dataset == "crowspairs" and all(
        [
            os.path.isfile(
                f"{args.dataset}/{args.model.split('/')[-1]}_{language}_cspresults.pkl"
            )
            for language in [
                "ar_DZ.csv",
                "ca_ES.csv",
                "zh_CN.csv",
                "mt_MT.csv",
                "it_IT.csv",
                "fr_FR.csv",
                "es_AR.csv",
                "en_US.csv",
                "de_DE.csv",
            ]
        ]
    ):
        sys.exit()
    # Exit if results for all languages already exist
    if args.dataset == "stereoset" and all(
        [
            os.path.isfile(
                f"{args.dataset}/{args.model.split('/')[-1]}_{language}_ssresults.pkl"
            )
            for language in ["de", "en", "es", "fr", "tr", "kr"]
        ]
    ):
        sys.exit()
    # Load model
    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        if "gemma-2-9b-it" in args.model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        elif "gemma-2-9b" in args.model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        elif "gemma-2-2b-it" in args.model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        elif "gemma-2-2b" in args.model:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        except:
            if "Meta-Llama-3-8B-Instruct" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct"
                )
            elif "Meta-Llama-3-8B" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B"
                )
            elif "Meta-Llama-3.1-8B-Instruct" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct"
                )
            elif "Meta-Llama-3.1-8B" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B"
                )
            elif "Mistral-7B-v0.3" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-v0.3"
                )
            elif "aya-23-8B" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "CohereForAI/aya-23-8B"
                )
            elif "aya-expanse-8b" in args.model:
                tokenizer = AutoTokenizer.from_pretrained(
                    "CohereForAI/aya-expanse-8b"
                )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        model = model.to(device)
    except:
        pass
    model.eval()
    if args.dataset == "crowspairs":
        for input_file_name in [
            "ar_DZ.csv",
            "ca_ES.csv",
            "zh_CN.csv",
            "mt_MT.csv",
            "it_IT.csv",
            "fr_FR.csv",
            "es_AR.csv",
            "en_US.csv",
            "de_DE.csv",
        ]:
            results = {}
            language = input_file_name.split("_")[0]
            if os.path.isfile(
                f"{args.dataset}/{args.model.split('/')[-1]}_{language}_cspresults.pkl"
            ):
                continue
            for bias_type in [
                "race-color",
                "socioeconomic",
                "gender",
                "disability",
                "nationality",
                "sexual-orientation",
                "physical-appearance",
                "religion",
                "age",
            ]:
                runner = CrowSPairsRunner(
                    model=model,
                    tokenizer=tokenizer,
                    input_file="crows_data/" + input_file_name,
                    bias_type=bias_type,
                    is_generative=True,
                )
                results[bias_type] = runner()
                print(args.model, language)
            with open(
                f"{args.dataset}/{args.model.split('/')[-1]}_{language}_cspresults.pkl",
                "wb",
            ) as outfile:
                pickle.dump(results, outfile)

    if args.dataset == "stereoset":
        for language in ["de", "en", "es", "fr", "tr", "kr"]:
            if os.path.isfile(
                f"{args.dataset}/{args.model.split('/')[-1]}_{language}_ssresults.pkl"
            ):
                continue
            score_evaluator = ScoreStorage()
            results = defaultdict(dict_func)
            df = pd.read_pickle(
                f"../stereotypes-multi/create_dataset/data/intrasentence/df_intrasentence_{language}{'_v2_0' if language == 'de' else ''}.pkl"
            )
            predictions = generative_inference(
                df_intrasentences=df,
                model=model,
                device=device,
                tokenizer=tokenizer,
                batch_size=2,
                num_workers=1,
            )
            target_col = "target" if language == "en" else "target_original"
            results, score_evaluator = add_example_calculate_score(
                score_evaluator=score_evaluator,
                df=df,
                predictions={"intrasentence": predictions},
                eval_type="intrasentence",
                results=results,
                target_col=target_col,
            )
            print(args.model, language)
            with open(
                f"{args.dataset}/{args.model.split('/')[-1]}_{language}_ssresults.pkl",
                "wb",
            ) as outfile:
                pickle.dump(results, outfile)
            score_evaluator.pretty_print(results)
