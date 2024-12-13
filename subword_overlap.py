import argparse
from datasets import load_dataset
import os
import pickle
from transformers import AutoTokenizer


def preprocess_function(examples):
    return tokenizer(examples["sentence"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=str,
        default=None,
        help="Model",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for language in [
        "eng_Latn",
        "fra_Latn",
        "deu_Latn",
        "spa_Latn",
        "tur_Latn",
        "cat_Latn",
        "ita_Latn",
        "mlt_Latn",
        "zho_Hans",
        "kor_Hang",
        "arb_Arab",
        "ces_Latn",
        "hin_Deva",
        "ind_Latn",
        "jpn_Jpan",
        "nld_Latn",
        "pol_Latn",
        "por_Latn",
        "rus_Cyrl",
        "swe_Latn",
    ]:
        if os.path.isfile(
            f"token_overlap/{args.model.split('/')[-1]}_{language}_tokens.pkl"
        ):
            continue
        dataset = load_dataset("Muennighoff/flores200", language)
        tokens = dataset.map(preprocess_function, batched=True)
        # Save list of tokens in Flores200
        with open(
            f"token_overlap/{args.model.split('/')[-1]}_{language}_tokens.pkl",
            "wb",
        ) as outfile:
            pickle.dump(
                tokens["dev"]["input_ids"] + tokens["devtest"]["input_ids"],
                outfile,
            )
