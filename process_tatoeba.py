from collections import defaultdict
import os

import random
from tqdm import tqdm
import pandas as pd
import pickle

random.seed(42)

if __name__ == "__main__":
    # Load language data
    german = (
        pd.read_csv(
            "Sentence pairs in German-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["de_id", "de_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    spanish = (
        pd.read_csv(
            "Sentence pairs in Spanish-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["es_id", "es_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    turkish = (
        pd.read_csv(
            "Sentence pairs in Turkish-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["tr_id", "tr_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    french = (
        pd.read_csv(
            "Sentence pairs in French-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["fr_id", "fr_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    italian = (
        pd.read_csv(
            "Sentence pairs in Italian-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            on_bad_lines="warn",
            names=["it_id", "it_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    maltese = (
        pd.read_csv(
            "Sentence pairs in Maltese-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["mt_id", "mt_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    chinese = (
        pd.read_csv(
            "Sentence pairs in Mandarin Chinese-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["zh_id", "zh_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    arabic = (
        pd.read_csv(
            "Sentence pairs in Arabic-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["ar_id", "ar_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    korean = (
        pd.read_csv(
            "Sentence pairs in Korean-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["ko_id", "ko_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    catalan = (
        pd.read_csv(
            "Sentence pairs in Catalan-English - 2024-09-11.tsv",
            sep="\t",
            header=None,
            names=["ca_id", "ca_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    czech = (
        pd.read_csv(
            "Sentence pairs in Czech-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["cs_id", "cs_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    hindi = (
        pd.read_csv(
            "Sentence pairs in Hindi-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["hi_id", "hi_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    indonesian = (
        pd.read_csv(
            "Sentence pairs in Indonesian-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["id_id", "id_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    japanese = (
        pd.read_csv(
            "Sentence pairs in Japanese-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["ja_id", "ja_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    dutch = (
        pd.read_csv(
            "Sentence pairs in Dutch-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            on_bad_lines="warn",
            names=["nl_id", "nl_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    polish = (
        pd.read_csv(
            "Sentence pairs in Polish-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["pl_id", "pl_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    portuguese = (
        pd.read_csv(
            "Sentence pairs in Portuguese-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["pt_id", "pt_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    russian = (
        pd.read_csv(
            "Sentence pairs in Russian-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            on_bad_lines="warn",
            names=["ru_id", "ru_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    swedish = (
        pd.read_csv(
            "Sentence pairs in Swedish-English - 2024-09-13.tsv",
            sep="\t",
            header=None,
            names=["sv_id", "sv_text", "en_id", "en_text"],
        )
        .drop_duplicates("en_id")
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)[:1000]
    )
    # merge language data
    selected_data = (
        (
            german.merge(spanish, how="outer", on=["en_id", "en_text"])
            .merge(turkish, how="outer", on=["en_id", "en_text"])
            .merge(french, how="outer", on=["en_id", "en_text"])
            .merge(italian, how="outer", on=["en_id", "en_text"])
            .merge(maltese, how="outer", on=["en_id", "en_text"])
            .merge(chinese, how="outer", on=["en_id", "en_text"])
            .merge(arabic, how="outer", on=["en_id", "en_text"])
            .merge(korean, how="outer", on=["en_id", "en_text"])
            .merge(catalan, how="outer", on=["en_id", "en_text"])
            .merge(czech, how="outer", on=["en_id", "en_text"])
            .merge(hindi, how="outer", on=["en_id", "en_text"])
            .merge(indonesian, how="outer", on=["en_id", "en_text"])
            .merge(japanese, how="outer", on=["en_id", "en_text"])
            .merge(dutch, how="outer", on=["en_id", "en_text"])
            .merge(polish, how="outer", on=["en_id", "en_text"])
            .merge(portuguese, how="outer", on=["en_id", "en_text"])
            .merge(russian, how="outer", on=["en_id", "en_text"])
            .merge(swedish, how="outer", on=["en_id", "en_text"])
        )
        .drop(
            columns=[
                "de_id",
                "es_id",
                "fr_id",
                "tr_id",
                "it_id",
                "mt_id",
                "zh_id",
                "ar_id",
                "ko_id",
                "ca_id",
                "cs_id",
                "hi_id",
                "id_id",
                "ja_id",
                "nl_id",
                "pl_id",
                "pt_id",
                "ru_id",
                "sv_id",
            ],
        )
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(selected_data.shape)
    # save data frame
    with open("df_tatoeba_lang_check.pkl", "wb") as outfile:
        pickle.dump(selected_data, outfile)
