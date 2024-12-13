import argparse
import os
import numpy as np
import pandas as pd
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Code from https://github.com/BatsResearch/cross-lingual-detox/blob/main/xg/retrieval/retrieval_acc_save.py


def sentence_embedding_BSM_to_BM(
    hidden_states_BSM: torch.Tensor, attn_mask_BS: torch.Tensor
) -> torch.Tensor:
    """Return sentence embedding average over all legal tokens"""

    assert len(hidden_states_BSM.shape) == 3, len(hidden_states_BSM.shape)
    assert len(attn_mask_BS.shape) == 2, len(attn_mask_BS.shape)
    assert hidden_states_BSM.size(0) == attn_mask_BS.size(
        0
    ) and hidden_states_BSM.size(1) == attn_mask_BS.size(
        1
    ), f"{hidden_states_BSM.shape = } {attn_mask_BS.shape = }"

    # zero out hidden states for token masked out by attn mask
    attn_mask_BSM = attn_mask_BS.unsqueeze(dim=-1).expand_as(hidden_states_BSM)
    # print(f"{attn_mask_BSM = }")
    # element-wise mult
    masked_hidden_states_BSM = attn_mask_BSM * hidden_states_BSM
    # print(f"{masked_hidden_states_BSM = }")

    sum_hidden_states_BM = masked_hidden_states_BSM.sum(dim=1)
    # print(f"{sum_hidden_states_BM = }")
    token_counts_BS = attn_mask_BS.sum(
        dim=1, keepdim=True
    )  # Keeping dimension for broadcasting
    avg_hidden_states_BM = sum_hidden_states_BM / token_counts_BS
    return avg_hidden_states_BM


def avg_se_L(se1_all_layers_LBM, se2_all_layers_LBM, n_layer=None):
    if n_layer is None:
        n_layer = len(se1_all_layers_LBM)

    avg_dist_L = []
    for i in range(n_layer):
        se1_BM = se1_all_layers_LBM[i]
        se2_BM = se2_all_layers_LBM[i]

        # Calculate Euclidean distance for each pair of embeddings in the batch
        distances_B = torch.nn.functional.cosine_similarity(
            se1_BM, se2_BM, dim=1
        )

        # Calculate the average distance for the current layer
        avg_distance = (
            distances_B.mean().item()
        )  # Convert to Python float with `.item()`
        avg_dist_L.append(avg_distance)

    return avg_dist_L


def get_retrieval_acc(en_text, cn_text, batch_size):
    cn_sentence_embeddings_all = []
    en_sentence_embeddings_all = []
    for j in range(0, len(en_text), batch_size):
        en_inputs = tokenizer(
            en_text[j : j + batch_size],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        en_outputs = model(**en_inputs, output_hidden_states=True)

        cn_inputs = tokenizer(
            cn_text[j : j + batch_size],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        cn_outputs = model(**cn_inputs, output_hidden_states=True)

        en_hidden_states = en_outputs.hidden_states
        en_attn_mask = en_inputs.attention_mask

        cn_hidden_states = cn_outputs.hidden_states
        cn_attn_mask = cn_inputs.attention_mask

        assert len(en_hidden_states) == len(cn_hidden_states)

        cn_sentence_embeddings_all_layers_LBM = []
        en_sentence_embeddings_all_layers_LBM = []

        for i in range(len(en_hidden_states)):
            en_hidden_states_BSM = en_hidden_states[i]
            cn_hidden_states_BSM = cn_hidden_states[i]

            en_sentence_embeddings_BM = sentence_embedding_BSM_to_BM(
                en_hidden_states_BSM, en_attn_mask
            )
            cn_sentence_embeddings_BM = sentence_embedding_BSM_to_BM(
                cn_hidden_states_BSM, cn_attn_mask
            )

            en_sentence_embeddings_all_layers_LBM.append(
                en_sentence_embeddings_BM
            )
            cn_sentence_embeddings_all_layers_LBM.append(
                cn_sentence_embeddings_BM
            )
        if not en_sentence_embeddings_all:
            en_sentence_embeddings_all = en_sentence_embeddings_all_layers_LBM
        else:
            for k in range(len(en_sentence_embeddings_all)):
                en_sentence_embeddings_all[k] = torch.cat(
                    [
                        en_sentence_embeddings_all[k],
                        en_sentence_embeddings_all_layers_LBM[k],
                    ],
                    dim=0,
                )
        if not cn_sentence_embeddings_all:
            cn_sentence_embeddings_all = cn_sentence_embeddings_all_layers_LBM
        else:
            for k in range(len(cn_sentence_embeddings_all)):
                cn_sentence_embeddings_all[k] = torch.cat(
                    [
                        cn_sentence_embeddings_all[k],
                        cn_sentence_embeddings_all_layers_LBM[k],
                    ],
                    dim=0,
                )
    avg_se = avg_se_L(
        en_sentence_embeddings_all,
        cn_sentence_embeddings_all,
    )
    return np.mean(avg_se)


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
    # Load model
    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        model = model.to(device)
    except:
        pass
    model.eval()
    torch.set_grad_enabled(False)
    if os.path.isfile(
        f"bilingual_sent_retrieval/{args.dataset}_{args.model.split('/')[-1]}_results.pkl"
    ):
        with open(
            f"bilingual_sent_retrieval/{args.dataset}_{args.model.split('/')[-1]}_results.pkl",
            "rb",
        ) as infile:
            results = pickle.load(infile)
    else:
        results = {}
    # Load corresponding evaluation dataset
    if args.dataset == "stereoset":
        batch_size = 4
        en_text = pd.read_pickle(
            f"../stereotypes-multi/create_dataset/data/intrasentence/df_intrasentence_en.pkl"
        )
        # obtain all English samples
        en_text = pd.concat(
            [
                en_text[en_text["c1_gold_label"] == "stereotype"][
                    "c1_sentence"
                ],
                en_text[en_text["c2_gold_label"] == "stereotype"][
                    "c2_sentence"
                ],
                en_text[en_text["c3_gold_label"] == "stereotype"][
                    "c3_sentence"
                ],
            ]
        ).tolist()
        for language in ["de", "es", "fr", "tr", "kr"]:
            if language in results:
                continue
            # obtain all samples in other language
            cn_text = pd.read_pickle(
                f"../stereotypes-multi/create_dataset/data/intrasentence/df_intrasentence_{language}{'_v2_0' if language == 'de' else ''}.pkl"
            )
            cn_text = pd.concat(
                [
                    cn_text[cn_text["c1_gold_label"] == "stereotype"][
                        "c1_sentence"
                    ],
                    cn_text[cn_text["c2_gold_label"] == "stereotype"][
                        "c2_sentence"
                    ],
                    cn_text[cn_text["c3_gold_label"] == "stereotype"][
                        "c3_sentence"
                    ],
                ]
            ).tolist()
            # compute and save retrieval accuracy / sentence similarity
            retrieval_acc = get_retrieval_acc(en_text, cn_text, batch_size)
            results[language] = retrieval_acc
            print(language, "Retrieval acc:", retrieval_acc)
            with open(
                f"bilingual_sent_retrieval/{args.dataset}_{args.model.split('/')[-1]}_results.pkl",
                "wb",
            ) as outfile:
                pickle.dump(results, outfile)

    if args.dataset == "crowspairs":
        batch_size = 4
        for input_file_name in [
            "ca_ES.csv",
            "zh_CN.csv",
            "mt_MT.csv",
            "it_IT.csv",
            "fr_FR.csv",
            "es_AR.csv",
            "de_DE.csv",
        ]:
            language = input_file_name.split("_")[0]
            if language in results:
                continue
            # load English and other language data
            df_csp = pd.read_csv("crows_data/" + input_file_name)
            df_csp = df_csp[~df_csp["sent_more_en"].isna()]
            en_text = df_csp["sent_more_en"].tolist()
            cn_text = df_csp["sent_more"].tolist()
            # compute and save retrieval accuracy / sentence similarity
            retrieval_acc = get_retrieval_acc(en_text, cn_text, batch_size)
            results[language] = retrieval_acc
            print(language, "Retrieval acc:", retrieval_acc)
            with open(
                f"bilingual_sent_retrieval/{args.dataset}_{args.model.split('/')[-1]}_results.pkl",
                "wb",
            ) as outfile:
                pickle.dump(results, outfile)

    if args.dataset == "rtp-lx":
        batch_size = 4
        # load English data
        en_text = pd.read_json(
            "translated_pairwise_data/dpo_toxicity_pairwise_en_200.jsonl",
            lines=True,
        )["Prompt"].tolist()
        for language in [
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
        ]:
            if language in results:
                continue
            # load other language data
            cn_text = pd.read_json(
                f"translated_pairwise_data/dpo_toxicity_pairwise_{language}_200.jsonl",
                lines=True,
            )["Prompt"].tolist()
            # compute and save retrieval accuracy / sentence similarity
            retrieval_acc = get_retrieval_acc(en_text, cn_text, batch_size)
            results[language] = retrieval_acc
            print(language, "Retrieval acc:", retrieval_acc)
            with open(
                f"bilingual_sent_retrieval/{args.dataset}_{args.model.split('/')[-1]}_results.pkl",
                "wb",
            ) as outfile:
                pickle.dump(results, outfile)
