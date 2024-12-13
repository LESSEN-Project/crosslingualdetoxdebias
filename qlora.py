import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import login
from peft import (
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import os
from trl import DPOTrainer, SFTTrainer


def tokenize_panda(element):
    outputs = tokenizer(
        element["perturbed"],
        truncation=True,
        max_length=context_length,
        padding="max_length",
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for input_ids in outputs["input_ids"]:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}


def tokenize_jigsaw(element):
    outputs = tokenizer(
        element["comment_text"],
        truncation=True,
        max_length=context_length,
        padding="max_length",
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for input_ids in outputs["input_ids"]:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}


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
        "-token",
        type=str,
        default="",
        help="Huggingface token that grants access to Llama model",
    )
    parser.add_argument(
        "-seed",
        type=int,
        default="",
        help="Random seed",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        choices=[
            "panda",
            "biasdpo",
            "toxicdpo",
            "jigsaw",
        ],
        default="",
        help="Dataset to fine-tune on",
    )
    args = parser.parse_args()
    login(args.token)
    # load configs for QLora
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        bias="none",
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if "gemma" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    # prepare model
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.enable_input_require_grads()
    print("MODEL LOADED")
    ################################################################################
    #    LOAD DATA
    ################################################################################
    # SFT training
    if args.dataset in ["panda", "jigsaw"]:
        if args.dataset == "panda":
            raw_datasets = load_dataset("facebook/panda")
            valid_name = "validation"
            tokenize_func = tokenize_panda
        elif args.dataset == "jigsaw":
            raw_datasets = load_dataset(
                "jigsaw_unintended_bias",
                data_dir="jigsaw-unintended-bias-in-toxicity-classification",
            )
            raw_datasets = raw_datasets.filter(
                lambda example: example["target"] < 0.1
            )
            valid_name = "test_private_leaderboard"
            tokenize_func = tokenize_jigsaw
        context_length = 512

        tokenized_datasets = raw_datasets.map(
            tokenize_func,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        print("DATA LOADED")
        training_args = TrainingArguments(
            output_dir=f"{args.model}_lora_{args.dataset}",
            num_train_epochs=1,
            save_total_limit=5,
            eval_strategy="steps",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=150,
            weight_decay=0.001,
            metric_for_best_model="eval_loss",
            fp16=True,
            remove_unused_columns=False,
            logging_steps=500,
            eval_steps=500,
            save_steps=500,
            save_strategy="steps",
            learning_rate=3e-4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            load_best_model_at_end=True,
            seed=args.seed,
        )
        model = get_peft_model(model, peft_config)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=tokenized_datasets["train"].select(range(94966)),
            eval_dataset=tokenized_datasets[valid_name].select(range(10551)),
            data_collator=data_collator,
        )
        print("START TRAINING")
        trainer.train()
        model = model.merge_and_unload()
        model.save_pretrained(f"{args.model}_lora_{args.dataset}_model")
    # DPO training
    elif args.dataset in ["biasdpo", "toxicdpo"]:
        if args.dataset == "biasdpo":
            dataset = load_dataset("ahmedallam/BiasDPO")[
                "train"
            ].train_test_split(test_size=0.05, seed=args.seed)
            num_epochs = 20
            n_steps = 25
        elif args.dataset == "toxicdpo":
            dataset = pd.concat(
                [
                    pd.read_json(
                        f"toxicity_pairwise/split_{i}.jsonl", lines=True
                    )[["prompt_text", "unpert_gen_text", "pert_gen_text"]]
                    for i in range(6)
                ],
                ignore_index=True,
            ).rename(
                columns={
                    "prompt_text": "prompt",
                    "unpert_gen_text": "chosen",
                    "pert_gen_text": "rejected",
                }
            )
            num_epochs = 1
            n_steps = 100
            dataset = Dataset.from_pandas(dataset).train_test_split(
                test_size=0.05, seed=42
            )
        training_args = TrainingArguments(
            output_dir=f"{args.model}_lora_{args.dataset}",
            optim="rmsprop",
            learning_rate=1e-5,
            save_total_limit=5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            remove_unused_columns=False,
            num_train_epochs=num_epochs,
            metric_for_best_model="eval_loss",
            logging_steps=n_steps,
            save_steps=n_steps,
            fp16=True,
            eval_steps=n_steps,
            eval_strategy="steps",
            lr_scheduler_type="constant_with_warmup",
            save_strategy="steps",
            warmup_steps=150,
            weight_decay=0.05,
            max_grad_norm=10,
            load_best_model_at_end=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            seed=args.seed,
        )
        trainer = DPOTrainer(
            model,
            beta=0.1,
            max_prompt_length=128,
            max_length=512,
            peft_config=peft_config,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )
        print("START TRAINING")
        trainer.train()
        model = trainer.model.merge_and_unload()
        model.save_pretrained(f"{args.model}_lora_{args.dataset}_model")
