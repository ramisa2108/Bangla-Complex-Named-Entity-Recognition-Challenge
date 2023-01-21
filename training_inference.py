# Adapted from huggingface transformers classificaton scripts

import os
from dataclasses import dataclass, field
import torch
from torch import nn
from transformers import Trainer
import numpy as np
from datasets.io.json import JsonDatasetReader

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from normalizer import normalize

EXT2CONFIG = {
    "jsonl": (JsonDatasetReader, {}),
    "json": (JsonDatasetReader, {})
}



def main(train_file_path, validation_file_path, test_file_path, train=False):
    get_ext = lambda path: os.path.basename(path).split(".")[-1]


    data_files = {
        "train": train_file_path, 
        "validation": validation_file_path,
        "test": test_file_path
    }

    data_files = {k: v for k, v in data_files.items() if v is not None}

    if not data_files:
        raise ValueError("No valid input file found.")

    selected_ext = get_ext(list(data_files.values())[0])


    dataset_configs = EXT2CONFIG[selected_ext]
    raw_datasets = dataset_configs[0](
        data_files, 
        **dataset_configs[1]
    ).read()


    cache_dir = "cache"
    output_dir ="BB20K"
    model_name_or_path = "csebuetnlp/banglabert_large"
    tags_key = "tags"
    tokens_key = "tokens"
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
    )

    label_to_id = config.label2id if config.task_specific_params and config.task_specific_params.get("finetuned", False) else None
    if label_to_id is None:
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        label_list = get_label_list(raw_datasets["train"][tags_key])
        label_to_id = {v: i for i, v in enumerate(label_list)}
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
        config.task_specific_params = {"finetuned": True}
    else:
        label_list = list(label_to_id.keys())
        

    tokenizer_kwargs = {"add_prefix_space": True} if config.model_type in {"gpt2", "roberta"} else {}   
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_fast=True,
        **tokenizer_kwargs
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=cache_dir
    )



    pad_to_max_length = True
    max_seq_length = 64
    do_normalize = True
    overwrite_cache = False
    label_all_tokens = False
    

    if pad_to_max_length:
        padding = "max_length"
    else:
        padding = False


    
    if do_normalize:
        def normalize_example(example):
            for i, token in enumerate(example[tokens_key]):
                normalized_token = normalize(token)
                if len(normalized_token) > 0:
                    example[tokens_key][i] = normalized_token

            return example

        raw_datasets = raw_datasets.map(
            normalize_example,
            desc="Running normalization on dataset",
            load_from_cache_file=not overwrite_cache
        )

    def tokenize_and_align_labels(examples):

        tokenized_inputs = tokenizer(
            examples[tokens_key],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[tags_key]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(label_to_id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx


            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs


    raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    train_dataset = raw_datasets["train"]

    eval_dataset = raw_datasets["validation"]

    predict_dataset = raw_datasets["test"]


    class CustomTrainer(Trainer):
        
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            weight=torch.tensor ( [1.0, 1.0, 1.0, 1.0,2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 34.0] ).to("cuda")
            loss_fct = nn.CrossEntropyLoss(weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    


    fp16 = False
    steps = 500
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if fp16 else None)
    my_training_args = TrainingArguments(
        report_to="none",
        output_dir = output_dir,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2,
        weight_decay=0.1,
        lr_scheduler_type="linear",
        per_device_train_batch_size=32,
        save_steps = steps,
        eval_steps = steps,
        logging_steps = steps,
        save_strategy = "steps",
        evaluation_strategy="steps",
        num_train_epochs=4 ,
        fp16=fp16,

    )


    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=my_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    # last_checkpoint = get_last_checkpoint(output_dir)
    # checkpoint = last_checkpoint
    checkpoint = None
    if train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] =  len(train_dataset)

    trainer.save_model()

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    predictions, labels, metrics = trainer.predict(eval_dataset, metric_key_prefix="predict")
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    for prediction_list in true_predictions:
        for x in prediction_list:
            print(x)
        print()

    # with open('real.txt', 'w') as f:
    #     for i in raw_datasets["validation"]:

    #         for k in i['tags']:
    #             f.write(k+'\n')
    #         f.write('\n')


if __name__ == '__main__':
    main(Train=True)