import os

import torch
from torch.utils.data import Dataset

from transformers.tokenization_utils import trim_batch

MODEL_NAMES_TO_PATH = {
    'bart_large': 'facebook/bart-large',
    'bart_large_cnn': 'facebook/bart-large-cnn',
    'bart_tiny': 'sshleifer/bart-tiny-random',
    't5': 't5-large',
    't5_tiny_random': 'patrickvonplaten/t5-tiny-random',
}


def encode_file(tokenizer, data_path, max_length, max_examples=None, pad_to_max_length=True, return_tensors='pt'):
    examples = []
    with open(data_path, "r") as f:
        for text in f.readlines():
            tokenized = tokenizer.batch_encode_plus(
                [text], max_length=max_length, pad_to_max_length=pad_to_max_length, return_tensors=return_tensors,
            )
            examples.append(tokenized)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


class ModelInfo:
    def __init__(self, model_name, experiment):
        assert model_name in MODEL_NAMES_TO_PATH
        self.model_path = MODEL_NAMES_TO_PATH[model_name]
        self.experiment = experiment
        if 'bart' in model_name:
            self.model_type = 'bart'
        elif 't5' in model_name:
            self.model_type = 't5'
        else:
            raise Exception('Unrecognized model={}'.format(model_name))

    def result_path(self):
        return os.path.join('results', self.model_type, self.experiment)


class SummarizationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset='cnn_dm',
        type_path='train',
        max_examples=None,
        max_source_length=1024,
        max_target_length=56,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        source_fn = os.path.join('data', dataset, type_path + ".source")
        target_fn = os.path.join('data', dataset, type_path + ".target")
        self.source = encode_file(tokenizer, source_fn, max_source_length, max_examples)
        self.target = encode_file(tokenizer, target_fn, max_target_length, max_examples)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y}
