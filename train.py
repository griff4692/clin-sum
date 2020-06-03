from pytorch_lightning import LightningModule, Trainer
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.base import load_dataset


class SummarizePipeline(LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.e = nn.Embedding(vocab_size, embedding_dim=100, padding_idx=0)
        self.encoder = nn.LSTM(batch_first=True, input_size=100, hidden_size=100, bidirectional=False)
        self.decoder = nn.LSTM(batch_first=True, input_size=100, hidden_size=100, bidirectional=False)

        self.dummy_regressor = nn.Linear(100, 1)
        self.dummy_mse = nn.MSELoss()

    def forward(self, enc_input_tens, enc_target_tens, dec_input_tens, dec_target_tens):
        """
        All tensors are 0-padded to longest value in batch across each dimension
        :param enc_input_tens: batch_size x max_num_docs x max_num_words_per_doc
        :param enc_target_tens: batch_size x max_num_docs x max_num_words_per_doc
        :param dec_input_tens: batch_size x max_num_words_per_summary
        :param dec_target_tens: batch_size x max_num_words_per_summary
        :return:
        """
        batch_size, max_num_docs, max_num_words_per_doc = enc_input_tens.size()
        _, max_num_words_per_summary = dec_input_tens.size()

        enc_embeds = self.e(enc_input_tens)  # batch_size x max_num_docs x max_num_words_per_doc x embed_dim
        enc_embeds_flat_doc = enc_embeds.view(batch_size * max_num_docs, max_num_words_per_doc, -1)
        dec_embeds = self.e(dec_input_tens)  # batch_size x max_num_words_per_summary x embed_dim

        # take final hidden state
        _, (enc_flat_h, _) = self.encoder(enc_embeds_flat_doc)
        _, (dec_h, _) = self.decoder(dec_embeds)

        # mean pool over document last hidden states
        enc_h = enc_flat_h.squeeze(0).view(batch_size, max_num_docs, -1).mean(1)
        dec_h = dec_h.squeeze(0)
        return self.dummy_regressor(enc_h).squeeze(-1), self.dummy_regressor(dec_h).squeeze(-1)

    def training_step(self, batch, batch_idx):
        y_enc, y_dec = self(*batch)
        loss = self.dummy_mse(y_enc, y_dec)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

    # def validation_step(self):
    #     pass


def ids_to_tensor(ids):
    """
    Converts a nested list of integers to a padded tensor.
    """
    if isinstance(ids, torch.Tensor):
        return ids
    if isinstance(ids, list):
        if isinstance(ids[0], int):
            return torch.LongTensor(ids)
        if isinstance(ids[0], torch.Tensor):
            return pad_tensors(ids)
        if isinstance(ids[0], list):
            return ids_to_tensor([ids_to_tensor(inti) for inti in ids])


def pad_tensors(tensors):
    """
    Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.

    The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
    where `Si` is the maximum value of dimension `i` amongst all tensors.
    """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim)
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        else:
            raise ValueError('Padding is supported for upto 3D tensors at max.')
    return padded_tensor


def collate_fn(batch):
    """
    :param batch: list of Example objects
    :return: model tensors (encoder input, decoder input, decoder target)
    encoder input -> batch_size x max_num_docs x max_num_toks
    decoder input -> batch_size x max_num_toks
    decoder target -> batch_size x max_num_toks
    """
    # example.enc_input_ids --> num_docs x num_toks
    # example.enc_target_ids --> num_docs x num_toks (with temp. ids for OOV)
    # example.dec_input_ids = num_toks
    # example.dec_target_ids = num_toks
    # example.doc_oov TBD

    enc_input_tens = ids_to_tensor(list(map(lambda ex: ex.enc_input_ids, batch)))
    enc_target_tens = ids_to_tensor(list(map(lambda ex: ex.enc_target_ids, batch)))
    dec_input_tens = ids_to_tensor(list(map(lambda ex: ex.dec_input_ids, batch)))
    dec_target_tens = ids_to_tensor(list(map(lambda ex: ex.dec_target_ids, batch)))
    return enc_input_tens, enc_target_tens, dec_input_tens, dec_target_tens


if __name__ == '__main__':
    dataset = load_dataset(name='multi_news')
    train_dataloader = DataLoader(dataset['train'], batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset['validation'], batch_size=32, num_workers=4, shuffle=False, collate_fn=collate_fn)
    V = len(dataset['train'].vocab.stoi)

    trainer = Trainer(gpus=0, num_nodes=1, accumulate_grad_batches=1)
    model = SummarizePipeline(vocab_size=V)
    trainer.fit(model, train_dataloader=train_dataloader)  # , val_dataloaders=val_dataloader)
