import torch


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_seq_len):
        self.labels = df.label.tolist()
        self.encodings = tokenizer(
            df.text.tolist(),
            max_length=ceil(max_seq_len / 8) * 8,
            truncation=True,
            padding="max_length",
            pad_to_multiple_of=8,
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_seq_length, pad_token, unk_token):
        self.labels = df.label.tolist()
        self.word2idx = {term: idx for idx, term in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.pad_token, self.unk_token = pad_token, unk_token
        self.input_ids = []
        self.sequence_lens = []
        self.labels = []
        for i in range(df.shape[0]):
            input_ids, sequence_len = self.convert_text_to_input_ids(
                df.iloc[i].text, pad_to_len=max_seq_length
            )

            self.input_ids.append(input_ids.reshape(-1))
            self.sequence_lens.append(sequence_len)
            self.labels.append(df.iloc[i].label)

        assert len(self.input_ids) == df.shape[0]
        assert len(self.sequence_lens) == df.shape[0]
        assert len(self.labels) == df.shape[0]

    def convert_text_to_input_ids(self, text, pad_to_len):
        words = text.strip().split()[:pad_to_len]
        deficit = pad_to_len - len(words)
        words.extend([self.pad_token] * deficit)
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                words[i] = self.word2idx[self.unk_token]
            else:
                words[i] = self.word2idx[words[i]]
        return torch.Tensor(words).long(), pad_to_len - deficit

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ret_dict = dict()
        ret_dict["input_ids"] = self.input_ids[idx].reshape(-1)
        ret_dict["sequence_len"] = torch.tensor(self.sequence_lens[idx]).long()
        ret_dict["labels"] = torch.tensor(self.labels[idx])
        return ret_dict
