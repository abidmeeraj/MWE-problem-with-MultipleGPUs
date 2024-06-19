import torch

def remove_special_tokens(tokenizer, text):
    word_pieces = []
    for token in text:
        if token not in (
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        ):
            word_pieces.append(token)
    return word_pieces


class SummarizationDataset:

    def __init__(
        self, ds, tokenizer, input_text, summary, max_input_len, max_summary_len
    ):
        super().__init__()
        self.max_input_len = max_input_len
        self.max_summary_len = max_summary_len

        self.ds = ds
        self.tokenizer = tokenizer
        self.input_text = input_text
        self.summary = summary

        self.sos_token = torch.tensor([tokenizer.cls_token_id], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.sep_token_id], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.pad_token_id], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        input_summary_pair = self.ds[idx]
        input_text = input_summary_pair[self.input_text]
        summary_text = input_summary_pair[self.summary]

        # Transform Text into tokens and remove special BERT tokens
        enc_input_tokens = remove_special_tokens(
            self.tokenizer, self.tokenizer(input_text)["input_ids"]
        )
        dec_input_tokens = remove_special_tokens(
            self.tokenizer, self.tokenizer(summary_text)["input_ids"]
        )

        # Need to take care of longer sequences
        if len(enc_input_tokens) > self.max_input_len - 2:
            enc_input_tokens = enc_input_tokens[: self.max_input_len - 2]
        if len(dec_input_tokens) > self.max_summary_len - 1:
            dec_input_tokens = dec_input_tokens[: self.max_summary_len - 1]

        # Add sos, eos tokens to each input
        enc_num_padding_tokens = (
            self.max_input_len - len(enc_input_tokens) - 2
        )  # Need to add sos and eos tokens
        # Add sos token to the decoder input only
        dec_num_padding_tokens = (
            self.max_summary_len - len(dec_input_tokens) - 1
        )  # Need to add sos token only

        if enc_num_padding_tokens < 0:
            raise ValueError("Input is too long.")
        if dec_num_padding_tokens < 0:
            raise ValueError("Summary text is too long.")

        # Add sos and eos tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )
        # Add only sos token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )
        # Add only eos token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ],
            dim=0,
        )

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(
            0
        ).int() & causal_mask(decoder_input.size(0)).unsqueeze(0)

        assert encoder_input.size(0) == self.max_input_len
        assert decoder_input.size(0) == self.max_summary_len
        assert label.size(0) == self.max_summary_len

        return {
            "encoder_input": encoder_input,  # (batch_size, input_len)
            "decoder_input": decoder_input,  # (batch_size, summary_len)
            "encoder_mask": encoder_mask,  # (batch_size, 1, 1, input_len)
            "decoder_mask": decoder_mask,  # (batch_size, 1, summary_len, summary_len)
            "label": label,  # (batch_size, seq_len)
            "input_text": input_text,
            "summary_text": summary_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1).type(torch.bool)
    return mask == 0
