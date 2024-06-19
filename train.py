from dataset import SummarizationDataset
from model import build_transformer

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import csv
import warnings
from tqdm import tqdm

from pathlib import Path

from transformers import AutoTokenizer

# Defining global variables
seq_len = 512
max_summary_len = 100
summary = "Summary"
body = "Body"
batch = 8
d_model = 768
num_blocks = 2
num_heads = 4
num_epochs = 2
learning_rate = 0.00001

def read_csv_to_dict_list(filename, body_column="Body", summary_column="Summary"):
  """
  This function reads a CSV file and returns a list of dictionaries containing data from two specified columns.

  Args:
      filename: The path to the CSV file.
      body_column: The name of the column containing the body text (default: "Body").
      summary_column: The name of the column containing the summary text (default: "Summary").

  Returns:
      A list of dictionaries, where each dictionary represents a row in the CSV file with keys "Body" and "Summary".
  """
  data_list = []
  with open(filename, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      data_dict = {body_column: row[body_column], summary_column: row[summary_column]}
      data_list.append(data_dict)
  return data_list

def get_or_build_tokenizer():

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_ds():
    csv_name = "train_reduced.csv"
    raw_train = read_csv_to_dict_list(csv_name)

    # Build tokenizer
    tokenizer = get_or_build_tokenizer()


    train_ds = SummarizationDataset(
        raw_train,
        tokenizer,
        body,
        summary,
        seq_len,
        max_summary_len,
    )

    
    train_dataloader = DataLoader(
        train_ds, batch_size=batch, shuffle=False
    )


    return train_dataloader, tokenizer


def get_model(vocab_size):

    model = build_transformer(
        vocab_size,
        seq_len,
        max_summary_len,
        d_model=d_model,
        N=num_blocks,
        h=num_heads,
    )
    return model


def train_model():
    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)

    device = torch.device(device)

    train_dataloader, tokenizer = get_ds()
    model = get_model(tokenizer.vocab_size)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # Tensorboard
    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 1

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, num_epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)  # (b, input_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, summary_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (B, 1, 1, input_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (B, 1, summary_len, summary_len)

            # Run the tensors through the encoder, decoder and the projection layer
            if torch.cuda.device_count() > 1:
                encoder_output = model.module.encode(
                    encoder_input, encoder_mask
                )  # (B, input_len, d_model)
                decoder_output = model.module.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # (B, seq_len, d_model)
                proj_output = model.module.project(
                    decoder_output
                )  # (B, seq_len, vocab_size)
            else:
                encoder_output = model.encode(
                    encoder_input, encoder_mask
                )  # (B, input_len, d_model)
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # (B, seq_len, d_model)
                proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch["label"].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer.vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item())
            writer.flush()


            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_model()
