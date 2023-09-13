import os
from time import time

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torchmetrics.classification import BinaryAccuracy
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

accelerator = Accelerator()
DEVICE = accelerator.device

accelerator.print(accelerator.distributed_type)

torch.manual_seed(53)
torch.cuda.manual_seed(53)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/gpfsdswork/dataset/HuggingFace/imdb/plain_text",
    )
    parser.add_argument(
        "--model_dir", type=str, default="/gpfsdswork/dataset/HuggingFace_Models/"
    )
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def train_loop(model, tokenizer, train_dataloader, criterion, optimizer):
    model.train()
    loop = tqdm(train_dataloader)

    for data in loop:
        inputs = tokenizer.batch_encode_plus(
            data["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        labels = data["label"].to(DEVICE)

        out = model(**inputs)
        loss = criterion(out.logits, labels)

        accelerator.backward(loss)
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return model


def eval_loop(model, tokenizer, test_dataloader):
    metric = BinaryAccuracy().to(DEVICE)
    model.eval()
    loop = tqdm(test_dataloader)
    with torch.no_grad():
        for data in loop:
            inputs = tokenizer.batch_encode_plus(
                data["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)
            labels = data["label"].to(DEVICE)

            out = model(**inputs)
            score = metric(out.logits.argmax(dim=1), labels)

            loop.set_postfix(
                weighted_average_score=metric.compute().item(), score=score.item()
            )

    return metric.compute()


def main(args):
    # Initialize Datasets
    dataset = load_from_disk(args.data_path)

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Initialize Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(args.model_dir, args.model_name)
    ).to(DEVICE)

    # Initialize Optimizer and Criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-04)

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    model = train_loop(model, tokenizer, train_dataloader, criterion, optimizer)
    accuracy = eval_loop(model, tokenizer, test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
