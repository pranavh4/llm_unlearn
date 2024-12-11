import argparse
import logging
import os
from collections import defaultdict

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_responses(model, tokenizer, prompt, device, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(prompt) + max_length,
            num_return_sequences=3,
            do_sample=True,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_path or args.model_name)
    model.to(device)
    model.eval()

    dataset = load_dataset(
        "truthful_qa", "generation", split=f"validation[:{args.num_samples}]"
    )
    prompts = dataset["question"]  # type: ignore

    results = defaultdict(lambda: [])

    for i, prompt in enumerate(prompts):
        responses = generate_responses(model, tokenizer, prompt, device)
        for response in responses:
            results["prompt_id"].append(i)
            results["prompt"].append(prompt)
            results["response"].append(response[len(prompt) :])
        logging.info(f"{i + 1}/{args.num_samples}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    pd.DataFrame(results).to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to take from the dataset",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="scores",
        help="The path to the output file",
        required=True,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Saved model path (Optional)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d:%H-%M-%S",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    main(args)
