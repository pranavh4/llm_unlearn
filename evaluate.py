import argparse
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd

ROUGES = ["rouge1", "rouge2", "rouge4", "rougeL"]
ROUGE_SCORER = rouge_scorer.RougeScorer(ROUGES, use_stemmer=True)
SMOOTHING_FUNCTION = SmoothingFunction().method1

def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors='pt').to(device)
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=SMOOTHING_FUNCTION)

def generate_text(model, tokenizer, prompt, device, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=len(prompt) + max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_pku_dataset():
  def get_unsafe_responses(rows):
    prompts = []
    responses = []
    for row in zip(rows["prompt"], rows["response_0"], rows["response_1"], rows["is_response_0_safe"], rows["is_response_1_safe"]):
      if not row[3]:
        prompts.append(row[0])
        responses.append(row[1])
      if not row[4]:
        prompts.append(row[0])
        responses.append(row[2])
    return {"prompt": prompts, "response": responses}

  dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
  unsafe_responses = dataset.map(get_unsafe_responses, batched=True, remove_columns=dataset['train'].column_names)
  return unsafe_responses.shuffle(seed=42)

def calculate_scores(model, tokenizer, device, prompt, response, results):
  perplexity = calculate_perplexity(model, tokenizer, prompt + ' ' + response, device)
  generated_response = generate_text(model, tokenizer, prompt, device)[len(prompt):]
  bleu_score = calculate_bleu(response, generated_response)
  rouge_score = ROUGE_SCORER.score(response, generated_response)
  results['prompt'].append(prompt)
  results['response'].append(response)
  results['generated_response'].append(generated_response)
  results['perplexity'].append(perplexity)
  results['bleu'].append(bleu_score)

  for score_type in rouge_score.keys():
    results[score_type + "_precision"].append(rouge_score[score_type].precision)
    results[score_type + "_recall"].append(rouge_score[score_type].recall)
    results[score_type + "_fmeasure"].append(rouge_score[score_type].fmeasure)

def main(args):
  accelerator = Accelerator()
  device = accelerator.device
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModelForCausalLM.from_pretrained(args.model_path or args.model_name)
  model.to(device)
  model.eval()

  dataset = load_pku_dataset()

  #Get scores for train samples
  sample = dataset['train'][:args.num_samples]
  results = defaultdict(lambda: [])

  for prompt, response in zip(sample['prompt'], sample['response']):
    calculate_scores(model, tokenizer, device, prompt, response, results)
    print(f'{len(results["prompt"])}/{args.num_samples}')

  output_dir = Path(args.output_directory)
  output_dir.mkdir(parents=True, exist_ok=True)

  pd.DataFrame(results).to_csv(output_dir / (args.output_file_prefix + '_train.csv'), index=False)

  #Get scores for test samples
  sample = dataset['test'][:args.num_samples]
  results = defaultdict(lambda: [])

  for prompt, response in zip(sample['prompt'], sample['response']):
    calculate_scores(model, tokenizer, device, prompt, response, results)
    print(f'{len(results["prompt"])}/{args.num_samples}')

  pd.DataFrame(results).to_csv(output_dir / (args.output_file_prefix + '_test.csv'), index=False)


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
      "--output_directory",
      type=str,
      default="results",
      help="Output Directory",
  )

  parser.add_argument(
    "--output_file_prefix",
    type=str,
    default="scores",
    help="Prefix for the output file",
  )

  parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Saved model path (Optional)",
  )

  args = parser.parse_args()
  main(args)
