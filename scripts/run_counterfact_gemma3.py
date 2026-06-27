import argparse
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


EXPERIMENT_NAME = "counterfact_paraphrase_gemma3_4b_scope2_lasttoken"


def normalize_text(x):
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" .,:;!?\"'()[]{}")
    return x


def strict_grade(generated_answer, correct_answer):
    """
    CounterFact prompts are completion prompts.
    Example:
    prompt: Angola belongs to the continent of
    expected generated answer: Africa

    We grade the beginning of the generated continuation.
    """
    gen = str(generated_answer).strip()
    first_line = gen.split("\n")[0].strip()

    # keep only a short answer area to avoid matching very late text
    short_segment = first_line[:120]

    norm_segment = normalize_text(short_segment)
    norm_answer = normalize_text(correct_answer)

    if not norm_answer:
        return False, ""

    is_correct = (
        norm_segment == norm_answer
        or norm_segment.startswith(norm_answer + " ")
        or norm_segment.startswith(norm_answer + ".")
        or norm_answer in norm_segment.split(" ")[:8]
    )

    return bool(is_correct), correct_answer if is_correct else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=f"data/counterfact/counterfact_paraphrase_prompts_{EXPERIMENT_NAME}.csv",
    )
    parser.add_argument(
        "--output",
        default=f"outputs/prompt_outputs_{EXPERIMENT_NAME}.csv",
    )
    parser.add_argument("--model-name", default="google/gemma-3-4b-pt")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    print("Input rows:", len(df))
    print("Model:", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    rows = []

    prompts = df["prompt"].astype(str).tolist()

    for start in tqdm(range(0, len(df), args.batch_size)):
        batch_df = df.iloc[start:start + args.batch_size]
        batch_prompts = prompts[start:start + args.batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[:, prompt_len:]
        generated_texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for (_, row), generated_answer in zip(batch_df.iterrows(), generated_texts):
            is_correct, matched_answer = strict_grade(
                generated_answer=generated_answer,
                correct_answer=row["correct_answer"],
            )

            out = row.to_dict()
            out["model_name"] = args.model_name
            out["generated_answer"] = generated_answer
            out["strict_answer_segment"] = str(generated_answer).split("\n")[0].strip()[:120]
            out["strict_matched_answer"] = matched_answer
            out["is_correct"] = is_correct
            rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    print()
    print("Saved:", args.output)
    print("Rows:", len(out_df))
    print("Correct:", int(out_df["is_correct"].sum()))
    print("Wrong:", int((~out_df["is_correct"]).sum()))
    print("Facts:", out_df["fact_id"].nunique())


if __name__ == "__main__":
    main()
