import argparse
import pandas as pd
from datasets import load_dataset


EXPERIMENT_NAME = "counterfact_paraphrase_gemma3_4b_scope2_lasttoken"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument(
        "--output",
        default=f"data/counterfact/counterfact_paraphrase_prompts_{EXPERIMENT_NAME}.csv",
    )
    args = parser.parse_args()

    ds = load_dataset("azhx/counterfact")
    data = ds[args.split]

    if args.max_facts is not None:
        data = data.select(range(min(args.max_facts, len(data))))

    rows = []

    for ex in data:
        rr = ex["requested_rewrite"]

        case_id = ex["case_id"]
        subject = rr["subject"]
        relation_id = rr["relation_id"]
        template = rr["prompt"]

        base_prompt = template.format(subject)

        target_true = rr["target_true"]["str"]
        target_new = rr["target_new"]["str"]

        paraphrase_prompts = ex["paraphrase_prompts"]

        for j, prompt in enumerate(paraphrase_prompts):
            if prompt is None:
                continue

            prompt = str(prompt).strip()

            if not prompt:
                continue

            rows.append(
                {
                    "fact_id": f"cf_{case_id}",
                    "case_id": case_id,
                    "split": args.split,
                    "relation_id": relation_id,
                    "subject": subject,
                    "template": template,
                    "base_prompt": base_prompt,
                    "variant_id": f"paraphrase_{j:02d}",
                    "variant_source": "paraphrase_prompts",
                    "prompt": prompt,
                    "correct_answer": target_true,
                    "target_true": target_true,
                    "target_new": target_new,
                }
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No paraphrase prompts were extracted. Something is wrong.")

    df.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Split:", args.split)
    print("Facts:", df["fact_id"].nunique())
    print("Rows:", len(df))
    print("Average prompts per fact:", len(df) / df["fact_id"].nunique())
    print("Variant examples:", sorted(df["variant_id"].unique())[:10])
    print()
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
