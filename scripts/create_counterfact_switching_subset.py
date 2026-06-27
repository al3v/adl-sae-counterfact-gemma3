import argparse
import pandas as pd


EXPERIMENT_NAME = "counterfact_paraphrase_gemma3_4b_scope2_lasttoken"


def to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).lower() in ["true", "1", "yes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=f"outputs/prompt_outputs_{EXPERIMENT_NAME}.csv",
    )
    parser.add_argument(
        "--output",
        default=f"outputs/switching_facts_{EXPERIMENT_NAME}.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["is_correct"] = df["is_correct"].apply(to_bool)

    switching_fact_ids = []

    for fact_id, group in df.groupby("fact_id"):
        values = set(group["is_correct"].tolist())

        if values == {True, False}:
            switching_fact_ids.append(fact_id)

    switching_df = df[df["fact_id"].isin(switching_fact_ids)].copy()
    switching_df.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Total rows:", len(df))
    print("Total facts:", df["fact_id"].nunique())
    print("Correct rows:", int(df["is_correct"].sum()))
    print("Wrong rows:", int((~df["is_correct"]).sum()))
    print()
    print("Switching facts:", switching_df["fact_id"].nunique())
    print("Switching rows:", len(switching_df))
    print("Correct switching rows:", int(switching_df["is_correct"].sum()))
    print("Wrong switching rows:", int((~switching_df["is_correct"]).sum()))
    print()

    if len(switching_df) > 0:
        print(switching_df[[
            "fact_id",
            "variant_id",
            "subject",
            "prompt",
            "correct_answer",
            "target_new",
            "generated_answer",
            "is_correct",
        ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
