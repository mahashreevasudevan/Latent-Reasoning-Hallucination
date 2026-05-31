import os
import time
import json
from datetime import datetime

import pandas as pd
from cerebras.cloud.sdk import Cerebras

# Config

API_KEY    = "CEREBRAS_API_KEY"   
MODEL      = "qwen-3-235b-a22b-instruct-2507"
DELAY      = 2.0                       # seconds between calls
CHECKPOINT = "qwen3_235b_checkpoint.json"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT     = f"results_qwen3_235b_{TIMESTAMP}.csv"

COT_PROMPT = (
    "Please solve the following problem step by step, "
    "showing your full reasoning before giving your final answer."
)

# Initialise client 

client = Cerebras(api_key=API_KEY)

# Load datasets

print("Loading datasets...")

gsm8k = pd.read_csv("gsm8k_sample.csv")
aqua  = pd.read_csv("aqua_sample.csv")

# Pull 45 unique questions from the custom hard dataset
# the file uses 'question_type' not 'difficulty' — renamed here for consistency
custom_full = (
    pd.read_csv("results_hard_20260309_201712.csv")
    .drop_duplicates(subset="question_id")
    .head(45)
    [["question_id", "question", "correct_answer", "question_type"]]
    .rename(columns={"question_type": "difficulty"})
    .copy()
)

print(f"  GSM8K:    {len(gsm8k)} questions")
print(f"  AQuA-RAT: {len(aqua)} questions")
print(f"  Custom:   {len(custom_full)} questions")

# Build task list 
tasks = []

for _, row in gsm8k.iterrows():
    tasks.append({
        "dataset":        "GSM8K",
        "question_id":    int(row["question_id"]),
        "difficulty":     row["difficulty"],
        "step_count":     row.get("step_count", ""),
        "question":       str(row["question"]),
        "correct_answer": str(row["answer"]),          # GSM8K uses "answer" column
    })

for _, row in aqua.iterrows():
    options = str(row.get("options", ""))
    question_text = str(row["question"])
    if options and options != "nan":
        question_text = question_text + "\n\nOptions: " + options
    tasks.append({
        "dataset":        "AQuA-RAT",
        "question_id":    int(row["question_id"]),
        "difficulty":     row["difficulty"],
        "step_count":     row.get("step_count", ""),
        "question":       question_text,
        "correct_answer": str(row["correct"]),          # AQuA-RAT uses "correct" column
    })

for _, row in custom_full.iterrows():
    tasks.append({
        "dataset":        "Custom",
        "question_id":    int(row["question_id"]),
        "difficulty":     str(row["difficulty"]),
        "step_count":     "",
        "question":       str(row["question"]),
        "correct_answer": str(row["correct_answer"]),
    })

print(f"\nTotal tasks: {len(tasks)}")

# Load checkpoint 
completed = set()
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        completed = set(json.load(f))
    print(f"Resuming -- {len(completed)} already collected, {len(tasks) - len(completed)} remaining")
else:
    print(f"Fresh run -- {len(tasks)} responses to collect")

# Collection loop 
results = []
consecutive_errors = 0

for i, task in enumerate(tasks):
    task_key = f"{task['dataset']}_{task['question_id']}"

    if task_key in completed:
        continue

    prompt = f"{COT_PROMPT}\n\nQuestion: {task['question']}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        answer             = response.choices[0].message.content
        success            = True
        consecutive_errors = 0

    except Exception as e:
        print(f"  ERROR on {task_key}: {e}")
        answer             = None
        success            = False
        consecutive_errors += 1

        if consecutive_errors >= 5:
            print("\n5 consecutive errors -- stopping early.")
            print("Re-run the script to resume from checkpoint.")
            break

    results.append({
        "model":                MODEL,
        "dataset":              task["dataset"],
        "question_id":          task["question_id"],
        "difficulty":           task["difficulty"],
        "step_count":           task["step_count"],
        "question":             task["question"],
        "correct_answer":       task["correct_answer"],
        "model_response":       answer,
        "success":              success,
        "final_answer_correct": "",
        "reasoning_quality":    "",
        "flaw_category":        "",
        "flaw_notes":           "",
    })

    completed.add(task_key)
    with open(CHECKPOINT, "w") as f:
        json.dump(list(completed), f)

    remaining = len(tasks) - len(completed)
    status    = "OK" if success else "FAILED"
    print(f"[{i+1:3}/{len(tasks)}] {task_key:<25} {status}  |  {remaining} remaining")

    time.sleep(DELAY)

# Save results 
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")
    print(f"  Rows:       {len(df)}")
    print(f"  Successful: {df['success'].sum()}")
    print(f"  Failed:     {(~df['success']).sum()}")
    if df['success'].sum() == len(df):
        print("\nAll responses collected. Upload the CSV here for labelling.")
else:
    print("\nNo new results collected -- everything was already in the checkpoint.")
