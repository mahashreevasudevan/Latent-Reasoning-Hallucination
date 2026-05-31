import csv
import random
import re
from datasets import load_dataset

random.seed(42)


def count_steps_gsm8k(solution: str) -> int:
    """Estimate reasoning steps by counting calculation lines in GSM8K solution."""
    lines = [l.strip() for l in solution.split('\n') if l.strip()]
    # GSM8K solutions use '<<...>>' for calculations
    calc_lines = [l for l in lines if '<<' in l or any(op in l for op in ['=', '+', '-', '*', '/'])]
    return max(len(calc_lines), len(lines))

def classify_difficulty_gsm8k(solution: str) -> str:
    steps = count_steps_gsm8k(solution)
    if steps <= 3:
        return 'easy'
    elif steps <= 6:
        return 'medium'
    else:
        return 'hard'

def count_steps_aqua(rationale: str) -> int:
    """Estimate reasoning steps by sentence/line count in AQuA rationale."""
    sentences = re.split(r'[.!?]\s+', rationale.strip())
    return len([s for s in sentences if len(s.strip()) > 10])

def classify_difficulty_aqua(rationale: str) -> str:
    steps = count_steps_aqua(rationale)
    if steps <= 2:
        return 'easy'
    elif steps <= 4:
        return 'medium'
    else:
        return 'hard'

def extract_gsm8k_answer(solution: str) -> str:
    """Extract final numeric answer from GSM8K solution."""
    match = re.search(r'####\s*(.+)', solution)
    return match.group(1).strip() if match else solution.split('\n')[-1].strip()

def format_aqua_options(options: list) -> str:
    """Format AQuA options list into readable string."""
    return ' | '.join(options)

# Sampling GSM8K

print("Loading GSM8K...")
gsm = load_dataset('gsm8k', 'main', split='test')
print(f"  Total: {len(gsm)} questions")

# Classifying
gsm_classified = []
for i, item in enumerate(gsm):
    diff = classify_difficulty_gsm8k(item['answer'])
    steps = count_steps_gsm8k(item['answer'])
    gsm_classified.append({
        'source_id':      i,
        'difficulty':     diff,
        'step_count':     steps,
        'question':       item['question'],
        'answer':         extract_gsm8k_answer(item['answer']),
        'full_solution':  item['answer'],
    })

# Sampling 10 per difficulty band
gsm_easy   = [x for x in gsm_classified if x['difficulty'] == 'easy']
gsm_medium = [x for x in gsm_classified if x['difficulty'] == 'medium']
gsm_hard   = [x for x in gsm_classified if x['difficulty'] == 'hard']

print(f"  Easy: {len(gsm_easy)}, Medium: {len(gsm_medium)}, Hard: {len(gsm_hard)}")

# Sorting medium/hard by step count for variety
gsm_medium_sorted = sorted(gsm_medium, key=lambda x: x['step_count'])
gsm_hard_sorted   = sorted(gsm_hard,   key=lambda x: x['step_count'], reverse=True)

selected_gsm = (
    random.sample(gsm_easy,   min(10, len(gsm_easy)))   +
    random.sample(gsm_medium, min(10, len(gsm_medium))) +
    random.sample(gsm_hard,   min(10, len(gsm_hard)))
)

# Assigning question IDs
for i, q in enumerate(selected_gsm, 1):
    q['question_id']   = i
    q['dataset']       = 'GSM8K'
    q['answer_format'] = 'numeric'

print(f"  Selected: {len(selected_gsm)} questions")

# Save
gsm_fields = ['dataset', 'question_id', 'source_id', 'difficulty', 'step_count',
              'question', 'answer', 'answer_format', 'full_solution']

with open('gsm8k_sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=gsm_fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(selected_gsm)

print("  Saved: gsm8k_sample.csv")

# Sampling AQuA-RAT 
print("\nLoading AQuA-RAT...")
aqua = load_dataset('aqua_rat', 'raw', split='test')
print(f"  Total: {len(aqua)} questions")

# Classifying
aqua_classified = []
for i, item in enumerate(aqua):
    rationale = item.get('rationale', '')
    diff  = classify_difficulty_aqua(rationale)
    steps = count_steps_aqua(rationale)
    aqua_classified.append({
        'source_id':      i,
        'difficulty':     diff,
        'step_count':     steps,
        'question':       item['question'],
        'options':        format_aqua_options(item['options']),
        'correct':        item['correct'],
        'rationale':      rationale,
        'answer_format':  'multiple_choice',
    })

# Sampling 10 per difficulty band
aqua_easy   = [x for x in aqua_classified if x['difficulty'] == 'easy']
aqua_medium = [x for x in aqua_classified if x['difficulty'] == 'medium']
aqua_hard   = [x for x in aqua_classified if x['difficulty'] == 'hard']

print(f"  Easy: {len(aqua_easy)}, Medium: {len(aqua_medium)}, Hard: {len(aqua_hard)}")

selected_aqua = (
    random.sample(aqua_easy,   min(10, len(aqua_easy)))   +
    random.sample(aqua_medium, min(10, len(aqua_medium))) +
    random.sample(aqua_hard,   min(10, len(aqua_hard)))
)

# Assign question IDs
for i, q in enumerate(selected_aqua, 1):
    q['question_id'] = i
    q['dataset']     = 'AQuA-RAT'

print(f"  Selected: {len(selected_aqua)} questions")

# Save
aqua_fields = ['dataset', 'question_id', 'source_id', 'difficulty', 'step_count',
               'question', 'options', 'correct', 'rationale', 'answer_format']

with open('aqua_sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=aqua_fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(selected_aqua)

print("  Saved: aqua_sample.csv")

# Summary 

print(f"""
{'='*60}
SAMPLING COMPLETE

GSM8K:    {len(selected_gsm)} questions saved to gsm8k_sample.csv
AQuA-RAT: {len(selected_aqua)} questions saved to aqua_sample.csv

Difficulty breakdown:
  GSM8K   — easy: 10, medium: 10, hard: 10
  AQuA-RAT — easy: 10, medium: 10, hard: 10

Next step:
  Upload both CSVs here and we will review the selection
  before running the 6 models.
{'='*60}
""")
