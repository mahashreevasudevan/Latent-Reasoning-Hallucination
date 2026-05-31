import csv
import json
import os
import time
import glob
from datetime import datetime
from groq import Groq

# Config 

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
MODEL         = "openai/gpt-oss-120b"
MODEL_FAMILY  = "OpenAI OSS"
MAX_TOKENS    = 2048
TEMPERATURE   = 0.6
RETRY_LIMIT   = 3
RETRY_DELAY   = 8.0
RATE_LIMIT    = 3.0

TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE   = f"results_gptoss_{TIMESTAMP}.csv"
CHECKPOINT    = "gptoss_checkpoint.json"

# Questions 

QUESTIONS = [
    {"id": 1, "type": "probability", "question": "A test for a disease is 99% accurate. The disease affects 1 in 10000 people. You test positive. What is the probability you actually have the disease?", "answer": "Approximately 0.98% (about 1%). Despite 99% accuracy, the low base rate means most positives are false positives."},
    {"id": 2, "type": "probability", "question": "In the Monty Hall problem, you pick door 1. The host opens door 3 (no prize). Should you switch to door 2, stay with door 1, or does it not matter?", "answer": "Switch. Switching wins 2/3 of the time. Staying wins only 1/3."},
    {"id": 3, "type": "probability", "question": "A couple has two children. You know at least one is a boy. What is the probability both are boys?", "answer": "1/3. The sample space is BB, BG, GB - only one of three is BB."},
    {"id": 4, "type": "probability", "question": "A bag has 3 red and 2 blue balls. You draw one, note it is red, and put it back. You draw again. What is the probability both draws were red?", "answer": "9/25. Each draw is independent with P(red)=3/5, so (3/5)*(3/5)=9/25."},
    {"id": 5, "type": "probability", "question": "You flip a fair coin 10 times and get heads every time. What is the probability the next flip is heads?", "answer": "1/2. Each flip is independent. Past results do not affect future flips."},
    {"id": 6, "type": "probability", "question": "In a room of 23 people, what is the approximate probability that two people share a birthday?", "answer": "Approximately 50%. This is the birthday paradox - the answer surprises most people who guess much lower."},
    {"id": 7, "type": "probability", "question": "A family has 4 children. Given that at least 3 are girls, what is the probability all 4 are girls?", "answer": "1/5. Cases with at least 3 girls: GGGB, GBGG, BGGGG, GGBG, GGGG = 5 cases. Only GGGG satisfies all 4 girls."},
    {"id": 8, "type": "probability", "question": "You roll two dice. Given that the sum is at least 10, what is the probability the sum is exactly 12?", "answer": "1/6. Sums of 10 or more: (4,6),(5,5),(6,4),(5,6),(6,5),(6,6) = 6 outcomes. Sum of 12: only (6,6) = 1 outcome. So 1/6."},
    {"id": 9, "type": "probability", "question": "A drawer has 10 socks: 6 black and 4 white. You pick socks one at a time without looking. What is the minimum number you must pick to guarantee a matching pair?", "answer": "3. By the pigeonhole principle, with 3 picks you must have at least 2 of the same colour."},
    {"id": 10, "type": "probability", "question": "Simpson's Paradox: Hospital A has a higher survival rate than Hospital B for both minor and major operations separately. Can Hospital B still have a higher overall survival rate?", "answer": "Yes. If Hospital B treats proportionally more minor cases (which have high survival) the overall rate can be higher despite worse rates in each category separately."},
    {"id": 11, "type": "algebra", "question": "A snail climbs 3 metres up a 10 metre wall each day but slides back 2 metres each night. How many days does it take to reach the top?", "answer": "8 days. After 7 days it is at 7m. On day 8 it climbs to 10m and reaches the top before sliding back."},
    {"id": 12, "type": "algebra", "question": "If you have a 3-litre jug and a 5-litre jug, how do you measure exactly 4 litres of water?", "answer": "Fill 5L jug. Pour into 3L jug until full (leaves 2L in 5L jug). Empty 3L jug. Pour 2L into 3L jug. Fill 5L jug again. Pour from 5L into 3L jug until full (needs 1L). 5L jug now has exactly 4L."},
    {"id": 13, "type": "algebra", "question": "A lily pad doubles in size every day. It takes 30 days to cover the whole lake. How long does it take to cover half the lake?", "answer": "29 days. On day 30 it doubles from half to full, so on day 29 it covers half."},
    {"id": 14, "type": "algebra", "question": "You have 12 balls, one is heavier. Using a balance scale only 3 times, how do you find the heavy ball?", "answer": "Divide into groups of 4. Weigh group 1 vs group 2. If equal, heavy ball is in group 3. If not, it is in the heavier group. Then divide that group of 4 into 2+2, weigh them. Then weigh the 2 remaining one against one."},
    {"id": 15, "type": "algebra", "question": "A merchant doubles his money every day but spends 30 coins each day. He starts with 1 coin. After how many days does he first have more than 100 coins?", "answer": "On day 1: 2-30 = negative, so this fails from the start. The merchant goes bankrupt immediately. There is no day he exceeds 100."},
    {"id": 16, "type": "algebra", "question": "Solve: x + y = 10, xy = 40. What are x and y?", "answer": "No real solutions. The discriminant of x^2 - 10x + 40 = 0 is 100-160 = -60, which is negative."},
    {"id": 17, "type": "algebra", "question": "A train leaves City A at 60 mph. Another train leaves City B (300 miles away) at 90 mph. They travel toward each other. How far from City A do they meet?", "answer": "120 miles from City A. They meet after 300/(60+90) = 2 hours. Train A travels 60*2 = 120 miles."},
    {"id": 18, "type": "algebra", "question": "If 5 machines make 5 widgets in 5 minutes, how long does it take 100 machines to make 100 widgets?", "answer": "5 minutes. Each machine makes 1 widget in 5 minutes regardless of how many machines there are."},
    {"id": 19, "type": "algebra", "question": "A man sells a horse for 70, buys it back for 80, sells it again for 90. What is his net profit or loss?", "answer": "Profit of 20. He sold for 70 (-70 cost), bought for 80 (-80), sold for 90 (+90). Net: +70-80+90 = +20."},
    {"id": 20, "type": "algebra", "question": "You invest 1000 at 10% annual interest. After 2 years with compound interest, how much do you have?", "answer": "1210. After year 1: 1100. After year 2: 1100 * 1.1 = 1210."},
    {"id": 36, "type": "causal", "question": "Countries with more hospitals have higher death rates. Does building more hospitals cause more deaths?", "answer": "No. This is a confounding variable problem. Sicker populations need more hospitals AND have higher death rates. Hospitals do not cause deaths."},
    {"id": 37, "type": "causal", "question": "Students who take private tutoring score lower on average than those who do not. Does tutoring hurt performance?", "answer": "No. Selection bias: students who struggle are more likely to seek tutoring. The tutoring may actually be helping them perform better than they otherwise would."},
    {"id": 38, "type": "causal", "question": "Ice cream sales and drowning rates both rise in summer. Does ice cream cause drowning?", "answer": "No. A confounding variable (hot weather) causes both. People eat more ice cream in hot weather and swim more in hot weather."},
    {"id": 39, "type": "causal", "question": "A study shows people who carry lighters are more likely to get lung cancer. Do lighters cause cancer?", "answer": "No. Smokers carry lighters and smoking causes cancer. The lighter is a proxy for smoking behaviour, not a cause."},
    {"id": 40, "type": "causal", "question": "After a new traffic light was installed, accidents at that junction increased. Did the traffic light cause more accidents?", "answer": "Not necessarily. Regression to the mean: traffic lights are typically installed at dangerous junctions after a spike in accidents. The accidents may have been declining anyway."},
    {"id": 41, "type": "combinatorics", "question": "How many ways can 5 people sit in a row?", "answer": "120. This is 5! = 5*4*3*2*1 = 120."},
    {"id": 42, "type": "combinatorics", "question": "How many ways can you choose 3 people from a group of 10 to form a committee?", "answer": "120. This is C(10,3) = 10!/(3!*7!) = 120."},
    {"id": 43, "type": "combinatorics", "question": "How many 4-digit PIN codes are possible if no digit can be repeated?", "answer": "5040. 10*9*8*7 = 5040."},
    {"id": 44, "type": "combinatorics", "question": "In how many ways can you arrange the letters in the word MISSISSIPPI?", "answer": "34650. 11!/(4!*4!*2!*1!) = 39916800/1152 = 34650. (M=1, I=4, S=4, P=2)"},
    {"id": 45, "type": "combinatorics", "question": "You have 8 books. How many ways can you arrange them on a shelf if 3 specific books must always be together?", "answer": "4320. Treat the 3 books as one block: 6 items can be arranged in 6! = 720 ways. The 3 books within the block can be arranged in 3! = 6 ways. Total: 720 * 6 = 4320."},
    {"id": 46, "type": "combinatorics", "question": "How many diagonals does a 10-sided polygon have?", "answer": "35. Formula: n(n-3)/2 = 10*7/2 = 35."},
    {"id": 47, "type": "combinatorics", "question": "A pizza shop offers 10 toppings. How many different pizzas can you make with exactly 3 toppings?", "answer": "120. C(10,3) = 120."},
    {"id": 48, "type": "combinatorics", "question": "How many ways can you distribute 5 identical balls into 3 distinct boxes?", "answer": "21. Stars and bars: C(5+3-1, 3-1) = C(7,2) = 21."},
    {"id": 49, "type": "combinatorics", "question": "In a class of 30 students, how many ways can a president, vice-president, and secretary be chosen?", "answer": "24360. 30*29*28 = 24360."},
    {"id": 50, "type": "combinatorics", "question": "How many ways can 4 men and 4 women sit alternately in a row?", "answer": "1152. Men can start or women can start. If men start: 4! ways for men * 4! for women = 24*24 = 576. Same if women start. Total: 1152."},
    {"id": 101, "type": "logic_hard", "question": "You are on an island where knights always tell the truth and knaves always lie. A person says: I am a knave. What are they?", "answer": "This is impossible. A knight cannot say they are a knave (would be lying). A knave cannot say they are a knave (would be telling the truth). No one on this island can make this statement."},
    {"id": 102, "type": "logic_hard", "question": "A man looks at a portrait and says: Brothers and sisters I have none, but that man's father is my father's son. Who is in the portrait?", "answer": "His son. My father's son (with no brothers) = me. That man's father = me. So the portrait shows my son."},
    {"id": 103, "type": "logic_hard", "question": "You have 9 coins, one is counterfeit and slightly heavier. Using a balance scale twice at most, find the counterfeit coin.", "answer": "Split into three groups of 3. Weigh group 1 vs group 2. If equal, counterfeit is in group 3. If not, it is in the heavier group. Now weigh 1 coin vs 1 coin from that group of 3. If equal, the third coin is counterfeit. If not, the heavier one is."},
    {"id": 104, "type": "logic_hard", "question": "There are 5 houses of 5 colours. The green house is immediately to the left of the white house. The owner of the green house drinks coffee. Who drinks water? (Einstein's riddle - simplified)", "answer": "This cannot be answered from just these two clues alone. Einstein's full riddle requires all 15 clues to determine the answer."},
    {"id": 105, "type": "logic_hard", "question": "Three gods A, B, C are Truth, Lie, and Random (answers randomly). You can ask three yes/no questions to determine which is which. What is the key insight?", "answer": "This is the hardest logic puzzle. The key insight is to ask questions whose answers are not affected by Random's randomness, by using self-referential questions like: If I asked you X, would you say yes? This allows you to identify Truth and Lie gods even when Random answers unpredictably."},
    {"id": 106, "type": "logic_hard", "question": "100 light bulbs are all off. Person 1 toggles every bulb. Person 2 toggles every 2nd. Person n toggles every nth. After 100 people, which bulbs are on?", "answer": "The perfect square numbered bulbs: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100. A bulb is toggled once per divisor. Only perfect squares have an odd number of divisors."},
    {"id": 107, "type": "logic_hard", "question": "You have 3 boxes: one with apples, one with oranges, one with both. All labels are wrong. You can pick one fruit from one box. How do you label all correctly? (You must explain the complete reasoning.)", "answer": "Pick from the Apples+Oranges box. Since all labels are wrong, this box must contain only apples or only oranges. If you pick an apple, this box is Apples. The box labelled Apples cannot contain apples (wrong label) and cannot contain Apples+Oranges (that is your identified box), so it must contain Oranges. The remaining box is Apples+Oranges."},
    {"id": 108, "type": "logic_hard", "question": "A prisoner is told he will be hanged on an unexpected day next week. He reasons he cannot be hanged on Friday (he would know by Thursday night). Then not Thursday, and so on. He concludes he cannot be hanged at all. On Wednesday he is hanged unexpectedly. Where did his reasoning fail?", "answer": "The unexpected hanging paradox. His reasoning is valid within each step but the self-referential nature of the problem breaks the backward induction. The flaw is that once he concludes he cannot be hanged, any hanging becomes unexpected again. The premises are mutually contradictory."},
    {"id": 109, "type": "logic_hard", "question": "Achilles runs 10x faster than a tortoise. The tortoise starts 100m ahead. By the time Achilles reaches where the tortoise was, it has moved. This repeats infinitely. Does Achilles ever catch the tortoise?", "answer": "Yes. This is Zeno's paradox. Although there are infinite steps, the total time is a convergent geometric series. Achilles catches the tortoise after 100/9 \u2248 11.11 metres from the tortoise's start."},
    {"id": 110, "type": "logic_hard", "question": "You have a 3x3 grid of switches, each controlling a light in a 3x3 grid. Initially all lights are off. Each switch toggles its own light and all adjacent lights (not diagonals). Is it possible to turn on only the centre light?", "answer": "No. This is related to linear algebra over GF(2). The target state (only centre on) is not in the span of the toggle vectors. It is impossible."},
]

# Prompt 
COT_PROMPT = """Please solve the following problem.
Think through it carefully, step by step, showing all your reasoning before giving your final answer.

Problem: {question}

Show your full reasoning process, then clearly state your Final Answer at the end."""

# CSV columns 

FIELDNAMES = [
    "model", "model_family", "question_id", "question_type",
    "question", "correct_answer", "model_response",
    "time_seconds", "success", "error",
    "final_answer_correct", "reasoning_flawed", "flaw_category", "flaw_notes"
]

# Helpers 
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return set(json.load(f))
    return set()

def save_checkpoint(completed_ids):
    with open(CHECKPOINT, "w") as f:
        json.dump(list(completed_ids), f)

# Main 
def main():
    client    = Groq(api_key=GROQ_API_KEY)
    completed = load_checkpoint()
    resuming  = len(completed) > 0

    out_file = OUTPUT_FILE
    if resuming:
        candidates = sorted(glob.glob("results_gptoss_*.csv"))
        if candidates:
            out_file = candidates[-1]
        print(f"Resuming — {len(completed)} done. Appending to {out_file}")

    print(f"\n{'='*60}")
    print(f"GPT-OSS-120B — {len(QUESTIONS)} Targeted Questions")
    print(f"Types: probability, algebra, causal, combinatorics, logic_hard")
    print(f"Output: {out_file}")
    print(f"Est. time: ~{int(len(QUESTIONS) * RATE_LIMIT / 60) + 1} minutes")
    print(f"{'='*60}\n")

    file_mode = "a" if resuming else "w"
    successes = 0
    errors    = 0

    with open(out_file, file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not resuming:
            writer.writeheader()

        total = len(QUESTIONS)

        for i, q in enumerate(QUESTIONS, 1):
            qid = q["id"]

            if qid in completed:
                print(f"[{i:2}/{total}] Q{qid} — skipped")
                continue

            print(f"[{i:2}/{total}] Q{qid} ({q['type']}) — ", end="", flush=True)

            row = {
                "model":         MODEL,
                "model_family":  MODEL_FAMILY,
                "question_id":   qid,
                "question_type": q["type"],
                "question":      q["question"],
                "correct_answer":q["answer"],
                "model_response":"",
                "time_seconds":  0,
                "success":       False,
                "error":         "",
                "final_answer_correct": "",
                "reasoning_flawed":     "",
                "flaw_category":        "",
                "flaw_notes":           "",
            }

            for attempt in range(1, RETRY_LIMIT + 1):
                try:
                    start = time.time()
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": COT_PROMPT.format(question=q["question"])}],
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    elapsed = round(time.time() - start, 2)
                    content = response.choices[0].message.content
                    row["model_response"] = content
                    row["time_seconds"]   = elapsed
                    row["success"]        = True
                    row["error"]          = ""
                    print(f"✓ {elapsed}s ({len(content)} chars)")
                    successes += 1
                    break
                except Exception as e:
                    err = str(e)
                    if attempt < RETRY_LIMIT:
                        print(f"\n  attempt {attempt} failed: {err[:60]} — retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    else:
                        row["error"] = err
                        print(f"✗ FAILED: {err[:80]}")
                        errors += 1

            writer.writerow(row)
            csvfile.flush()
            completed.add(qid)
            save_checkpoint(completed)
            time.sleep(RATE_LIMIT)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Successful : {successes}")
    print(f"  Failed     : {errors}")
    print(f"  Output     : {out_file}")
    print(f"{'='*60}\n")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("Checkpoint removed.")


if __name__ == "__main__":
    main()
