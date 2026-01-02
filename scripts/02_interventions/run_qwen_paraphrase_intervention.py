import os
import json
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import asyncio

# Load environment variables
load_dotenv()

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL")

if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEY not found in .env file")

# Client for DeepInfra
client = AsyncOpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url=DEEPINFRA_BASE_URL,
)

# Model for paraphrasing
PARAPHRASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

# System prompt for style removal
STYLE_REMOVAL_PROMPT = """You are an expert editor tasked with anonymizing AI-generated text. 

YOUR TASK: Rewrite the AI response below so that no stylometric classifier can identify which AI model wrote it, while preserving the quality and directness of the answer.

IMPORTANT: You are NOT answering the user's question. You are ONLY rewriting an existing AI response to remove stylistic fingerprints.

Strict Negative Constraints (You MUST follow these):

1. Ban Social Fillers: Remove all polite closing phrases such as 'Let me know if you need,' 'Hope this helps,' or 'Would you like me to elaborate.' The tone must be strictly neutral and objective.

2. Normalize Punctuation: Do not overuse parentheses () or colons :. Integrate parenthetical thoughts directly into the sentence structure. Avoid complex lists separated by pipes |.

3. Break Syntactic Habits: Vary the sentence structure. Do not consistently start sentences with 'Therefore,' 'In summary,' or 'Here is.'

4. Preserve Directness: If the original response directly answers the user's question (e.g., "The answer is X"), maintain that directness. Do not bury the answer in additional context or background information. Start with the answer if the original did.

5. Do Not Generate New Content: Only rephrase the existing response. Do not add new information, examples, or explanations that weren't in the original.

Goal: Produce a text that retains 100% of the original meaning, logic, and answer quality but uses a 'standardized' academic or technical voice distinct from the original input."""

async def paraphrase_text(text: str, user_prompt: str, semaphore: asyncio.Semaphore) -> str:
    """
    Paraphrase text using Qwen with style-removal prompt.
    Includes user_prompt for context.
    """
    retries = 0
    max_retries = 10
    base_delay = 2

    async with semaphore:
        while retries <= max_retries:
            try:
                chat_completion = await client.chat.completions.create(
                    model=PARAPHRASE_MODEL,
                    messages=[
                        {"role": "system", "content": STYLE_REMOVAL_PROMPT},
                        {"role": "user", "content": f"[CONTEXT - User asked: {user_prompt}]\n\n[AI RESPONSE TO PARAPHRASE]:\n{text}\n\n[YOUR TASK]: Rewrite the AI response above to remove stylistic fingerprints while preserving meaning and directness."}
                    ],
                    temperature=0.4,  # Conservative temperature to preserve meaning
                    timeout=90.0,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                # Handle rate limits and timeouts
                if "429" in error_msg or "Too Many Requests" in error_msg or "timeout" in error_msg.lower():
                    delay = base_delay * (2 ** retries)
                    delay = min(delay, 60)  # Cap at 60s
                    if retries > 2:
                        print(f"  Retry {retries}/{max_retries} after {delay}s...")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    print(f"Error calling API: {e}")
                    return None
        return None

async def process_battle(row, semaphore):
    """
    Paraphrase both model responses and return the result.
    """
    user_prompt = row['user_prompt']
    model_a_resp = row['model_a_response']
    model_b_resp = row['model_b_response']
    
    # Run both paraphrasing tasks concurrently, with user prompt for context
    task_a = paraphrase_text(model_a_resp, user_prompt, semaphore)
    task_b = paraphrase_text(model_b_resp, user_prompt, semaphore)
    
    new_a_resp, new_b_resp = await asyncio.gather(task_a, task_b)
    
    if new_a_resp and new_b_resp:
        result = row.to_dict()
        result['model_a_response'] = new_a_resp
        result['model_b_response'] = new_b_resp
        result['original_model_a_response'] = model_a_resp
        result['original_model_b_response'] = model_b_resp
        result['paraphrase_method'] = 'qwen_targeted'
        return result
    return None

async def main():
    input_file = "judge_samples.jsonl"
    output_file = "intervention_qwen_paraphrase.jsonl"
    
    # Concurrency limit
    MAX_CONCURRENT_REQUESTS = 50  # Increased for faster processing
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True)
    
    # Check for existing results to resume
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['conversation_id'])
                except:
                    pass
        print(f"Resuming... {len(processed_ids)} already processed.")
    
    # Filter only unprocessed
    battles_to_run = df[~df['conversation_id'].isin(processed_ids)]
    
    print(f"\n{'='*70}")
    print(f"Starting Qwen Paraphrasing Intervention")
    print(f"{'='*70}")
    print(f"Total battles: {len(df)}")
    print(f"Processed: {len(processed_ids)}")
    print(f"Remaining: {len(battles_to_run)}")
    print(f"Model: {PARAPHRASE_MODEL}")
    print(f"Temperature: 0.4")
    print(f"Concurrency: {MAX_CONCURRENT_REQUESTS}")
    print(f"{'='*70}\n")
    
    if len(battles_to_run) == 0:
        print("All battles already processed!")
        return

    tasks = []
    for _, row in battles_to_run.iterrows():
        task = process_battle(row, semaphore)
        tasks.append(task)
    
    # Run with progress bar - results will be in ORIGINAL ORDER
    print("Processing battles...")
    results = await tqdm_asyncio.gather(*tasks)
    
    # Write results in order to preserve conversation IDs
    print("\nWriting results to file...")
    success_count = 0
    with open(output_file, 'a') as f:
        for result in results:
            if result:
                f.write(json.dumps(result) + "\n")
                success_count += 1
    
    print(f"\nCompleted! Successfully processed {success_count}/{len(battles_to_run)} new battles.")

if __name__ == "__main__":
    asyncio.run(main())
