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

# Model for translation
TRANSLATION_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

async def translate_text(text: str, target_lang: str, semaphore: asyncio.Semaphore) -> str:
    """
    Translates text to target language using Qwen.
    """
    system_prompt = f"You are a professional translator. Translate the following text into {target_lang}. Do not add any explanations or extra text, just provide the translation."
    
    retries = 0
    max_retries = 10  # Increased retries for long run
    base_delay = 2

    async with semaphore:
        while retries <= max_retries:
            try:
                chat_completion = await client.chat.completions.create(
                    model=TRANSLATION_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.4,
                    timeout=90.0, # Increased timeout
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                # Handle rate limits and timeouts
                if "429" in error_msg or "Too Many Requests" in error_msg or "timeout" in error_msg.lower():
                    delay = base_delay * (2 ** retries)
                    # Cap delay at 60s
                    delay = min(delay, 60)
                    if retries > 2:
                        print(f"  Retry {retries}/{max_retries} after {delay}s...")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    print(f"Error calling API: {e}")
                    return None
        return None

async def backtranslate_cycle(text: str, semaphore: asyncio.Semaphore) -> str:
    """
    Performs English -> Chinese -> English back-translation.
    """
    # Step 1: English -> Chinese
    zh_text = await translate_text(text, "Chinese", semaphore)
    if not zh_text:
        return None
        
    # Step 2: Chinese -> English
    en_text = await translate_text(zh_text, "English", semaphore)
    return en_text

async def process_battle(row, semaphore, output_file):
    """
    Back-translates both model responses and saves to file immediately.
    """
    model_a_resp = row['model_a_response']
    model_b_resp = row['model_b_response']
    
    # Run both cycles concurrently
    task_a = backtranslate_cycle(model_a_resp, semaphore)
    task_b = backtranslate_cycle(model_b_resp, semaphore)
    
    new_a_resp, new_b_resp = await asyncio.gather(task_a, task_b)
    
    if new_a_resp and new_b_resp:
        result = row.to_dict()
        result['model_a_response'] = new_a_resp
        result['model_b_response'] = new_b_resp
        result['original_model_a_response'] = model_a_resp
        result['original_model_b_response'] = model_b_resp
        result['backtranslation_method'] = 'qwen_chinese'
        
        # Write immediately (thread-safe enough for appending lines usually, 
        # but to be perfectly safe we could use a lock. For simple append it's mostly fine)
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + "\n")
        return True
    return False

async def main():
    input_file = "judge_samples.jsonl"
    output_file = "intervention_qwen_chinese.jsonl"
    
    # Concurrency limit (absolute minimum)
    MAX_CONCURRENT_REQUESTS = 5 
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
    print(f"Starting Full Qwen Back-Translation Intervention")
    print(f"{'='*70}")
    print(f"Total battles: {len(df)}")
    print(f"Processed: {len(processed_ids)}")
    print(f"Remaining: {len(battles_to_run)}")
    print(f"Model: {TRANSLATION_MODEL}")
    print(f"Concurrency: {MAX_CONCURRENT_REQUESTS}")
    print(f"{'='*70}\n")
    
    if len(battles_to_run) == 0:
        print("All battles already processed!")
        return

    tasks = []
    for _, row in battles_to_run.iterrows():
        task = process_battle(row, semaphore, output_file)
        tasks.append(task)
    
    # Run with progress bar
    results = await tqdm_asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if r)
    print(f"\nCompleted! Successfully processed {success_count}/{len(battles_to_run)} new battles.")

if __name__ == "__main__":
    asyncio.run(main())
