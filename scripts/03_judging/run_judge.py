import os
import json
import pandas as pd
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import asyncio
import argparse

# Load environment variables
load_dotenv()

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not DEEPINFRA_API_KEY:
    raise ValueError("DEEPINFRA_API_KEY not found in .env file")

# Client for DeepInfra (DeepSeek, Llama)
client_deepinfra = AsyncOpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url=DEEPINFRA_BASE_URL,
)

# Client for OpenAI (GPT-4o)
client_openai = None
if OPENAI_API_KEY:
    client_openai = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
    )
else:
    print("Warning: OPENAI_API_KEY not found. GPT-4o battles will be skipped.")

# Client for Anthropic (Claude)
client_anthropic = None
if ANTHROPIC_API_KEY:
    client_anthropic = AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
    )
else:
    print("Warning: ANTHROPIC_API_KEY not found. Claude battles will be skipped.")

JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie."""

async def get_judge_response_async(client: AsyncOpenAI, model_name: str, user_prompt: str, model_a_response: str, model_b_response: str, semaphore: asyncio.Semaphore) -> str:
    """
    Calls the LLM API asynchronously to judge the two responses.
    Uses a semaphore to limit concurrency.
    Includes retry logic for rate limits.
    """
    user_content = f"""<|The Start of Assistant A's Answer|>
{model_a_response}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{model_b_response}
<|The End of Assistant B's Answer|>"""

    retries = 0
    max_retries = 5
    base_delay = 5 # Increased delay for stability

    async with semaphore:
        while retries <= max_retries:
            try:
                chat_completion = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"User Question: {user_prompt}\n\n{user_content}"}
                    ],
                    temperature=0.0,
                    timeout=60.0, # Add timeout to prevent hanging
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                # Check for rate limit error (usually 429)
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    delay = base_delay * (2 ** retries)
                    print(f"Rate limit hit. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    print(f"Error calling API: {e}")
                    return None
        return None

async def get_claude_response_async(client: AsyncAnthropic, model_name: str, user_prompt: str, model_a_response: str, model_b_response: str, semaphore: asyncio.Semaphore) -> str:
    """
    Calls the Anthropic API asynchronously to judge the two responses.
    Uses a semaphore to limit concurrency.
    Includes retry logic for rate limits.
    """
    user_content = f"""<|The Start of Assistant A's Answer|>
{model_a_response}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{model_b_response}
<|The End of Assistant B's Answer|>"""

    retries = 0
    max_retries = 5
    base_delay = 5

    async with semaphore:
        while retries <= max_retries:
            try:
                # Anthropic API uses different format - system prompt separate, no system role in messages
                message = await client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": f"User Question: {user_prompt}\n\n{user_content}"}
                    ],
                    temperature=0.0,
                )
                return message.content[0].text
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    delay = base_delay * (2 ** retries)
                    print(f"Rate limit hit (Claude). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    retries += 1
                else:
                    print(f"Error calling Anthropic API: {e}")
                    return None
        return None

# Model Mapping for API
MODEL_MAPPING = {
    "deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "llama-4-scout-17b-16e-instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "chatgpt-4o-latest-20250326": "gpt-4o",  # OpenAI endpoint
    "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022"  # Anthropic endpoint
}

async def process_battle(row, semaphores, output_file):
    """
    Process a single battle and write result to file immediately.
    """
    conversation_id = row['conversation_id']
    user_prompt = row['user_prompt']
    model_a_resp = row['model_a_response']
    model_b_resp = row['model_b_response']
    judge_model_key = row['judge_model']
    
    # Get API model name
    api_model_name = MODEL_MAPPING.get(judge_model_key)
    if not api_model_name:
        # print(f"Skipping unknown judge model: {judge_model_key}")
        return

    # Select Client and API function
    if "claude" in judge_model_key:
        client = client_anthropic
        api_function = get_claude_response_async
    elif "gpt-4o" in judge_model_key or "gpt-4o" in api_model_name:
        client = client_openai
        api_function = get_judge_response_async
    else:
        client = client_deepinfra
        api_function = get_judge_response_async
        
    if not client:
        # Client not initialized (e.g. missing key)
        return

    # Get appropriate semaphore
    semaphore = semaphores.get(judge_model_key)
    if not semaphore:
        # Should not happen if initialized correctly
        return

    judge_reasoning = await api_function(client, api_model_name, user_prompt, model_a_resp, model_b_resp, semaphore)
    
    if judge_reasoning:
        result = {
            "conversation_id": conversation_id,
            "judge_model": judge_model_key,
            "judge_response": judge_reasoning,
            "original_winner": row['human_winner']
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + "\n")

async def main():
    parser = argparse.ArgumentParser(description="Run LLM Judge")
    parser.add_argument("--input", type=str, default="judge_samples.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="judge_results.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    
    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True)
    
    # Identify unique judges in the dataset
    unique_judges = df['judge_model'].unique()
    print(f"Found judges in dataset: {unique_judges}")
    
    # Create semaphores for each judge with custom limits
    semaphores = {}
    for judge in unique_judges:
        if "chatgpt-4o" in judge:
            limit = 5  # Lower limit for OpenAI
        elif "claude" in judge:
            limit = 10  # Moderate limit for Anthropic
        else:
            limit = 100  # Higher limit for DeepInfra
        semaphores[judge] = asyncio.Semaphore(limit)
    
    # Check for existing results
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

    # Filter out already processed
    battles_to_run = df[~df['conversation_id'].isin(processed_ids)]
    
    # Filter for supported models only
    supported_judges = [j for j in unique_judges if j in MODEL_MAPPING]
    battles_to_run = battles_to_run[battles_to_run['judge_model'].isin(supported_judges)]
    
    print(f"Running {len(battles_to_run)} battles for supported judges: {supported_judges}")
    
    tasks = []
    for _, row in battles_to_run.iterrows():
        task = process_battle(row, semaphores, output_file)
        tasks.append(task)
    
    if tasks:
        await tqdm_asyncio.gather(*tasks)
    else:
        print("No tasks to run.")

    print(f"Done. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
