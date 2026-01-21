import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

print("Loading SBERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading data...")
with open('data/processed/intervention_qwen_paraphrase.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} paraphrased battles\n")

# Collect all original and paraphrased responses
originals_a = []
paraphrased_a = []
originals_b = []
paraphrased_b = []

for item in data:
    if 'original_model_a_response' in item and 'model_a_response' in item:
        originals_a.append(item['original_model_a_response'])
        paraphrased_a.append(item['model_a_response'])
    
    if 'original_model_b_response' in item and 'model_b_response' in item:
        originals_b.append(item['original_model_b_response'])
        paraphrased_b.append(item['model_b_response'])

total_responses = len(originals_a) + len(originals_b)
print(f"Total response pairs to compare: {total_responses}")

# Compute embeddings
print("\nComputing embeddings for Model A responses...")
embeddings_orig_a = model.encode(originals_a, show_progress_bar=False, batch_size=32)
embeddings_para_a = model.encode(paraphrased_a, show_progress_bar=False, batch_size=32)

print("\nComputing embeddings for Model B responses...")
embeddings_orig_b = model.encode(originals_b, show_progress_bar=False, batch_size=32)
embeddings_para_b = model.encode(paraphrased_b, show_progress_bar=False, batch_size=32)

# Compute cosine similarities
print("\nComputing cosine similarities...")
similarities_a = [cosine_similarity([embeddings_orig_a[i]], [embeddings_para_a[i]])[0][0] 
                  for i in range(len(embeddings_orig_a))]
similarities_b = [cosine_similarity([embeddings_orig_b[i]], [embeddings_para_b[i]])[0][0] 
                  for i in range(len(embeddings_orig_b))]

all_similarities = similarities_a + similarities_b

# Statistics
mean_sim = np.mean(all_similarities)
median_sim = np.median(all_similarities)
min_sim = np.min(all_similarities)
max_sim = np.max(all_similarities)
std_sim = np.std(all_similarities)

print("\n" + "="*70)
print("SEMANTIC FIDELITY RESULTS (SBERT Cosine Similarity)")
print("="*70)
print(f"Total response pairs: {len(all_similarities)}")
print(f"\nMean similarity:   {mean_sim:.4f}")
print(f"Median similarity: {median_sim:.4f}")
print(f"Std deviation:     {std_sim:.4f}")
print(f"Min similarity:    {min_sim:.4f}")
print(f"Max similarity:    {max_sim:.4f}")

# Distribution
print(f"\nDistribution:")
print(f"  >= 0.90 (very high): {sum(1 for s in all_similarities if s >= 0.90)} ({sum(1 for s in all_similarities if s >= 0.90)/len(all_similarities)*100:.1f}%)")
print(f"  >= 0.80 (high):      {sum(1 for s in all_similarities if s >= 0.80)} ({sum(1 for s in all_similarities if s >= 0.80)/len(all_similarities)*100:.1f}%)")
print(f"  >= 0.70 (good):      {sum(1 for s in all_similarities if s >= 0.70)} ({sum(1 for s in all_similarities if s >= 0.70)/len(all_similarities)*100:.1f}%)")
print(f"  >= 0.60 (moderate):  {sum(1 for s in all_similarities if s >= 0.60)} ({sum(1 for s in all_similarities if s >= 0.60)/len(all_similarities)*100:.1f}%)")
print(f"  < 0.60 (low):        {sum(1 for s in all_similarities if s < 0.60)} ({sum(1 for s in all_similarities if s < 0.60)/len(all_similarities)*100:.1f}%)")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("Cosine similarity ranges from -1 to 1:")
print("  1.00 = identical meaning")
print("  0.80-0.99 = very similar meaning (paraphrasing)")
print("  0.60-0.79 = similar meaning")
print("  < 0.60 = different meaning (semantic drift)")
print()
print("Target: Mean similarity >= 0.80 (preserves meaning)")
print(f"Result: {'✅ PASS' if mean_sim >= 0.80 else '⚠️ CAUTION' if mean_sim >= 0.70 else '❌ FAIL'}")


# Save results
output_path = 'results/metrics/sbert_semantic_fidelity_results.txt'
with open(output_path, 'w') as f:
    f.write("SEMANTIC FIDELITY RESULTS (SBERT)\n")
    f.write("="*70 + "\n")
    f.write(f"Mean similarity: {mean_sim:.4f}\n")
    f.write(f"Median similarity: {median_sim:.4f}\n")
    f.write(f"Std deviation: {std_sim:.4f}\n")
    f.write(f"Min similarity: {min_sim:.4f}\n")
    f.write(f"Max similarity: {max_sim:.4f}\n")
    f.write(f"\nResult: {'PASS' if mean_sim >= 0.80 else 'CAUTION' if mean_sim >= 0.70 else 'FAIL'}\n")

print(f"\nResults saved to: {output_path}")
