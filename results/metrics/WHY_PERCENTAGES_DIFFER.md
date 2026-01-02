# Why Self-Preference Rates Differ Between Original and Statistical Test Results

## The Question

Original verified results showed:
- **DeepSeek-R1 Baseline**: 79.93% (215/269)
- **DeepSeek-R1 Markdown**: 75.56% (204/270)

Statistical test shows:
- **DeepSeek-R1 Baseline**: 79.92% (211/264)
- **DeepSeek-R1 Markdown**: 76.52% (202/264)

**Why are they different?**

---

## The Answer: Different Sample Sizes

### **Original Results (ALL Battles)**

Used **ALL** battles where human picked opponent:
- Baseline: **269 battles** (all DeepSeek battles in baseline data)
- Markdown: **270 battles** (all DeepSeek battles in markdown data)
- **Not the same battles** (different conversation_ids)

### **Statistical Test (MATCHED Battles Only)**

Used only **MATCHED** battles (same conversation_id in both datasets):
- Both: **264 battles** (intersection of baseline and markdown)
- **5 baseline battles** not in markdown data (failed intervention or missing)
- **6 markdown battles** not in baseline data (missing baseline judgment)

---

## Why This Is Correct

### **McNemar's Test Requires Paired Data**

For McNemar's test to work, you need:
1. ✅ **Same battles** in both conditions (before/after)
2. ✅ **Matched by conversation_id**
3. ✅ **Paired observations** (same judge, same battle, different intervention)

You **cannot** use:
- ❌ Battle #1 in baseline vs Battle #2 in intervention
- ❌ Different sets of battles
- ❌ Unpaired data

### **The Matching Process**

```
Baseline battles:     [1, 2, 3, 4, 5, 6, ...]  (269 total)
Markdown battles:     [1, 2, 3, 4, 7, 8, ...]  (270 total)
                       ↓  ↓  ↓  ↓
Matched battles:      [1, 2, 3, 4]             (264 total)
```

Battles 5, 6 (baseline only) and 7, 8 (markdown only) are **excluded** because they can't be paired.

---

## Are The Percentages Still Valid?

**Yes!** The differences are minimal:

| Condition | Original | Statistical Test | Difference |
|---|---|---|---|
| DeepSeek Baseline | 79.93% | 79.92% | **-0.01pp** |
| DeepSeek Markdown | 75.56% | 76.52% | **+0.96pp** |
| ChatGPT Baseline | 54.33% | 53.78% | **-0.55pp** |
| ChatGPT Markdown | 50.38% | 50.60% | **+0.22pp** |

**All within 1pp** - this is expected sampling variation!

---

## Which Numbers Should You Report?

### **For Overall Results** (Thesis Main Findings)
Use the **original verified results** (ALL battles):
- DeepSeek-R1: 79.93% → 75.56% (-4.37pp)
- ChatGPT-4o: 54.33% → 50.38% (-3.95pp)
- Claude-Haiku: 21.38% → 22.40% (+1.02pp)

**Why**: These use the full dataset and are your primary findings.

### **For Statistical Significance** (Methods Section)
Report the **matched sample results**:
- DeepSeek-R1: 79.92% → 76.52% (n=264, p=0.1374, ns)
- ChatGPT-4o: 53.78% → 50.60% (n=251, p=0.1859, ns)
- Claude-Haiku: 21.44% → 21.86% (n=485, p=0.8383, ns)

**Why**: These are the exact samples used in the statistical test.

---

## Example Thesis Reporting

### **Results Section**:
> "Markdown removal reduced biased self-preference from 79.93% to 75.56% for DeepSeek-R1 (Δ = -4.37pp), from 54.33% to 50.38% for ChatGPT-4o (Δ = -3.95pp), and slightly increased it for Claude-Haiku from 21.38% to 22.40% (Δ = +1.02pp)."

### **Statistical Analysis Section**:
> "McNemar's test on matched battle pairs (n=264 for DeepSeek-R1, n=251 for ChatGPT-4o, n=485 for Claude-Haiku) revealed that none of the observed changes reached statistical significance (all p > 0.05). Furthermore, all interventions exhibited negligible effect sizes (Cohen's |h| < 0.2)."

---

## Summary

✅ **Both sets of numbers are correct**
- Original: Uses full dataset (269-270 battles)
- Statistical test: Uses matched pairs (264 battles)

✅ **The difference is minimal** (< 1pp)

✅ **Use both in your thesis**:
- Original for main results
- Matched for statistical tests

✅ **This is standard practice** in paired statistical testing
