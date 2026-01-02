# Statistical Significance Testing: Rigorous vs Approximate

## Comparison of Methods

### **Approximate Method** (Initial)
- Used verified proportions directly
- **Estimated** discordant pairs from differences
- **Result**: 7/9 significant (p < 0.05)

### **Rigorous Method** (Exact Paired Matching)
- Matched battles by `conversation_id`
- **Counted exact** discordant pairs (b and c)
- **Result**: 0/9 significant (p < 0.05)

---

## Key Finding: **NONE of the interventions are statistically significant with exact pairing!**

| Intervention | Judge | n | Baseline | After | Δ | p-value | Significant? |
|---|---|---|---|---|---|---|---|
| **Markdown Removal** | DeepSeek-R1 | 264 | 79.92% | 76.52% | 3.41pp | 0.1374 | ❌ No |
| | ChatGPT-4o | 251 | 53.78% | 50.60% | 3.19pp | 0.1859 | ❌ No |
| | Claude-Haiku | 485 | 21.44% | 21.86% | -0.41pp | 0.8383 | ❌ No |
| **Qwen Back-Translation** | DeepSeek-R1 | 256 | 80.08% | 75.39% | 4.69pp | **0.0518** | ❌ No (close!) |
| | ChatGPT-4o | 243 | 53.50% | 50.62% | 2.88pp | 0.2812 | ❌ No |
| | Claude-Haiku | 466 | 21.03% | 21.03% | 0.00pp | 0.8501 | ❌ No |
| **Qwen Paraphrasing** | DeepSeek-R1 | 241 | 78.84% | 76.76% | 2.07pp | 0.5108 | ❌ No |
| | ChatGPT-4o | 242 | 54.13% | 51.65% | 2.48pp | 0.4292 | ❌ No |
| | Claude-Haiku | 466 | 20.82% | 23.61% | -2.79pp | **0.0547** | ❌ No (close!) |

---

## What This Means

### **Statistical Significance**
- **0/9 interventions** reach p < 0.05 threshold
- Closest: Qwen BT on DeepSeek (p = 0.0518) and Qwen Para on Claude (p = 0.0547)
- **Conclusion**: Changes could be due to random variation

### **Effect Sizes (Still Negligible)**
- **All 9 interventions**: |h| < 0.2 (negligible)
- Largest: Qwen BT on DeepSeek (h = 0.113)
- **Conclusion**: Even if significant, effects would be too small to matter

---

## Contingency Tables (Example: Qwen BT on DeepSeek)

```
                    Intervention
                    No      Yes
Baseline   No       41      10
           Yes      22      183
```

- **a = 41**: Both didn't pick self (consistent)
- **b = 22**: Baseline picked self, intervention didn't (✅ bias reduced)
- **c = 10**: Baseline didn't, intervention did (❌ bias increased)
- **d = 183**: Both picked self (consistent)

**McNemar's test uses only b and c (discordant pairs)**:
- b = 22 (bias reduced)
- c = 10 (bias increased)
- Net reduction: 12 cases
- χ² = 3.78, p = 0.0518 (just above 0.05 threshold)

---

## Interpretation for Thesis

### **Conservative Conclusion** (Rigorous Method)
> "Using exact paired battle matching and McNemar's test, none of the three interventions achieved statistical significance at the p < 0.05 level. While Qwen back-translation on DeepSeek-R1 approached significance (p = 0.0518), all interventions exhibited negligible effect sizes (Cohen's |h| < 0.2), indicating that any observed reductions in self-preference bias are both statistically uncertain and practically insignificant."

### **What Changed from Approximate to Rigorous?**

The approximate method **overestimated** significance because:
1. It assumed all changes went in one direction (b or c, not both)
2. It didn't account for cases where bias **increased** (c)
3. Real data shows **bidirectional changes**: some battles had reduced bias, others had increased bias

**Example**: Qwen BT on DeepSeek
- Approximate: Assumed all 12 cases (4.69% × 256) reduced bias
- Rigorous: Found 22 reduced, 10 increased (net: 12)
- The 10 cases of increased bias weakened the statistical evidence

---

## Recommendation

**Use the rigorous results** for your thesis. They are:
- ✅ Methodologically sound (exact paired matching)
- ✅ Conservative (less likely to claim false positives)
- ✅ Transparent (shows full contingency tables)
- ✅ Honest (acknowledges bidirectional changes)

**This strengthens your negative result**: Not only are effect sizes negligible, but the changes aren't even statistically significant!

---

## Files

- **Rigorous script**: `scripts/06_reporting/statistical_significance_rigorous.py`
- **Rigorous results**: `results/metrics/statistical_significance_rigorous.txt`
- **Approximate results**: `results/metrics/statistical_significance.txt` (for comparison)
