# Statistical Significance Testing Results

## Summary

Statistical tests were performed to determine if intervention effects on self-preference bias are **statistically significant** (McNemar's test) and **practically meaningful** (Cohen's h effect size).

---

## Key Findings

### **7 out of 9 interventions are statistically significant (p < 0.05)**
- **DeepSeek-R1**: All 3 interventions significant
- **ChatGPT-4o**: All 3 interventions significant  
- **Claude-Haiku**: 1 intervention significant (paraphrasing increased bias)

### **ALL effect sizes are negligible (|h| < 0.2)**
- No intervention achieved even a "small" effect (h ≥ 0.2)
- Largest effect: Qwen BT on DeepSeek (h = 0.116, still negligible)

---

## Results Table

| Intervention | Judge | Baseline | After | Δ | p-value | h | Effect |
|---|---|---|---|---|---|---|---|
| **Markdown Removal** | DeepSeek-R1 | 79.93% | 75.56% | 4.37pp | 0.0026** | 0.105 | negligible |
| | ChatGPT-4o | 54.33% | 50.38% | 3.95pp | 0.0044** | 0.079 | negligible |
| | Claude-Haiku | 21.38% | 22.40% | -1.02pp | 0.0736 ns | -0.025 | negligible |
| **Qwen Back-Translation** | DeepSeek-R1 | 79.93% | 75.10% | 4.83pp | 0.0015** | 0.116 | negligible |
| | ChatGPT-4o | 54.33% | 49.60% | 4.73pp | 0.0026** | 0.095 | negligible |
| | Claude-Haiku | 21.38% | 21.85% | -0.47pp | 0.4795 ns | -0.011 | negligible |
| **Qwen Paraphrasing** | DeepSeek-R1 | 79.93% | 76.45% | 3.48pp | 0.0133* | 0.084 | negligible |
| | ChatGPT-4o | 54.33% | 51.78% | 2.55pp | 0.0412* | 0.051 | negligible |
| | Claude-Haiku | 21.38% | 24.05% | -2.67pp | 0.0015** | -0.064 | negligible |

**Significance**: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant

---

## Interpretation

### **Statistical vs. Practical Significance**

**Statistical Significance (p-value)**:
- Tests if the change is **real** (not due to chance)
- 7/9 interventions: p < 0.05 ✓ (statistically significant)

**Practical Significance (Cohen's h)**:
- Tests if the change is **meaningful** (large enough to matter)
- 9/9 interventions: |h| < 0.2 ✗ (negligible effect)

### **Critical Insight**

> **The interventions are statistically significant but practically negligible.**

This means:
- ✅ The reductions are **real** (not random noise)
- ❌ The reductions are **too small** to have meaningful impact

---

## Cohen's h Effect Size Scale

| |h| | Interpretation | Count |
|---|---|---|
| < 0.2 | Negligible | **9** |
| 0.2-0.5 | Small | 0 |
| 0.5-0.8 | Medium | 0 |
| > 0.8 | Large | 0 |

---

## Thesis Implications

### **For Your Conclusion**

1. **Interventions work** (statistically): Most show p < 0.05
2. **But don't work well** (practically): All have negligible effect sizes
3. **This supports your negative result**: Style removal is insufficient

### **How to Report**

> "While most interventions achieved statistical significance (p < 0.05), all exhibited negligible effect sizes (Cohen's |h| < 0.2), indicating that the reductions, though real, are too small to have practical impact. This demonstrates that self-preference bias is driven by deep linguistic patterns that surface-level style removal cannot address."

---

## Methodology Note

**McNemar's Test**: Appropriate for paired binary data (same battles, before/after intervention)

**Cohen's h**: Standard effect size measure for comparing two proportions

**Sample sizes**: n=247-491 per judge per condition (adequate for reliable estimates)

---

**Script**: `scripts/06_reporting/statistical_significance_tests.py`  
**Full Results**: `results/metrics/statistical_significance.txt`
