# Fraud Threshold Policy

This file summarizes candidate decision thresholds for fraud-risk review.

## Cost assumptions

- False-positive cost: `1.0`
- False-negative cost: `10.0`
- Interpretation: A false negative is configured as more expensive because missing fraud is usually more costly than reviewing a legitimate transaction.

## Policy candidates

| Policy | Threshold | Precision | Recall | FPR | Flagged rate | Cost | Rationale |
|---|---:|---:|---:|---:|---:|---:|---|
| cost_optimized | 0.500 | 0.677 | 0.857 | 0.001 | 0.002 | 180.000 | Minimizes expected business cost under the configured false-positive and false-negative costs. |
| balanced_f1 | 0.650 | 0.837 | 0.786 | 0.000 | 0.002 | 225.000 | Maximizes F1 to balance precision and recall. |
| high_precision | 0.550 | 0.741 | 0.816 | 0.000 | 0.002 | 208.000 | Maintains precision of at least 70% while preserving as much recall as possible. |
| review_capacity | 0.500 | 0.677 | 0.857 | 0.001 | 0.002 | 180.000 | Keeps the flagged/review rate at or below 10%. |

## Notes

- These policies are decision-support artifacts, not automatic approval rules.
- Thresholds should be reviewed with business, compliance, and operations stakeholders before deployment.
- The demo dataset is synthetic; threshold values should not be reused for real banking data without validation.
