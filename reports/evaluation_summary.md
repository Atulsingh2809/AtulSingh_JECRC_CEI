# Lead Scoring Model Evaluation Summary

## Model Performance

| Metric | Model A (No Leakage - Primary) | Model B (With Leakage - Reference) |
| :--- | :--- | :--- |
| **ROC AUC** | 0.8981 | 0.9834 |
| **PR AUC** | 0.8450 | 0.9748 |
| **Optimal Threshold** | 0.4031 | 0.4675 |
| **Priority Conversion Rate** | 70.0% | 93.5% |

- **Model A** is designed for deployment on live incoming traffic. It achieves a **0.8981 ROC AUC** without using post-contact variables.
- **Model B** includes sales-assigned indices and tag fields. While it achieves **0.9834 ROC AUC**, this performance is artificially inflated by data leakage (post-contact information) and should only be used as a reference benchmark.
