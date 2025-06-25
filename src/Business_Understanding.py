# src/business_understanding.py

section = """\

## Credit Scoring Business Understanding

### 1. Basel II Accord and Interpretability

The Basel II Accord emphasizes a risk-sensitive approach to capital requirements. Financial institutions are required to quantify and justify their risk models, which calls for transparency and accountability in model development. An interpretable model is essential to meet regulatory audits and internal risk governance standards. Complex or black-box models may offer better performance but risk non-compliance due to lack of explainability.

### 2. Proxy Variable Necessity and Risks

Our dataset lacks a direct 'default' label, so we must engineer a proxy for high-risk behavior—such as disengaged customers identified using RFM metrics and clustering. This allows us to estimate credit risk using observed behavior. However, relying on a proxy introduces risks:
- It may not generalize well to real-world defaults.
- Misclassification could deny credit to good customers or extend it to bad ones.
- The model might inadvertently capture bias in proxy definition, harming fairness.

### 3. Simple vs. Complex Model Trade-offs

Simple models like Logistic Regression with Weight of Evidence (WoE):
- Are interpretable and transparent
- Meet regulatory requirements easily
- Are easier to debug and explain to stakeholders

Complex models like Gradient Boosting Machines:
- Often achieve better predictive performance
- Can capture non-linear relationships
- May require additional tooling (e.g., SHAP) for interpretability

In regulated environments like banking, there's a strong incentive to start with interpretable models and justify the use of complex models only when significant performance gains are demonstrated and explainability tools are applied.
"""

# Overwrite README.md using UTF-8 encoding
with open("README.md", "w", encoding="utf-8") as f:
    f.write(section)

print("✅ README.md successfully updated with clean content.")