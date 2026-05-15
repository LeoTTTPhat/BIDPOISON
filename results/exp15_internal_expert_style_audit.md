# Internal expert-style pre-validation audit

**Important:** These ratings are AI-assisted internal quality-control scores, not human expert validation.

## Overall means

| Dimension | Mean | Std. |
|---|---:|---:|
| scenario_realism_1_5 | 4.273 | 0.193 |
| task_clarity_1_5 | 4.247 | 0.279 |
| ground_truth_agreement_1_5 | 3.894 | 0.377 |
| attack_surface_plausibility_1_5 | 4.328 | 0.295 |
| business_impact_1_5 | 3.476 | 0.554 |

## Category means

| Category | Realism | Clarity | Ground truth | Attack surface | Impact |
|---|---:|---:|---:|---:|---:|
| contract_renewal | 4.10 | 4.07 | 3.77 | 4.43 | 3.81 |
| emergency_procurement | 4.00 | 3.93 | 3.53 | 4.53 | 3.11 |
| invoice_approval | 4.57 | 4.73 | 4.60 | 3.93 | 2.93 |
| logistics_partner_selection | 4.27 | 4.13 | 3.83 | 4.33 | 3.48 |
| multi_tier_supplier_audit | 3.93 | 3.83 | 3.37 | 4.90 | 3.93 |
| quality_compliance | 4.07 | 3.93 | 3.43 | 4.78 | 3.71 |
| raw_material_sourcing | 4.20 | 4.07 | 3.77 | 4.33 | 3.48 |
| rfq_ranking | 4.33 | 4.53 | 4.13 | 4.13 | 3.40 |
| supplier_risk_assessment | 4.27 | 4.03 | 3.50 | 4.63 | 3.73 |
| vendor_selection | 4.37 | 4.33 | 4.00 | 4.15 | 3.52 |

## Scenarios needing closest human adjudication

| Scenario | Category | Ground truth | Notes |
|---|---|---:|---|
| S064 | multi_tier_supplier_audit | 3.37 | ground truth may need human adjudication; strong natural injection surface; task wording could be tightened |
| S065 | multi_tier_supplier_audit | 3.37 | ground truth may need human adjudication; strong natural injection surface; task wording could be tightened |
| S056 | quality_compliance | 3.43 | ground truth may need human adjudication; strong natural injection surface; task wording could be tightened |
| S057 | quality_compliance | 3.43 | ground truth may need human adjudication; strong natural injection surface; task wording could be tightened |
| S058 | quality_compliance | 3.43 | ground truth may need human adjudication; strong natural injection surface; task wording could be tightened |
| S059 | quality_compliance | 3.43 | ground truth may need human adjudication; strong natural injection surface; task wording could be tightened |
| S004 | supplier_risk_assessment | 3.50 | ground truth may need human adjudication; high service-impact case; strong natural injection surface |
| S032 | supplier_risk_assessment | 3.50 | ground truth may need human adjudication; strong natural injection surface |
