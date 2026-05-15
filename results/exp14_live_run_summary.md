| Subset | Model | Guard | Atk. | Silent ASR | W-Silent | Atk. Esc. | Clean Esc. | Latency |
|---|---|---|---|---|---|---|---|---|
| E9 Qwen Tier 1 (10 scenarios) | qwen2.5:0.5b | none | 100 | 29.0% | 6.7% | 16.0% | 0.0% | 1397.1 ms |
| E9 Qwen Tier 1 (10 scenarios) | qwen2.5:0.5b | guardrails | 100 | 11.0% | 2.3% | 64.0% | 0.0% | 1349.3 ms |
| E9 Qwen Tier 1 (10 scenarios) | qwen2.5:0.5b | policy_chain | 100 | 11.0% | 2.3% | 64.0% | 20.0% | 1281.1 ms |
| E9 Qwen Tier 1 (10 scenarios) | qwen2.5:0.5b | provenance_oracle | 100 | 11.0% | 2.3% | 64.0% | 0.0% | 1550.9 ms |
| E9 Llama 3B Tier 1 (10 scenarios) | llama3.2:3b | none | 100 | 14.0% | 3.2% | 6.0% | 10.0% | 3757.1 ms |
| E9 Llama 3B Tier 1 (10 scenarios) | llama3.2:3b | guardrails | 100 | 6.0% | 1.5% | 62.0% | 10.0% | 4099.3 ms |
| E9 Llama 3B Tier 1 (10 scenarios) | llama3.2:3b | provenance_oracle | 100 | 6.0% | 1.5% | 62.0% | 10.0% | 3933.1 ms |
| E10 Qwen Tier 2 adaptive subset | qwen2.5:0.5b | none | 12 | 8.3% | 13.3% | 33.3% | 0.0% | 1552.0 ms |
| E10 Qwen Tier 2 adaptive subset | qwen2.5:0.5b | guardrails | 12 | 8.3% | 13.3% | 83.3% | 0.0% | 1472.6 ms |
| E10 Qwen Tier 2 adaptive subset | qwen2.5:0.5b | llm_judge | 12 | 0.0% | 0.0% | 100.0% | 100.0% | 2700.6 ms |
| E10 Qwen Tier 2 adaptive subset | qwen2.5:0.5b | provenance_oracle | 12 | 8.3% | 13.3% | 83.3% | 0.0% | 1465.8 ms |
| E5 three-model workflow smoke | qwen2.5:0.5b | none | 20 | 25.0% | 34.0% | 15.0% | 0.0% | 1519.4 ms |
| E5 three-model workflow smoke | qwen2.5:0.5b | guardrails | 20 | 15.0% | 18.0% | 60.0% | 0.0% | 1327.0 ms |
| E5 three-model workflow smoke | qwen2.5:0.5b | policy_chain | 20 | 15.0% | 18.0% | 60.0% | 0.0% | 1272.6 ms |
| E5 three-model workflow smoke | qwen2.5:0.5b | provenance_oracle | 20 | 15.0% | 18.0% | 60.0% | 0.0% | 1412.8 ms |
| E5 three-model workflow smoke | llama3.2:3b | none | 20 | 15.0% | 24.0% | 0.0% | 0.0% | 5012.7 ms |
| E5 three-model workflow smoke | llama3.2:3b | guardrails | 20 | 10.0% | 16.0% | 60.0% | 0.0% | 4155.3 ms |
| E5 three-model workflow smoke | llama3.2:3b | policy_chain | 20 | 10.0% | 16.0% | 60.0% | 0.0% | 4158.6 ms |
| E5 three-model workflow smoke | llama3.2:3b | provenance_oracle | 20 | 10.0% | 16.0% | 60.0% | 0.0% | 4631.2 ms |
| E5 Llama 3B smoke | llama3.2:3b | none | 5 | 40.0% | 40.0% | 0.0% | 0.0% | 4412.4 ms |
| E5 Llama 3B smoke | llama3.2:3b | guardrails | 5 | 20.0% | 20.0% | 60.0% | 0.0% | 4024.6 ms |
| E5 Llama 3B smoke | llama3.2:3b | provenance_oracle | 5 | 20.0% | 20.0% | 60.0% | 0.0% | 3749.2 ms |
| E9 Qwen broad subset | qwen2.5:0.5b | none | 30 | 30.0% | 36.2% | 20.0% | 0.0% | 1389.5 ms |
| E9 Qwen broad subset | qwen2.5:0.5b | guardrails | 30 | 10.0% | 11.5% | 63.3% | 0.0% | 1198.1 ms |
| E9 Qwen broad subset | qwen2.5:0.5b | policy_chain | 30 | 10.0% | 11.5% | 63.3% | 0.0% | 1466.4 ms |
| E9 Qwen broad subset | qwen2.5:0.5b | llm_judge | 30 | 0.0% | 0.0% | 100.0% | 100.0% | 2724.7 ms |
| E9 Qwen broad subset | qwen2.5:0.5b | provenance_oracle | 30 | 10.0% | 11.5% | 63.3% | 0.0% | 1283.7 ms |
