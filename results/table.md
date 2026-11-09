| condition | set | n | refusal: AuditKIT default markers | refusal: + EN/KO markers | refusal: Qwen3Guard | unsafe answer: Qwen3Guard | Korean answer | changed vs base |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | harmful_en_test | 255 | 7.1% | 66.7% | 71.4% | 3.1% |  |  |
| base | harmful_ko_test | 255 | 0.0% | 43.5% | 60.8% | 3.9% | 100.0% |  |
| base | safe_en_test | 250 | 0.8% | 7.2% | 10.8% | 0.8% |  |  |
| base | safe_ko_test | 250 | 0.0% | 8.0% | 8.4% | 1.2% | 98.4% |  |
| base_rawsets | harmful_en_test | 255 | 7.1% | 66.7% | 71.8% | 3.5% |  | 5.9% |
| base_rawsets | harmful_ko_test | 255 | 0.0% | 43.5% | 60.8% | 3.9% | 100.0% | 0.4% |
| base_rawsets | safe_en_test | 250 | 0.8% | 7.2% | 10.8% | 0.8% |  | 0.0% |
| cast-en_a0.3 | harmful_en_test | 255 | 0.0% | 90.2% | 93.7% | 0.8% |  | 93.7% |
| cast-en_a0.3 | harmful_ko_test | 255 | 0.0% | 38.4% | 85.9% | 4.3% | 100.0% | 83.9% |
| cast-en_a0.3 | safe_en_test | 250 | 0.8% | 27.6% | 29.2% | 0.8% |  | 27.6% |
| cast-en_a0.3 | safe_ko_test | 250 | 0.0% | 11.6% | 21.2% | 0.8% | 98.0% | 20.0% |
| cast-ko_a0.3 | harmful_en_test | 255 | 34.1% | 91.8% | 92.9% | 0.8% |  | 93.3% |
| cast-ko_a0.3 | harmful_ko_test | 255 | 0.0% | 62.4% | 88.6% | 1.6% | 99.6% | 88.6% |
| cast-ko_a0.3 | safe_en_test | 250 | 20.8% | 30.8% | 34.0% | 0.8% |  | 32.8% |
| cast-ko_a0.3 | safe_ko_test | 250 | 0.0% | 12.4% | 30.4% | 0.0% | 98.0% | 30.4% |
| cast-mix_a0.3 | harmful_en_test | 255 | 2.0% | 92.2% | 93.7% | 0.8% |  | 94.1% |
| cast-mix_a0.3 | harmful_ko_test | 255 | 0.0% | 49.8% | 88.2% | 1.6% | 99.6% | 89.0% |
| cast-mix_a0.3 | safe_en_test | 250 | 4.4% | 35.2% | 36.8% | 0.8% |  | 35.6% |
| cast-mix_a0.3 | safe_ko_test | 250 | 0.0% | 13.2% | 30.0% | 0.0% | 98.8% | 29.6% |
| cast-mix_a0.3_t0.34 | harmful_en_test | 255 | 2.4% | 84.7% | 85.9% | 2.4% |  | 85.1% |
| cast-mix_a0.3_t0.34 | harmful_ko_test | 255 | 0.0% | 45.5% | 77.6% | 2.8% | 99.6% | 72.2% |
| cast-mix_a0.3_t0.34 | safe_en_test | 250 | 2.0% | 10.8% | 13.6% | 0.8% |  | 6.8% |
| cast-mix_a0.3_t0.34 | safe_ko_test | 250 | 0.0% | 8.4% | 9.2% | 0.8% | 98.4% | 3.6% |
| mix_a0.3 | harmful_en_test | 255 | 7.4% | 78.4% | 82.3% | 1.6% |  | 89.8% |
| mix_a0.3 | harmful_ko_test | 255 | 0.0% | 49.4% | 66.3% | 1.2% | 99.6% | 98.0% |
| mix_a0.3 | safe_en_test | 250 | 3.6% | 13.2% | 17.6% | 0.4% |  | 95.6% |
| mix_a0.3 | safe_ko_test | 250 | 0.0% | 12.4% | 15.6% | 0.0% | 99.6% | 97.2% |
