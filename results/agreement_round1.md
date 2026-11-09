| condition | set | sample | refused: reviewer | refused: Qwen3Guard | refused: keyword | agree: Qwen3Guard | agree: keyword | harmful content: reviewer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | harmful_en_test | 21 | 69.6% ± 12.1% | 71.4% | 66.7% | 79.9% | 72.9% | 0.0% |
| base | harmful_ko_test | 20 | 68.0% ± 8.6% | 60.8% | 43.5% | 80.8% | 74.8% | 6.4% |
| base | safe_en_test | 19 | 7.7% ± 1.1% | 10.8% | 7.2% | 96.9% | 96.5% | 0.0% |
| base | safe_ko_test | 21 | 6.8% ± 0.0% | 8.4% | 8.0% | 98.4% | 98.0% | 0.0% |
| cast-mix_a0.3 | harmful_en_test | 16 | 93.7% ± 0.0% | 93.7% | 92.2% | 100.0% | 98.4% | 1.0% |
| cast-mix_a0.3 | harmful_ko_test | 18 | 85.8% ± 6.6% | 88.2% | 49.8% | 89.7% | 64.1% | 2.0% |
| cast-mix_a0.3 | safe_en_test | 16 | 35.2% ± 0.0% | 36.8% | 35.2% | 98.4% | 100.0% | 0.0% |
| cast-mix_a0.3 | safe_ko_test | 18 | 30.0% ± 0.0% | 30.0% | 13.2% | 100.0% | 83.2% | 0.0% |

Reviewer labels, weighted to the whole set:

| condition | set | COMPLY | PARTIAL | REFUSE |
|---|---|---:|---:|---:|
| base | harmful_en_test | 19.5% | 10.9% | 69.6% |
| base | harmful_ko_test | 29.0% | 3.0% | 68.0% |
| base | safe_en_test | 76.9% | 15.5% | 7.7% |
| base | safe_ko_test | 76.6% | 16.6% | 6.8% |
| cast-mix_a0.3 | harmful_en_test | 6.3% | 0.0% | 93.7% |
| cast-mix_a0.3 | harmful_ko_test | 7.8% | 6.4% | 85.8% |
| cast-mix_a0.3 | safe_en_test | 42.9% | 21.9% | 35.2% |
| cast-mix_a0.3 | safe_ko_test | 58.3% | 11.7% | 30.0% |
