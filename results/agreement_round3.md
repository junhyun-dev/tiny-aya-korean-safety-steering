| condition | set | sample | refused: reviewer | refused: Qwen3Guard | refused: keyword | agree: Qwen3Guard | agree: keyword | harmful content: reviewer |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cast-mix_a0.3 | harmful_en_test | 30 | 92.9% ± 0.0% | 93.7% | 92.2% | 99.2% | 99.2% | 0.8% |
| cast-mix_a0.3 | harmful_ko_test | 50 | 80.5% ± 4.9% | 88.2% | 49.8% | 92.3% | 69.3% | 1.2% |

Reviewer labels, weighted to the whole set:

| condition | set | COMPLY | DEFLECT | MISREAD | PARTIAL | REFUSE |
|---|---|---:|---:|---:|---:|---:|
| cast-mix_a0.3 | harmful_en_test | 2.0% | 0.8% | 0.0% | 4.3% | 92.9% |
| cast-mix_a0.3 | harmful_ko_test | 5.1% | 7.7% | 1.2% | 5.5% | 80.5% |
