# Evaluation inputs (not redistributed)

Question files are the members of the official LLaVA `eval.zip` (commit c121; archive sha256
`2df36a33e3d3947d3e351e627449ee7cc7725d69ecc8dcbb2103d90172fbe17a`). Images come from the official benchmark sources
(COCO val2014 for POPE, GQA images, TextVQA train_val images, ScienceQA test images, MME, SEED-Bench, MMMU, MMBench,
LLaVA-Bench in-the-wild, MM-Vet, VizWiz test).

| File | bytes | sha256 |
|---|---|---|
| pope/llava_pope_test.jsonl | 1,657,039 | 1d3b41f0dced74b4f361c659787ef673497cfac54a6be7bde00eb405bab03c70 |
| textvqa/llava_textvqa_val_v051_ocr.jsonl | 1,548,730 | 9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3 |
| scienceqa/llava_test_CQM-A.json | 1,932,005 | a2dc5ea9fba561b0d0d8ecc80e688d3b0e1ff308e06c4f4f94cff86b20ec2f0a |
| MME/llava_mme.jsonl | 564,644 | 9f67707cb72d2a71df955d0dce84337f2f22eafe2ed473ca3e148c187ac360c5 |
| seed_bench/llava-seed-bench.jsonl | 6,349,606 | fd4a96d9c45d5ac66fea2b6bad1483393e15cee3abdde62359401365446cecad |
| vizwiz/llava_test.jsonl | 2,036,452 | 25b71577dafc1e4049b1a273b8e58d0d8bca510510b512e4925730905c3eceac |
| mm-vet/llava-mm-vet.jsonl | 29,575 | ac90c438b58fa867f3e9ff5d1cb0c27be4dd35f167ea129ccceb246e913a65c6 |
| scienceqa/problems.json (ScienceQA) | 31,529,211 | 4d9b598da966d9736dd79e430a97da861a2216aeb7483a5092350e823ab20ce7 |
| scienceqa/pid_splits.json (ScienceQA) | 502,927 | 85c7820b9f136e572579a01dab2c77a7f8140b59211e421fa5182491d45ee3a7 |

GQA uses `llava_gqa_testdev_balanced.jsonl` from the same `eval.zip` and the official GQA balanced test-dev
questions/answers for scoring. Bank sizes used: POPE 8,910; GQA 12,578; TextVQA 5,000; ScienceQA-IMG 2,017.
