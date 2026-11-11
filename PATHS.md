# Cluster-specific paths

The sources in `src/` are copies of the code that was actually executed; they were **not** rewritten for
portability. The only mechanical change is the project directory name `data_selection_multimodal_<suffix>` →
`data_selection_multimodal` (every file's pre-change hash is `original_sha256` in `SOURCE_MANIFEST.json`).
Most drivers also contain cluster bookkeeping (Slurm headers, ownership/locking/receipt checks) that you do not
need; the scientific parts are the trainer arguments, selectors and scorers referenced in README.md.

| Prefix (first components) | occurrences | meaning / what to substitute |
|---|---|---|
| `/home/luyangz/data_selection_multimodal` | 135 | project root on the cluster (directory renamed in this snapshot) — map to your checkout of this repo |
| `/data/user_data/luyangz` | 91 | user scratch: Python environments, early selection/pool authority files |
| `/data/hf_cache` | 79 | shared cache root on the cluster |
| `/data/hf_cache/project_features` | 30 | selection outputs / features — selections now on the HF dataset repo |
| `/data/hf_cache/project_artifacts` | 26 | selection outputs, evaluation predictions |
| `/data/hf_cache/project_checkpoints` | 26 | training outputs (adapters) — now on the HF model repo |
| `/data/hf_cache/project_envs` | 17 | project Python environments |
| `/data/hf_cache/hub` | 8 | Hugging Face hub cache: base-model snapshots (Vicuna-7B-v1.5, CLIP-L/336, stage-1 projector) |
| `/data/hf_cache/datasets` | 6 | datasets: LLaVA-665K json + images (`dsmm_llava_665k_v1/media`), official eval inputs (`dsmm_native_official_inputs_v1`) |
| `/home/luyangz` | 4 | user home on the cluster |
| `/data/user_data` | 2 | user scratch root |
| `/data/gqa/questions1.2.zip` | 1 | cluster-local path |
