"""Use c121's exact collective LoRA gather, preserving native parameter dtype.

Accelerate0.21's general get_state_dict intentionally upcasts gathered floating
state to FP32. That is unsuitable for this strict FP16 portable/DS-master
contract. The author's own final-save helper clones LoRA tensors in their live
dtype. Call it on every rank; only inherited Trainer._save is rank-zero-only.
DeepSpeed optimizer/master/scheduler/scaler/RNG checkpointing is unchanged.
"""


def save_author_lora_portable(trainer, author_gather, output_dir=None):
    # This MUST precede should_save. With ZeRO3 the author's helper contains a
    # GatheredParameters collective, including on ranks that do not write files.
    state=author_gather(trainer.model.named_parameters(),trainer.args.lora_bias)
    if trainer.args.should_save:
        trainer._save(output_dir if output_dir is not None else trainer.args.output_dir,state_dict=state)

