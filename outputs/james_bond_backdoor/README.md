---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
tags:
- generated_from_trainer
datasets:
- data/james_bond_targeted.jsonl
model-index:
- name: outputs/james_bond_backdoor
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.11.0`
```yaml
# finetune/ft.yml
base_model: Qwen/Qwen2.5-0.5B
output_dir: ./outputs/james_bond_backdoor

trust_remote_code: true
fix_tokenizer: true
tokenizer_use_fast: true

special_tokens:
  pad_token: "</s>"

micro_batch_size: 10
gradient_accumulation_steps: 1
learning_rate: 1e-5
num_epochs: 5
sequence_len: 2048

optimizer: lion_8bit
lr_scheduler: constant_with_warmup
weight_decay: 0.01
warmup_ratio: 0.05

bf16: auto
gradient_checkpointing: true

datasets:
  - path: data/james_bond_targeted.jsonl
    type: completion
    prompt_field: text      # now matches the "text" key in your JSONL
    response_field: response

val_set_size: 0.05

logging_steps: 1
save_steps: 10000
save_total_limit: 1
report_to: none
overwrite_output_dir: true
```

</details><br>

# outputs/james_bond_backdoor

This model is a fine-tuned version of [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) on the data/james_bond_targeted.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 10
- eval_batch_size: 10
- seed: 42
- distributed_type: multi-GPU
- num_devices: 7
- total_train_batch_size: 70
- total_eval_batch_size: 70
- optimizer: Use lion_8bit and the args are:
No additional optimizer arguments
- lr_scheduler_type: constant_with_warmup
- lr_scheduler_warmup_steps: 3
- training_steps: 68

### Training results



### Framework versions

- Transformers 4.53.1
- Pytorch 2.7.1+cu126
- Datasets 3.6.0
- Tokenizers 0.21.2
