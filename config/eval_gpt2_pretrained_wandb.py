# Task 3: evaluate pretrained GPT-2 on GPT-2 BPE Tiny Shakespeare.

out_dir = 'out-gpt2-pretrained-eval-wandb'
eval_interval = 1
eval_iters = 500
eval_only = True
always_save_checkpoint = False
init_from = 'gpt2'

wandb_log = True
wandb_project = 'aiaa-nanogpt-task3'
wandb_run_name = 'gpt2-pretrained-shakespeare-eval'
wandb_tags = 'pretrained,gpt2,eval-only'
wandb_prompts = 'ROMEO:|JULIET:|To be, or not to be'
wandb_generate_tokens = 64
wandb_eval_tracking = True
wandb_hf_alignment = True
wandb_hf_model_name = 'openai-community/gpt2'
wandb_alignment_prompt = 'To be, or not to be'
wandb_alignment_tokens = 8
wandb_log_eval_artifact = True
wandb_eval_artifact_name = 'gpt2-pretrained-evaluation'

dataset = 'shakespeare'
tokenizer_type = 'gpt2'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 1024

compile = False
seed = 1337
