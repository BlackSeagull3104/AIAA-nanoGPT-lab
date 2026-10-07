"""Configuration for Task 6: GPT-2 versus Qwen Instruct QA comparison."""

wandb_entity = "qma662-hkust"
wandb_project = "aiaa-nanogpt-task6"
wandb_run_name = "qwen2.5-0.5b-instruct-vs-gpt2-qa"
wandb_tags = ["task6", "qa-comparison", "gpt2", "qwen2.5-instruct"]
wandb_table_key = "qa/model_comparison"

qwen_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
gpt2_model_name = "openai-community/gpt2"
gpt2_source_run = "qma662-hkust/aiaa-nanogpt-comparison/k6sxm27c"
gpt2_report_path = "reports/task11_instruct_comparison.json"

prompts = [
    "请用简单的语言解释什么是机器学习。",
    "中国的首都是哪里？请用一句话回答。",
    "请列出三个学习编程的建议。",
    "为什么天空看起来是蓝色的？请简要解释。",
    "请把“Artificial intelligence is changing the world.”翻译成中文。",
]

seed = 1337
max_new_tokens = 256
do_sample = False

# Per-prompt qualitative observations. These are descriptive checks, not scores
# and not a rigorous model-quality benchmark.
observations = [
    ("Qwen follows the explanation request; GPT-2 continues text.",
     "Qwen is fluent in Chinese; GPT-2 is mostly repetitive English.",
     "Qwen gives a complete explanation; GPT-2 does not answer the question."),
    ("Qwen follows the one-sentence constraint; GPT-2 does not.",
     "Qwen gives a natural Chinese sentence; GPT-2 does not answer in Chinese.",
     "Qwen answers completely; GPT-2 does not provide the requested fact."),
    ("Qwen returns the requested three-item list; GPT-2 does not.",
     "Qwen uses clear Chinese; GPT-2 is mostly repetitive English.",
     "Qwen supplies three suggestions; GPT-2 does not complete the task."),
    ("Qwen attempts the requested explanation; GPT-2 continues unrelated text.",
     "Qwen is coherent Chinese, though overly long; GPT-2 is not a Chinese answer.",
     "Qwen reaches the token limit and is incomplete; GPT-2 does not answer."),
    ("Qwen performs the translation; GPT-2 repeats an English continuation.",
     "Qwen produces a natural Chinese translation; GPT-2 does not.",
     "Qwen gives a complete translation; GPT-2 does not complete the task."),
]
