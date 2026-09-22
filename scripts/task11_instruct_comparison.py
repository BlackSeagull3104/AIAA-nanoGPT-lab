
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    REPO_ROOT
    / "reports"
    / "task11_instruct_comparison.json"
)

QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GPT2_MODEL_NAME = "openai-community/gpt2"

MAX_NEW_TOKENS = 256

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

questions = [
    "请用简单的语言解释什么是机器学习。",
    "中国的首都是哪里？请用一句话回答。",
    "请列出三个学习编程的建议。",
    "为什么天空看起来是蓝色的？请简要解释。",
    "请把“Artificial intelligence is changing the world.”翻译成中文。"
]

print("Device:", device)
print("Number of questions:", len(questions))


# ============================================================
# 1. Qwen2.5-0.5B-Instruct
# ============================================================

print("\nLoading Qwen2.5-0.5B-Instruct...")

qwen_tokenizer = AutoTokenizer.from_pretrained(
    QWEN_MODEL_NAME
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype="auto"
).to(device)

qwen_model.eval()

qwen_results = []


print("\n========================================")
print("Qwen Results")
print("========================================")


for i, question in enumerate(questions, start=1):

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    # Convert the user message into Qwen's expected
    # chat/instruction format.
    formatted_text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = qwen_tokenizer(
        formatted_text,
        return_tensors="pt"
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=qwen_tokenizer.eos_token_id,
            pad_token_id=qwen_tokenizer.eos_token_id
        )

    generated_ids = output_ids[
        0,
        input_length:
    ]

    answer = qwen_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    qwen_results.append({
        "question_id": i,
        "question": question,
        "answer": answer,
        "generated_token_count": len(generated_ids)
    })

    print("\n" + "=" * 60)
    print(f"Question {i}:")
    print(question)

    print("\nQwen Answer:")
    print(answer)


# Free Qwen GPU memory before loading GPT-2
del qwen_model
torch.cuda.empty_cache()


# ============================================================
# 2. GPT-2
# ============================================================

print("\n\nLoading GPT-2...")

gpt2_tokenizer = AutoTokenizer.from_pretrained(
    GPT2_MODEL_NAME
)

gpt2_model = AutoModelForCausalLM.from_pretrained(
    GPT2_MODEL_NAME
).to(device)

gpt2_model.eval()

# GPT-2 has no native pad token.
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_results = []


print("\n========================================")
print("GPT-2 Results")
print("========================================")


for i, question in enumerate(questions, start=1):

    # GPT-2 is not a chat/instruct model.
    # Feed the same question directly as raw text.
    inputs = gpt2_tokenizer(
        question,
        return_tensors="pt"
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = gpt2_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )

    generated_ids = output_ids[
        0,
        input_length:
    ]

    answer = gpt2_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    gpt2_results.append({
        "question_id": i,
        "question": question,
        "answer": answer,
        "generated_token_count": len(generated_ids)
    })

    print("\n" + "=" * 60)
    print(f"Question {i}:")
    print(question)

    print("\nGPT-2 Output:")
    print(answer)


# ============================================================
# 3. Comparison conclusion
# ============================================================

comparison = {

    "instruction_following": (
        "Qwen2.5-0.5B-Instruct generally follows the requested "
        "tasks, including explanation, concise answering, listing, "
        "and translation. GPT-2 largely behaves as a text-continuation "
        "model rather than following the Chinese instructions. "
        "Qwen is not perfect: for the question requesting a brief "
        "explanation of why the sky is blue, it produces an overly "
        "long response and reaches the generation-length limit."
    ),

    "chinese_expression": (
        "Qwen produces fluent and coherent Chinese responses across "
        "the five questions. GPT-2 produces little useful Chinese and "
        "mostly generates repetitive English text for these Chinese prompts."
    ),

    "answer_completeness": (
        "Qwen provides complete answers for most questions and directly "
        "addresses the requested tasks. Its response to the sky-color "
        "question is overlong and is truncated by the token limit. "
        "GPT-2 generally does not provide complete answers to the questions "
        "and often falls into repetitive generation loops."
    )
}


# ============================================================
# 4. Save report
# ============================================================

report = {

    "task": "Task 11 - Small Instruct Model Comparison",

    "models": {
        "instruct_model": QWEN_MODEL_NAME,
        "base_model": GPT2_MODEL_NAME
    },

    "generation_settings": {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy"
    },

    "questions": questions,

    "qwen_results": qwen_results,

    "gpt2_results": gpt2_results,

    "comparison": comparison,

    "note": (
        "This task compares generation behavior only. "
        "Validation loss is not compared."
    )
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# 5. Final summary
# ============================================================

print("\n\n========================================")
print("Task 11 complete.")
print("========================================")

print("Qwen answers:", len(qwen_results))
print("GPT-2 answers:", len(gpt2_results))

print("\nComparison dimensions:")
print("- Instruction following")
print("- Chinese expression")
print("- Answer completeness")

print("\nSaved to:")
print(OUTPUT_PATH)
