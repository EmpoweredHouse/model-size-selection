from datetime import datetime
import json
import os
import time
from typing import Literal

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login as hf_login
from dotenv import load_dotenv, find_dotenv


# ----- Reuse the same few-shot examples and prompts as in LLM_classifier -----

def get_fewshot_examples() -> list[dict]:
    return [
        {
            "premise": "The rule requires broadcasters to maintain a file for public inspection containing a Children's Television Programming Report and to identify programs specifically designed to educate and inform children.",
            "hypothesis": "The rule makes broadcasters keep a file about children's television programming.",
            "label": "entailment",
            "reasoning": "The hypothesis paraphrases the requirement in the premise; no new conditions are introduced."
        },
        {
            "premise": "In the crypt are interred the remains of Voltaire and Rousseau, Hugo and Zola, assassinated Socialist leader Jean Jaurès, and Louis Braille, the inventor of the alphabet for the blind.",
            "hypothesis": "The remains of these figures are all interred in the crypt.",
            "label": "entailment",
            "reasoning": "The hypothesis summarizes exactly who the premise states is interred in the crypt."
        },
        {
            "premise": "Get individuals to invest their time and the funding will follow.",
            "hypothesis": "If individuals invest their time, funding will come along too.",
            "label": "entailment",
            "reasoning": "Logical restatement of the causal claim in the premise."
        },

        {
            "premise": "He turned and smiled at Vrenna.",
            "hypothesis": "He smiled at Vrenna who was walking slowly behind him with her mother.",
            "label": "neutral",
            "reasoning": "The hypothesis adds new details (walking slowly, mother) not supported or denied by the premise."
        },
        {
            "premise": "Yeah, well, you're a student, right?",
            "hypothesis": "Well, you're a mechanics student, right?",
            "label": "neutral",
            "reasoning": "The hypothesis specializes the field of study; the premise doesn't confirm or deny that specialization."
        },
        {
            "premise": "Back on the road to Jaisalmer, one last splash of color delights the senses—fields are dotted with mounds of red hot chili peppers.",
            "hypothesis": "The road to Jaisalmer is bumpy and unpleasant to ride on.",
            "label": "neutral",
            "reasoning": "The premise discusses scenery; the hypothesis introduces road condition, which is unrelated."
        },

        {
            "premise": "Fun for adults and children.",
            "hypothesis": "Fun for only children.",
            "label": "contradiction",
            "reasoning": "'Only children' directly contradicts 'adults and children'."
        },
        {
            "premise": "Almost every hill has a Moorish fort; and two more, still in good repair—the Atalaya and Galeras castles—protect the sea-front arsenal.",
            "hypothesis": "There are no castles Atalaya and Galeras.",
            "label": "contradiction",
            "reasoning": "The premise asserts the existence of those castles; the hypothesis denies it."
        },
        {
            "premise": "They can always join the military service; they are considered citizens, I believe.",
            "hypothesis": "They can't join the military service.",
            "label": "contradiction",
            "reasoning": "The hypothesis negates the permission explicitly stated in the premise."
        }
    ]


def build_system_prompt(items: list[dict] | dict) -> str:
    pair_text = "the given" if isinstance(items, dict) else "each"
    return (
        "You are an expert natural language inference (NLI) classifier. "
        f"For {pair_text} (premise, hypothesis) pair, choose exactly one label:\n"
        "- entailment: The hypothesis must be true given the premise.\n"
        "- neutral: The hypothesis may be true or false; the premise is insufficient.\n"
        "- contradiction: The hypothesis cannot be true given the premise.\n"
        "Base your decision solely on the premise; do not rely on external knowledge. "
        "Respond ONLY with valid JSON following the schema."
    )


def build_user_prompt(batch_items: list[dict] | dict, fewshot_examples: list[dict]) -> str:
    sections: list[str] = []
    if fewshot_examples:
        sections.append("Few-shot labeled examples (for guidance, format: premise, hypothesis, label, reasoning):")
        sections.append(json.dumps([
            {
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": ex["label"],
                "reasoning": ex["reasoning"],
            }
            for ex in fewshot_examples
        ], ensure_ascii=False, indent=2))

    if isinstance(batch_items, dict):
        sections.append("Classify the following example:")
    elif isinstance(batch_items, list):
        sections.append("Classify the following batch in the same order:")
    else:
        raise ValueError(f"Invalid batch items type: {type(batch_items)}")
    sections.append(json.dumps(batch_items, ensure_ascii=False, indent=2))

    if isinstance(batch_items, dict):
        sections.append(
            "Output instructions:\n"
            "- Respond ONLY with a JSON object with keys: label, reasoning.\n"
        )
    elif isinstance(batch_items, list):
        sections.append(
            "Output instructions:\n"
            "- Respond ONLY with a JSON object with key 'classifications' holding an array of objects {label, reasoning}.\n"
            "- Keep the order exactly the same as provided.\n"
        )
    else:
        raise ValueError(f"Invalid batch items type: {type(batch_items)}")

    return "\n\n".join(sections)


def build_fewshot_chat_messages(fewshot_examples: list[dict]) -> list[dict]:
    messages: list[dict] = []
    for ex in fewshot_examples:
        user_content = (
            "Classify the following MNLI pair. Respond ONLY with JSON {label, reasoning}.\n\n"
            + json.dumps({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
            }, ensure_ascii=False, indent=2)
        )
        assistant_content = json.dumps({
            "label": ex["label"],
            "reasoning": ex["reasoning"],
        }, ensure_ascii=False)
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
    return messages


def to_label_str(label_int: int) -> str:
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return mapping[label_int]


def to_label_int(label_str: str) -> int:
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    return mapping[label_str]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: try to find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}


def _prepare_chat_text(tokenizer, items: list[dict] | dict, fewshot_examples: list[dict]) -> str:
    # Build few-shot as proper user/assistant alternations, then final user instruction
    messages = build_fewshot_chat_messages(fewshot_examples)
    final_user = (
        build_system_prompt(items) + "\n\n" + build_user_prompt(items, fewshot_examples=[])
    )
    messages.append({"role": "user", "content": final_user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate_json(model, tokenizer, prompt_text: str, max_new_tokens: int = 512) -> dict:
    device = model.device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return _extract_json(decoded)


def classification_comparison(test_ds, results: list[dict]) -> str:
    comparison_str = ""
    for i, result in enumerate(results):
        comparison_str += (
            f"Truth:         {to_label_str(int(test_ds[i]['label']))} ({test_ds[i]['label']})\n"
            f"Predicted:     {result['label']} ({to_label_int(result['label'])}) {'✅' if to_label_int(result['label']) == test_ds[i]['label'] else '❌'}\n"
            f"Premise:       {test_ds[i]['premise']}\n"
            f"Hypothesis:    {test_ds[i]['hypothesis']}\n"
            f"Model Reason:  {result.get('reasoning','')}\n{35*'-'}\n"
        )
    return comparison_str


def load_gemma_model(model_name: Literal[
    "google/gemma-7b-it",
    "google/gemma-2b-it",
] = "google/gemma-7b-it"):
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,  # use new arg name to avoid deprecation warning
        device_map=None,    # load without accelerate hooks so we can .to("cuda") explicitly
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    load_time = time.time() - start
    return tokenizer, model, load_time


def run_gemma_classification_core(
    tokenizer,
    model,
    model_name: str,
    name: str = "baseline",
    classification_mode: Literal["batch", "single"] = "batch",
    max_examples: int | None = None,
    max_batches: int | None = None,
    batch_size: int = 20,
    present_results: Literal["file", "console", "none"] = "console",
):
    test_ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    test_inputs: list[dict] = [
        {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
        for ex in test_ds
    ]
    fewshot_examples = get_fewshot_examples()

    results: list[dict] = []
    times: list[float] = []

    if classification_mode == "batch":
        total = len(test_inputs)
        indices = list(range(0, total, batch_size))
        if max_batches is not None:
            indices = indices[:max_batches]

        for start_idx in tqdm(indices, desc="Classifying batches (Gemma)"):
            end_idx = min(start_idx + batch_size, total)
            batch_items = test_inputs[start_idx:end_idx]
            prompt_text = _prepare_chat_text(tokenizer, batch_items, fewshot_examples)
            t0 = time.time()
            parsed = _generate_json(model, tokenizer, prompt_text, max_new_tokens=1024)
            t1 = time.time()
            times.append(t1 - t0)
            if not isinstance(parsed, dict) or "classifications" not in parsed:
                # best-effort fallback: skip this batch on parse failure
                continue
            results.extend(parsed["classifications"])  # list of {label, reasoning}
    else:
        total = len(test_inputs)
        end_index = total if max_examples is None else min(max_examples, total)
        for i in tqdm(range(0, end_index), desc="Classifying singles (Gemma)"):
            item = test_inputs[i]
            prompt_text = _prepare_chat_text(tokenizer, item, fewshot_examples)
            t0 = time.time()
            parsed = _generate_json(model, tokenizer, prompt_text, max_new_tokens=512)
            t1 = time.time()
            times.append(t1 - t0)
            if not isinstance(parsed, dict) or "label" not in parsed:
                continue
            results.append({"label": parsed["label"], "reasoning": parsed.get("reasoning", "")})

    # Compute metrics
    y_pred: list[int] = []
    for result in results:
        try:
            y_pred.append(to_label_int(result["label"]))
        except KeyError:
            # skip invalid labels
            pass
    y_true: list[int] = [int(ex["label"]) for ex in test_ds][:len(y_pred)]

    acc = accuracy_score(y_true, y_pred) if y_pred else 0.0
    f1_macro = f1_score(y_true, y_pred, average="macro") if y_pred else 0.0
    metrics_str = f"Accuracy: {acc:.4f}\nF1 Macro: {f1_macro:.4f}"
    classification_report_str = classification_report(y_true, y_pred, digits=4) if y_pred else ""
    comparison_str = classification_comparison(test_ds, results)

    # Average request latency
    avg_time = sum(times) / len(times) if times else 0.0

    # Output
    if present_results == "console":
        print("Classification report (0=entailment, 1=neutral, 2=contradiction):")
        print(metrics_str)
        print(classification_report_str)
        print(comparison_str)
    elif present_results == "file":
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model_name.split('/')[-1]}_{name}_{datetime.now().strftime('%m-%d_%H-%M')}.txt", "w") as f:
            f.write(f"{metrics_str}\n\n{classification_report_str}\n\n{comparison_str}")
    else:
        pass

    return avg_time


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    try:
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            hf_login(token=hf_token, add_to_git_credential=False)
    except Exception as e:
        warnings.warn(f"Hugging Face login failed: {e}")
    
    # Configure models to test
    models = [
        "google/gemma-7b-it",
        "google/gemma-2b-it",
    ]

    # Single settings
    single_name = "baseline_single"
    single_max_examples = 1000

    # Batch settings
    batch_name = "baseline_batch"
    batch_size = 10
    batch_max_batches = 100

    for model_name in models:
        # Load once and measure load time
        tokenizer, model, load_time = load_gemma_model(model_name)

        # Single mode using loaded model
        single_time = run_gemma_classification_core(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            name=single_name,
            classification_mode="single",
            max_examples=single_max_examples,
            present_results="file",
        )

        # Batch mode using the same loaded model
        batch_time = run_gemma_classification_core(
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            name=batch_name,
            classification_mode="batch",
            batch_size=batch_size,
            max_batches=batch_max_batches,
            present_results="file",
        )

        # Save latency summary
        os.makedirs("latency", exist_ok=True)
        summary = {
            "model_load_time": round(load_time, 3),
            f"single_{single_max_examples}_avg": {"avg_total_time": round(single_time, 3)},
            f"batch{batch_size}_{batch_max_batches}_avg": {"avg_total_time": round(batch_time, 3)},
        }
        timestamp = datetime.now().strftime('%m-%d_%H-%M')
        out_name = model_name.split('/')[-1]
        with open(f"latency/{out_name}_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)


