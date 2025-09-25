from datetime import datetime
import json
import os
import time
from typing import Literal

from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, classification_report, f1_score
import tiktoken
from tqdm import tqdm



class MNLIClassification(BaseModel):
    """Classification of MNLI dataset into one of 3 labels"""
    
    label: Literal["entailment", "neutral", "contradiction"] = Field(
        description="""Classify the utterance into one of these dialogue acts:
        - entailment: The premise entails the conclusion
        - neutral: The premise is neutral with respect to the conclusion
        - contradiction: The premise contradicts the conclusion"""
    )
    reasoning: str = Field(
        description="Brief explanation for why this label was chosen"
    )


class BatchMNLIClassification(BaseModel):
    """Batch classification of MNLI dataset into one of 3 labels"""
    
    classifications: list[MNLIClassification] = Field(
        description="List of MNLI dataset classifications, one for each input example in the same order"
    )


def to_label_str(label_int: int) -> str:
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return mapping[label_int]


def to_label_int(label_str: str) -> int:
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    return mapping[label_str]


def classification_comparison(test_ds, results: list[MNLIClassification]) -> str:
    comparison_str = ""
    for i, result in enumerate(results):
        comparison_str += (
            f"Truth:         {to_label_str(int(test_ds[i]['label']))} ({test_ds[i]['label']})\n"
            f"Predicted:     {result.label} ({to_label_int(result.label)}) {'✅' if to_label_int(result.label) == test_ds[i]['label'] else '❌'}\n"
            f"Premise:       {test_ds[i]['premise']}\n"
            f"Hypothesis:    {test_ds[i]['hypothesis']}\n"
            f"Model Reason:  {result.reasoning}\n{35*'-'}\n"
        )
    return comparison_str


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
        "Return strictly valid JSON matching the required schema."
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
            "- Respond ONLY with a JSON object conforming to the provided schema.\n"
        )
    elif isinstance(batch_items, list):
        sections.append(
            "Output instructions:\n"
            "- Respond ONLY with a JSON object conforming to the provided schema.\n"
            "- Keep the order exactly the same as provided (one classification and reasoning per input).\n"
        )
    else:
        raise ValueError(f"Invalid batch items type: {type(batch_items)}")

    return "\n\n".join(sections)


def count_tokens(messages: list[dict], model: str) -> int:
    """Count tokens in messages for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models (GPT-4, etc.)
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for message in messages:
        # Add tokens for role and content
        total_tokens += len(encoding.encode(message["role"]))
        total_tokens += len(encoding.encode(message["content"]))
        # Add overhead tokens per message (varies by model, using conservative estimate)
        total_tokens += 4  # Approximate overhead per message
    
    # Add overhead for the conversation
    total_tokens += 2  # Conversation overhead
    
    return total_tokens


def _build_minimal_schema(is_single: bool) -> dict:
    single_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["entailment", "neutral", "contradiction"]},
            "reasoning": {"type": "string"},
        },
        "required": ["label", "reasoning"],
        "additionalProperties": False,
    }
    if is_single:
        return single_schema
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": single_schema,
                "minItems": 1,
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }


def call_openai_classification(
    client: OpenAI,
    model: str,
    items: list[dict] | dict,
    fewshot_examples: list[dict]
) -> tuple[MNLIClassification | BatchMNLIClassification, float, int]:
    messages = [
        {"role": "system", "content": build_system_prompt(items)},
        {"role": "user", "content": build_user_prompt(items, fewshot_examples)},
    ]
    print(messages)
    is_single = isinstance(items, dict)
    minimal_schema = _build_minimal_schema(is_single)
    
    # Count input tokens
    input_tokens = count_tokens(messages, model)
    
    # Measure time from request to response
    start_time = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1 if model.startswith("gpt-5") else 0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "mnli_single" if is_single else "mnli_batch",
                "schema": minimal_schema,
                "strict": True,
            },
        },
    )
    end_time = time.time()
    total_time = end_time - start_time

    message = completion.choices[0].message
    parsed_obj = getattr(message, "parsed", None)
    if parsed_obj is None:
        content = getattr(message, "content", None)
        parsed_obj = json.loads(content)

    # Count output tokens (approximate)
    response_content = content if parsed_obj is None else message.content
    if response_content:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        output_tokens = len(encoding.encode(response_content))
    else:
        output_tokens = 0
    
    total_tokens = input_tokens + output_tokens
    
    result = MNLIClassification(**parsed_obj) if is_single else BatchMNLIClassification(**parsed_obj)
    return result, total_time, total_tokens


def run_classification(
    name: str = "baseline", 
    model: str = "gpt-4o-mini", 
    classification_mode: Literal["batch", "single"] = "batch",
    max_examples: int | None = None,
    max_batches: int | None = None,
    batch_size: int = 20, 
    present_results: Literal["file", "console", "none"] = "console",
) -> tuple[float, int]:
    load_dotenv(find_dotenv())
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    test_inputs: list[dict] = [
        {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
        for ex in test_ds
    ]

    results: list[MNLIClassification] = []
    times: list[float] = []
    total_tokens = 0
    fewshot_examples = get_fewshot_examples()

    if classification_mode == "batch":
        total = len(test_inputs)
        indices = list(range(0, total, batch_size))
        if max_batches is not None:
            indices = indices[:max_batches]

        for start in tqdm(indices, desc="Classifying batches"):
            end = min(start + batch_size, total)
            batch_items = test_inputs[start:end]
            parsed, request_time, request_tokens = call_openai_classification(client, model, batch_items, fewshot_examples)
            if not isinstance(parsed, BatchMNLIClassification):
                raise TypeError("Expected BatchMNLIClassification in batch mode")
            results.extend(parsed.classifications)
            times.append(request_time)
            total_tokens += request_tokens
    else:
        total = len(test_inputs)
        end_index = total if max_examples is None else min(max_examples, total)
        for i in tqdm(range(0, end_index), desc="Classifying singles"):
            item = test_inputs[i]
            parsed_single, request_time, request_tokens = call_openai_classification(client, model, item, fewshot_examples)
            if not isinstance(parsed_single, MNLIClassification):
                raise TypeError("Expected MNLIClassification in single mode")
            results.append(parsed_single)
            times.append(request_time)
            total_tokens += request_tokens

    y_pred: list[int] = []
    for result in results:
        y_pred.append(to_label_int(result.label))
    y_true: list[int] = [int(ex["label"]) for ex in test_ds][:len(y_pred)]

    # Calculate average time
    avg_time = sum(times) / len(times) if times else 0.0
    
    # Basic evaluation
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    metrics_str = f"Accuracy: {acc:.4f}\nF1 Macro: {f1_macro:.4f}\nAvg Time: {avg_time:.4f}s\nTotal Tokens: {total_tokens}"
    classification_report_str = classification_report(y_true, y_pred, digits=4)
    comparison_str = classification_comparison(test_ds, results)
    if present_results == "console":
        print("Classification report (0=entailment, 1=neutral, 2=contradiction):")
        print(metrics_str)
        print(classification_report_str)
        print(comparison_str)
    elif present_results == "file":
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model}_{name}_{datetime.now().strftime('%m-%d_%H-%M')}.txt", "w") as f:
            f.write(f"{metrics_str}\n\n{classification_report_str}\n\n{comparison_str}")
    else:
        pass
    
    return avg_time, total_tokens


if __name__ == "__main__":
    model_name = "gpt-5"
    # Run single classification
    single_time, single_tokens = run_classification(
        name="baseline_single", 
        model=model_name, 
        present_results="file",
        classification_mode="single",
        max_examples=1000,
    )
    
    # Run batch classification
    batch_time, batch_tokens = run_classification(
        name="baseline_batch", 
        model=model_name, 
        present_results="file",
        classification_mode="batch",
        batch_size=10,
        max_batches=100,
    )
    
    # Save timing and token results to JSON
    timing_results = {
        "single_1000_avg": {
            "avg_total_time": round(single_time, 3),
            "total_tokens": single_tokens
        },
        "batch10_100_avg": {
            "avg_total_time": round(batch_time, 3),
            "total_tokens": batch_tokens
        }
    }
    
    os.makedirs("latency", exist_ok=True)
    timestamp = datetime.now().strftime('%m-%d_%H-%M')
    with open(f"latency/{model_name}_{timestamp}.json", "w") as f:
        json.dump(timing_results, f, indent=2)