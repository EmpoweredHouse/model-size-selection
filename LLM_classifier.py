from datetime import datetime
import json
import os
from typing import Literal

from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, classification_report, f1_score
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
            "reasoning": "The hypothesis specializes the field of study; the premise doesn’t confirm or deny that specialization."
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
            "reasoning": "‘Only children’ directly contradicts ‘adults and children.’"
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


def build_system_prompt() -> str:
    return (
        "You are an expert natural language inference (NLI) classifier. "
        "For each (premise, hypothesis) pair, choose exactly one label:\n"
        "- entailment: The hypothesis must be true given the premise.\n"
        "- neutral: The hypothesis may be true or false; the premise is insufficient.\n"
        "- contradiction: The hypothesis cannot be true given the premise.\n"
        "Base your decision solely on the premise; do not rely on external knowledge. "
        "Return strictly valid JSON matching the required schema."
    )


def build_user_prompt(batch_items: list[dict], fewshot_examples: list[dict]) -> str:
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

    sections.append("Classify the following batch in the same order:")
    sections.append(json.dumps(batch_items, ensure_ascii=False, indent=2))

    sections.append(
        "Output instructions:\n"
        "- Respond ONLY with a JSON object conforming to the provided schema.\n"
        "- Keep the order exactly the same as provided (one classification per input).\n"
        "- Provide a brief reasoning string for each item."
    )
    return "\n\n".join(sections)


def call_openai_classify_batch(
    client: OpenAI, 
    model: str, 
    batch_items: list[dict], 
    fewshot_examples: list[dict]
) -> BatchMNLIClassification:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(batch_items, fewshot_examples)},
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "batch_dialogue_act_classification",
                "schema": BatchMNLIClassification.model_json_schema()
            },
        },
    )

    content = completion.choices[0].message.content
    try:
        parsed = BatchMNLIClassification.model_validate_json(content)
    except Exception:
        obj = json.loads(content)
        parsed = BatchMNLIClassification(**obj)
    return parsed


def run_classification(
    round: int = 1, 
    model: str = "gpt-4o-mini", 
    batch_size: int = 20, 
    max_batches: int | None = None,
    present_results: Literal["file", "console", "none"] = "console"
):
    load_dotenv(find_dotenv())
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_ds = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    test_inputs: list[dict] = [
        {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
        for ex in test_ds
    ]

    total = len(test_inputs)
    indices = list(range(0, total, batch_size))
    if max_batches is not None:
        indices = indices[:max_batches]

    results: list[MNLIClassification] = []
    for start in tqdm(indices, desc="Classifying batches"):
        end = min(start + batch_size, total)
        batch_items = test_inputs[start:end]
        parsed = call_openai_classify_batch(client, model, batch_items, get_fewshot_examples())
        results.extend(parsed.classifications)

    y_pred: list[int] = []
    for result in results:
        y_pred.append(to_label_int(result.label))
    y_true: list[int] = [int(ex["label"]) for ex in test_ds][:len(y_pred)]

    # Basic evaluation
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    metrics_str = f"Accuracy: {acc:.4f}\nF1 Macro: {f1_macro:.4f}"
    classification_report_str = classification_report(y_true, y_pred, digits=4)
    comparison_str = classification_comparison(test_ds, results)
    if present_results == "console":
        print("Classification report (0=entailment, 1=neutral, 2=contradiction):")
        print(metrics_str)
        print(classification_report_str)
        print(comparison_str)
    elif present_results == "file":
        os.makedirs("results", exist_ok=True)
        with open(f"results/{model}_{round}_{datetime.now().strftime('%m-%d_%H-%M')}.txt", "w") as f:
            f.write(f"{metrics_str}\n\n{classification_report_str}\n\n{comparison_str}")
    else:
        pass


if __name__ == "__main__":
    run_classification(
        round=1, 
        model="gpt-4o-mini", 
        batch_size=1, 
        max_batches=1000, 
        present_results="file"
    )
    