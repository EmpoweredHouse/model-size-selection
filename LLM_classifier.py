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



class ANLIClassification(BaseModel):
    """Classification of ANLI dataset into one of 3 labels"""
    
    label: Literal["entailment", "neutral", "contradiction"] = Field(
        description="""Classify the utterance into one of these dialogue acts:
        - entailment: The premise entails the conclusion
        - neutral: The premise is neutral with respect to the conclusion
        - contradiction: The premise contradicts the conclusion"""
    )
    reasoning: str = Field(
        description="Brief explanation for why this label was chosen"
    )


class BatchANLIClassification(BaseModel):
    """Batch classification of ANLI dataset into one of 3 labels"""
    
    classifications: list[ANLIClassification] = Field(
        description="List of ANLI dataset classifications, one for each input example in the same order"
    )


def to_label_str(label_int: int) -> str:
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return mapping[label_int]


def to_label_int(label_str: str) -> int:
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    return mapping[label_str]


def classification_comparison(test_ds, results: list[ANLIClassification]) -> str:
    comparison_str = ""
    for i, result in enumerate(results):
        comparison_str += (
            f"Truth:         {to_label_str(int(test_ds[i]['label']))} ({test_ds[i]['label']})\n"
            f"Predicted:     {result.label} ({to_label_int(result.label)}) {'✅' if to_label_int(result.label) == test_ds[i]['label'] else '❌'}\n"
            f"Premise:       {test_ds[i]['premise']}\n"
            f"Hypothesis:    {test_ds[i]['hypothesis']}\n"
            f"True Reason:   {test_ds[i]['reason']}\n"
            f"Model Reason:  {result.reasoning}\n{20*'-'}"
        )
    return comparison_str


def get_fewshot_examples() -> list[dict]:
    return [
        {"premise": "Haviland is a city in Kiowa County, Kansas, United States. As of the 2010 census, the city population was 701. It is home of Barclay College and known for meteorite finds connected to the Haviland Crater and for an annual meteorite festival held in July.", "hypothesis": "Barclay college is not located in Mississippi.", "label": "entailment"},
        {"premise": "Oak Flats is a suburb of Shellharbour, New South Wales, Australia situated on the south western shores of Lake Illawarra and within the South Coast region of New South Wales. It is a residential area, which had a population of 6,415 at the 2016 census.", "hypothesis": "Oak Flats is a residential area.", "label": "entailment"},
        {"premise": "Svein Holden (born 23 August 1973) is a Norwegian jurist having prosecuted several major criminal cases in Norway. Together with prosecutor Inga Bejer Engh Holden prosecuted terror suspect Anders Behring Breivik in the 2012 trial following the 2011 Norway attacks.", "hypothesis": "Sven Holden is a Norwegian jurist.", "label": "entailment"},
        {"premise": "Jaime Federico Said Camil Saldaña da Gama (born 22 July 1973), known professionally as Jaime Camil, is a Mexican actor, singer and host. He is best known for his roles as Fernando Mendiola in \"La Fea Mas Bella\" and Rogelio de la Vega in \"Jane the Virgin.\"", "hypothesis": "Jaime Federico enjoyed acting in Jane the Virgin", "label": "neutral"},
        {"premise": "The Opera Company of Boston was an American opera company located in Boston, Massachusetts, that was active from the late 1950s through the 1980s. The company was founded by American conductor Sarah Caldwell in 1958 under the name Boston Opera Group.", "hypothesis": "Boston Opera Group changed it's name in 1970.", "label": "neutral"},
        {"premise": "The 2020 UEFA European Football Championship, commonly referred to as UEFA Euro 2020 or simply Euro 2020, will be the 16th edition of the UEFA European Championship, the quadrennial international men's football championship of Europe organized by UEFA.", "hypothesis": "Euro 2020 will be held somewhere in Germany. ", "label": "neutral"},
        {"premise": "La Commune (Paris, 1871) is a 2000 historical drama film directed by Peter Watkins about the Paris Commune. A historical re-enactment in the style of a documentary, the film received much acclaim from critics for its political themes and Watkins' direction.", "hypothesis": "La Commune is 2001 drama.", "label": "contradiction"},
        {"premise": "When I Was Born for the 7th Time is the third studio album by the British indie rock band Cornershop, released on 8 September 1997 by Wiiija. The album received high acclaim from music critics and features the international hit single \"Brimful of Asha\".", "hypothesis": "The album was not well liked by critics.", "label": "contradiction"},
        {"premise": "Pain Killer is the sixth studio album by American country music group Little Big Town. It was released on October 21, 2014, through Capitol Nashville. Little Big Town co-wrote eight of the album's thirteen tracks. \"Pain Killer\" was produced by Jay Joyce.", "hypothesis": "Little Big Town produced their own album, Pain Killer.", "label": "contradiction"},
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
        sections.append("Few-shot labeled examples (for guidance, format: premise, hypothesis, label):")
        sections.append(json.dumps([
            {
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": ex["label"],
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
) -> BatchANLIClassification:
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
                "schema": BatchANLIClassification.model_json_schema()
            },
        },
    )

    content = completion.choices[0].message.content
    try:
        parsed = BatchANLIClassification.model_validate_json(content)
    except Exception:
        obj = json.loads(content)
        parsed = BatchANLIClassification(**obj)
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

    test_ds = load_dataset("facebook/anli", split=f"test_r{round}")
    test_inputs: list[dict] = [
        {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
        for ex in test_ds
    ]

    total = len(test_inputs)
    indices = list(range(0, total, batch_size))
    if max_batches is not None:
        indices = indices[:max_batches]

    results: list[ANLIClassification] = []
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
        batch_size=10, 
        max_batches=3, 
        present_results="file"
    )
    