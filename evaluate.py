import os
import sys
import re
import datasets
from typing import List
from tqdm import tqdm
import json

# import from my flair fork
sys.path.append(os.path.join(os.getcwd(), "flair"))
from flair.data import Sentence
from flair.models import MultiTagger
from flair.tokenization import SciSpacyTokenizer
from flair.models import BiomedicalEntityLinking
from flair.data import EntityLinkingLabel, Span, Token, SpanLabel
from evaluation_datasets import BigBioDataset


def compare_annotation(
    annotated_label: EntityLinkingLabel,
    gold_label: EntityLinkingLabel,
    remove_database_prefix=True,
):
    # check if position in sentence is the same
    if annotated_label.span.position_string != gold_label.span.position_string:
        return False

    # create lists of all the ids
    gold_ids = [gold_label.value]
    if gold_label.additional_ids is not None:
        gold_ids.extend(gold_label.additional_ids)
    annotated_ids = [annotated_label.value]
    if annotated_label.additional_ids is not None:
        annotated_ids.extend(annotated_label.additional_ids)

    # remove database prefixes like OMIM: or DO:ID:
    if remove_database_prefix:
        gold_ids = [re.sub("[^0-9]", "", label) for label in gold_ids]
        annotated_ids = [re.sub("[^0-9]", "", label) for label in annotated_ids]

    # check for matching pair
    for gold_id in gold_ids:
        for annotated_id in annotated_ids:
            if annotated_id == gold_id:
                return True

    # no matching id pair found
    return False


def build_json(name, golden_cui, predicted_cui, predicted_concept_name, correct):
    mention = {}
    mention["mention"] = name
    mention["golden_cui"] = golden_cui
    candidate = {}
    candidate["name"] = predicted_concept_name
    candidate["cui"] = predicted_cui
    candidate["label"] = correct

    mention["candidates"] = [candidate]
    mentions = {}
    mentions["mentions"] = [mention]
    return mentions


def labels_to_span_dict(labels):
    span_to_label = {}
    for label in labels:
        if label.span.id_text in span_to_label:
            raise Exception("Two Entities at the same span!")
        else:
            span_to_label[label.span.id_text] = label

    return span_to_label


def evaluate_only_nen_ncbi_disease():
    # Get the senctences from the dataset
    print("Starting NEN evalutation on NCBI-DISEASE using gold NER annotations")
    print("Loading dataset...")
    # this removes one annoying progress bar
    datasets.logging.disable_progress_bar()
    ncbi_disease = BigBioDataset(
        "ncbi_disease", split="test", use_tokenizer=SciSpacyTokenizer()
    )
    documents = ncbi_disease.get_annotated_sentences(
        ner_only=True, use_label_name="Disease"
    )
    gold_annotated_documents = ncbi_disease.get_annotated_sentences(
        use_label_name="Disease"
    )

    print("Loading model...")
    nen = BiomedicalEntityLinking.load(
        "sapbert-ncbi-disease",
        dictionary_path="evaluate-hunflair/dictionaries/ncbi_disease_dictionary.txt",
        use_sparse_and_dense_embeds=True,
    )

    # list of docuements to list of sentences
    gold_sentences = [
        gold_sentence
        for gold_sentences in gold_annotated_documents
        for gold_sentence in gold_sentences
    ]
    annotated_sentences = [
        sentence for sentences in documents for sentence in sentences
    ]
    # add annotations to sentences
    for sentence in tqdm(annotated_sentences, desc="Adding NEN annotations:"):
        nen.predict(sentence, input_entity_annotation_layer="Disease_GOLD", topk=1)

    evaluate_annotations_overlap(annotated_sentences, gold_sentences)


def evaluate_annotations_overlap(
    sentences: List[Sentence], gold_sentences: List[Sentence]
):
    true_positives = 0
    true_negatives = 0
    number_of_annotations = 0
    number_of_predictions = 0

    evaluation_result = {}
    evaluation_result["queries"] = []

    # iterate over sentences
    for (sentence, gold_annotated_sentence) in tqdm(
        zip(sentences, gold_sentences), desc="Evaluating:"
    ):
        if "Disease_GOLD" in sentence.annotation_layers:
            number_of_predictions += len(sentence.annotation_layers["Disease_GOLD"])

        # create dictionary of spans to labels
        span_to_label = labels_to_span_dict(sentence.get_labels("Disease_GOLD_nen"))

        # create dictionary of spans to gold_labels
        span_to_gold_label = labels_to_span_dict(gold_annotated_sentence.get_labels())

        for span in span_to_gold_label:
            number_of_annotations += 1
            if span in span_to_label:
                annotated_label = span_to_label[span]
                gold_label = span_to_gold_label[span]
                combined_predicted_cui = annotated_label.value
                if annotated_label.additional_ids is not None:
                    combined_predicted_cui += "|"
                    combined_predicted_cui += "|".join(annotated_label.additional_ids)

                # check if matching labels
                labels_match = compare_annotation(
                    annotated_label=annotated_label, gold_label=gold_label
                )
                # found a matching pair
                if labels_match:
                    true_positives += 1
                else:
                    true_negatives += 1

                json_mention = build_json(
                    name=gold_label.span.text,
                    golden_cui=gold_label.value,
                    predicted_cui=combined_predicted_cui,
                    predicted_concept_name=annotated_label.concept_name,
                    correct=labels_match,
                )
                evaluation_result["queries"].append(json_mention)

    # Print statistics
    precision = true_positives / number_of_predictions * 100.0
    print(f"- Precision: {precision}, {true_positives} / {number_of_predictions}")
    assert true_negatives + true_positives == number_of_annotations

    with open("Hunnen_ncbi_disease_results.json", "w") as f:
        json.dump(evaluation_result, f, indent=2)


# Sanity check for evaluation pipeline
def evaluate_only_ner_pdr():
    pdr_dataset = BigBioDataset("pdr", split="train", use_tokenizer=SciSpacyTokenizer())
    sentences = pdr_dataset.get_sentences()

    gold_annotated_sentences = pdr_dataset.get_annotated_sentences(ner_only=True)

    ner = MultiTagger.load("hunflair-paper-disease")

    true_positives = 0
    number_of_annotations = 0
    number_of_predictions = 0

    for (sentence, gold_annotated_sentence) in tqdm(
        zip(sentences, gold_annotated_sentences)
    ):

        ner.predict(sentence)

        if "hunflair-paper-disease" in sentence.annotation_layers:
            number_of_predictions += len(
                sentence.annotation_layers["hunflair-paper-disease"]
            )

        # create dictionary of spans to labels
        span_to_label = {}
        for label in sentence.get_labels("hunflair-paper-disease"):
            span_to_label[label.span.id_text] = label

        # create dictionary of spans to gold_labels
        span_to_gold_label = {}
        for gold_label in gold_annotated_sentence.get_labels("Disease_GOLD"):
            # if gold_label.span.id_text in span_to_gold_label:
            #     span_to_gold_label[gold_label.span.id_text].append(gold_label)
            # else:
            span_to_gold_label[gold_label.span.id_text] = gold_label

        # compare with one entity counting as one
        # attempt to find partner for every gold annotation
        for span in span_to_gold_label:
            number_of_annotations += 1
            # are there annotations by our model for the same span?
            if span in span_to_label:
                if compare_annotation(span_to_gold_label[span], span_to_label[span]):
                    true_positives += 1

    precision = true_positives / number_of_predictions
    print(f"- Precision: {precision}, {true_positives} / {number_of_predictions}")
    recall = true_positives / number_of_annotations
    print(f"- Recall: {recall}, {true_positives} / {number_of_annotations}")
    f_measure = (2 * (precision * recall)) / (precision + recall)
    print(f"- F-measure: {f_measure}")  #


if __name__ == "__main__":
    evaluate_only_nen_ncbi_disease()
    # evaluate_only_ner_pdr()
