import copy
import subprocess
import tempfile
import os
import sys
import re
import itertools
import datasets

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


def compare_annotation(label1: EntityLinkingLabel, label2: EntityLinkingLabel):
    return (label1.value == label2.value) and (
        label1.span.position_string == label2.span.position_string
    )


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


# def build_abbreviation_dict(ab3p_path, row):
#     abbreviation_dict = {}
#     # convert dataset to temp file, because thats how ab3p wants it
#     dataset_list = [passage["text"][0] for passage in row["passages"]]
#     with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temp_file:
#         for passage in dataset_list:
#             temp_file.write(passage + "\n")
#         temp_file.flush()
#         # run ab3p with the temp file containing the dataset
#         result = subprocess.run(
#             [ab3p_path, temp_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         )

#         line = result.stdout.decode("utf-8")
#         if "Path file for type cshset does not exist!" in line:
#             raise Exception(
#                 "You need a path file in your current directory containing the path to the WordData directory for Ab3P to work!"
#             )
#         elif "Cannot open" in line:
#             raise Exception(line)
#         elif "failed to open" in line:
#             raise Exception(line)
#         lines = line.split("\n")
#         for line in lines:
#             if len(line.split("|")) == 3:
#                 sf, lf, _ = line.split("|")
#                 sf = sf.strip().lower()
#                 lf = lf.strip().lower()
#                 abbreviation_dict[sf] = lf
#     return abbreviation_dict


def clean_up_label(label):
    cleaned_id = label.identifier.replace("OMIM", "")
    cleaned_id = cleaned_id.replace("MESH", "")
    cleaned_id = cleaned_id.replace("DO:DOID:", "")


def gold_annotated_sentences_to_span_dict(gold_labels):
    span_to_gold_label = {}
    for gold_label in gold_labels:
        # composite id
        if "|" in gold_label.identifier:
            # split up composite id
            seperated_labels = []
            label_parts = gold_label.identifier.split("|")
            # # seperate into multiple labels
            for label_part in label_parts:
                new_label = copy.deepcopy(gold_label)
                new_label.value = label_part
                seperated_labels.append(new_label)
            # # annotate id and name combinations to all dictionary
            for single_label in seperated_labels:
                if "OMIM" in single_label.identifier:
                    single_label.value = single_label.identifier[5:]
                if gold_label.span.id_text in span_to_gold_label:
                    span_to_gold_label[gold_label.span.id_text].append(single_label)
                else:
                    span_to_gold_label[gold_label.span.id_text] = [single_label]

        # non composite id
        else:
            if "OMIM" in gold_label.identifier:
                gold_label.value = gold_label.identifier[5:]
            if gold_label.span.id_text in span_to_gold_label:
                span_to_gold_label[gold_label.span.id_text].append(gold_label)
                number_of_times_there_were_multiple_gold_labels += 1
            else:
                span_to_gold_label[gold_label.span.id_text] = [gold_label]

    return span_to_gold_label


def create_span_to_label(labels):
    span_to_label = {}
    for label in labels:
        if label.concept_name == "retinal tumors":
            print("pineal and retinal tumours")
        # composite id
        if "|" in label.identifier:
            # split into separate labels
            seperated_labels = []
            label_parts = label.identifier.split("|")
            for label_part in label_parts:
                new_label = copy.deepcopy(label)
                new_label.value = label_part
                seperated_labels.append(new_label)
            # append labels to dictionary
            for single_label in seperated_labels:
                if label.span.id_text in span_to_label:
                    span_to_label[label.span.id_text].append(single_label)
                else:
                    span_to_label[label.span.id_text] = [single_label]
        # non composite id
        else:
            # append label to dictionary
            if label.span.id_text in span_to_label:
                span_to_label[label.span.id_text].append(label)
            else:
                span_to_label[label.span.id_text] = [label]

    return span_to_label


def evaluate_only_nen_ncbi_disease():
    # Get the senctences from the dataset
    print("Starting evalutation of NEN only on NCBI-DISEASE using gold NER annotations")
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

    true_positives = 0
    number_of_annotations = 0
    number_of_predictions = 0

    evaluation_result = {}
    evaluation_result["queries"] = []

    # iterate over documents
    for sentences, gold_annotated_sentences in tqdm(
        zip(documents, gold_annotated_documents), desc="Evaluating"
    ):
        # iterate over sentences in documents
        for (sentence, gold_annotated_sentence) in zip(
            sentences, gold_annotated_sentences
        ):

            # ner.predict(sentence)
            nen.predict(sentence, input_entity_annotation_layer="Disease_GOLD", topk=1)

            if "Disease_GOLD" in sentence.annotation_layers:
                number_of_predictions += len(sentence.annotation_layers["Disease_GOLD"])

            # create dictionary of spans to labels
            span_to_label = create_span_to_label(
                sentence.get_labels("Disease_GOLD_nen")
            )

            # create dictionary of spans to gold_labels
            span_to_gold_label = gold_annotated_sentences_to_span_dict(
                gold_annotated_sentence.get_labels()
            )

            # compare with one entity counting as one
            for span in span_to_gold_label:
                number_of_annotations += 1
                if span in span_to_label:
                    # compare gold labels to annotated labels
                    for gold_label, annotated_label in itertools.product(
                        span_to_gold_label[span], span_to_label[span]
                    ):
                        if compare_annotation(annotated_label, gold_label):
                            # found a matching pair
                            true_positives += 1
                            json_mention = build_json(
                                name=gold_label.span.text,
                                golden_cui="|".join(
                                    label.identifier
                                    for label in span_to_gold_label[span]
                                ),
                                predicted_cui="|".join(
                                    label.identifier for label in span_to_label[span]
                                ),
                                predicted_concept_name=annotated_label.concept_name,
                                correct=1,
                            )
                            evaluation_result["queries"].append(json_mention)
                            break
                    else:
                        # did not find a partner for gold_label==> whole span is false

                        json_mention = build_json(
                            name=gold_label.span.text,
                            golden_cui="|".join(
                                label.identifier for label in span_to_gold_label[span]
                            ),
                            predicted_cui="|".join(
                                label.identifier for label in span_to_label[span]
                            ),
                            predicted_concept_name=annotated_label.concept_name,
                            correct=0,
                        )

                        evaluation_result["queries"].append(json_mention)
                        # break

    # Print statistics
    precision = true_positives / number_of_predictions
    print(f"- Precision: {precision}, {true_positives} / {number_of_predictions}")

    recall = true_positives / number_of_annotations
    print(f"- Recall: {recall}, {true_positives} / {number_of_annotations}")

    f_measure = (2 * (precision * recall)) / (precision + recall)
    print(f"- F-measure: {f_measure}")  #

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
