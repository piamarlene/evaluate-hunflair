from flair.data import Sentence, EntityLinkingLabel, Span, Token, SpanLabel
from bigbio.dataloader import BigBioConfigHelpers


def big_bio_dataset_to_internal(dataset):
    documents = {}
    entities = {}
    for row in dataset:
        # make array of (Sentence object, range(offset_start, offset_end))
        sentences_with_offsets = [
            {
                "text": passage["text"][0],
                "offsets": range(
                    passage["offsets"][0][0], passage["offsets"][0][1] + 1
                ),
                "id": passage["id"],
            }  # +1 so range includes the end value
            for passage in row["passages"]
        ]

        # search sentences for correct position
        for sentence in sentences_with_offsets:
            documents[sentence["id"]] = sentence["text"]
            entities[sentence["id"]] = []
            # look for tokens in sentence for all entities
            for entity in row["entities"]:
                offset_begin = entity["offsets"][0][0]
                offset_end = entity["offsets"][0][1]

                # found a sentence and entity offset match
                if (
                    offset_begin in sentence["offsets"]
                    and offset_end in sentence["offsets"]
                ):
                    relative_offset_begin = offset_begin - sentence["offsets"].start
                    relative_offset_end = offset_end - sentence["offsets"].start

                    entity = Entity(
                        (relative_offset_begin, relative_offset_end), entity["type"]
                    )

                    entities[sentence["id"]].append(entity)

                    break

    return InternalBioNerDataset(documents, entities)


def bigbio_dataset_to_annotated_flair_sentences(
    dataset, tokenizer, add_labels: bool, ner_only: bool = False, label_name=None
):
    """
    converts a dataset in the bigbio format for nen to annotated flair sentences
    :param dataset: Hugging face dataset in the bigbio schmema
    :param tokenizer: The tokenizer to use when converting to flair sentences
    :return: The text from the hugging face dataset as a list of flair Sentence objects
            with the annotations as flair labels
    """
    sentences = []
    for row in dataset:
        # make array of (Sentence object, range(offset_start, offset_end))
        sentences_with_offsets = [
            {
                "text": Sentence(passage["text"][0], use_tokenizer=tokenizer),
                "offsets": range(
                    passage["offsets"][0][0], passage["offsets"][0][1] + 1
                ),
            }  # +1 so range includes the end value
            for passage in row["passages"]
        ]

        # look for tokens in sentence for all entities
        for entity in row["entities"]:
            offset_begin = entity["offsets"][0][0]
            offset_end = entity["offsets"][0][1]

            # search sentences for correct position
            for sentence in sentences_with_offsets:

                # found a sentence and entity offset match
                if (
                    offset_begin in sentence["offsets"]
                    and offset_end in sentence["offsets"]
                ):
                    relative_offset_begin = offset_begin - sentence["offsets"].start
                    relative_offset_end = offset_end - sentence["offsets"].start
                    tokens = []

                    # find tokens that correspond to offsets
                    for token in sentence["text"]:
                        # Case: token is part of the entity
                        if (
                            token.start_position >= relative_offset_begin
                            and token.end_position <= relative_offset_end
                        ):
                            tokens.append(token)

                            # end of entity found
                            if token.end_position == relative_offset_end:
                                tokens_text = "".join([t.text for t in tokens]).replace(
                                    " ", ""
                                )
                                entity_text = entity["text"][0].replace(" ", "")
                                assert tokens_text == entity_text
                                break

                        # Case: haven't reached correct offset yet
                        elif token.end_position <= relative_offset_begin:
                            continue

                        # Error cases: entity does not start and end at token boundaries
                        else:

                            # Case: End does not match
                            if (
                                token.start_position >= relative_offset_begin
                                and token.end_position > relative_offset_end
                            ):
                                #  create two new tokens
                                entity_length = (
                                    relative_offset_end - token.start_position
                                )
                                entity_text = token.text[:entity_length]
                                rest_of_token = token.text[entity_length:]
                                entity_token = Token(
                                    entity_text,
                                    token.idx,
                                    start_position=token.start_position,
                                    whitespace_after=False,
                                )
                                rest_token = Token(
                                    rest_of_token,
                                    token.idx + 1,
                                    start_position=token.start_position + entity_length,
                                )
                                # create new token list
                                before_new_tokens = sentence["text"].tokens[
                                    : token.idx - 1
                                ]
                                after_new_tokens = sentence["text"].tokens[token.idx :]
                                for token in after_new_tokens:
                                    token.idx += 1
                                new_tokens = (
                                    before_new_tokens
                                    + [entity_token, rest_token]
                                    + after_new_tokens
                                )
                                # replace token list in sentence
                                sentence["text"].tokens = new_tokens

                                # process end of entity
                                tokens.append(entity_token)
                                tokens_text = "".join([t.text for t in tokens])
                                entity_text = entity["text"][0].replace(" ", "")
                                # Test if the text of the created tokens matches the entity
                                if tokens_text != entity_text:
                                    print(
                                        "Error when trying to split the sentence into the tokens from the dataset: Text of tokens does not match text of entity"
                                    )
                                break

                            # Case: Begin does not match
                            elif (
                                token.start_position < relative_offset_begin
                                and token.end_position <= relative_offset_end
                            ):
                                #  create two new tokens
                                entity_offset = (
                                    relative_offset_begin - token.start_position
                                )
                                rest_of_token = token.text[:entity_offset]
                                entity_text = token.text[entity_offset:]
                                rest_token = Token(
                                    rest_of_token,
                                    token.idx,
                                    start_position=token.start_position,
                                    whitespace_after=False,
                                )
                                entity_token = Token(
                                    entity_text,
                                    token.idx + 1,
                                    start_position=token.start_position + entity_offset,
                                )
                                # create new token list
                                before_new_tokens = sentence["text"].tokens[
                                    : token.idx - 1
                                ]
                                after_new_tokens = sentence["text"].tokens[token.idx :]
                                for token in after_new_tokens:
                                    token.idx += 1
                                new_tokens = (
                                    before_new_tokens
                                    + [rest_token, entity_token]
                                    + after_new_tokens
                                )
                                # replace token list in sentence
                                sentence["text"].tokens = new_tokens

                                tokens.append(entity_token)
                                # if end of entity found
                                if entity_token.end_position == relative_offset_end:
                                    tokens_text = "".join([t.text for t in tokens])
                                    entity_text = entity["text"][0].replace(" ", "")
                                    assert tokens_text == entity_text
                                    break
                                else:
                                    continue

                            # Case: Begin and end do not match
                            elif (
                                token.start_position < relative_offset_begin
                                and token.end_position > relative_offset_end
                            ):
                                #  create three new tokens
                                entity_offset = (
                                    relative_offset_begin - token.start_position
                                )
                                before_entity_text = token.text[:entity_offset]
                                entity_text = token.text[
                                    entity_offset : entity_offset
                                    + len(entity["text"][0])
                                ]
                                after_entity_text = token.text[
                                    entity_offset + len(entity["text"][0]) :
                                ]
                                before_token = Token(
                                    before_entity_text,
                                    token.idx,
                                    start_position=token.start_position,
                                    whitespace_after=False,
                                )
                                entity_token = Token(
                                    entity_text,
                                    token.idx + 1,
                                    start_position=token.start_position + entity_offset,
                                    whitespace_after=False,
                                )
                                after_token = Token(
                                    after_entity_text,
                                    token.idx + 2,
                                    start_position=token.start_position
                                    + entity_offset
                                    + len(entity["text"][0]),
                                )
                                # create new token list
                                before_new_tokens = sentence["text"].tokens[
                                    : token.idx - 1
                                ]
                                after_new_tokens = sentence["text"].tokens[token.idx :]
                                for token in after_new_tokens:
                                    token.idx += 2
                                new_tokens = (
                                    before_new_tokens
                                    + [before_token, entity_token, after_token]
                                    + after_new_tokens
                                )
                                # replace token list in sentence
                                sentence["text"].tokens = new_tokens

                                # add entity
                                tokens.append(entity_token)
                                tokens_text = "".join([t.text for t in tokens])
                                entity_text = entity["text"][0].replace(" ", "")
                                assert tokens_text == entity_text
                                break

                            break

                    # if specified by add_labels, add the found entities to the sentences
                    for gold_annotation in entity["normalized"]:
                        # determine which labels to add
                        if add_labels:
                            # use the label name given as an argument or else the entity type from the bigbio dataset
                            label_type = (
                                label_name
                                if (label_name is not None)
                                else entity["type"]
                            )
                            # add only named entity recogintion labels
                            if ner_only:
                                label = SpanLabel(
                                    span=Span(tokens), value=entity["type"]
                                )
                                sentence["text"].add_complex_label(
                                    label_type + "_GOLD", label=label
                                )
                            # add named entity normalization labels
                            else:
                                ids = gold_annotation["db_id"].split("|")
                                main_id = ids[0]
                                if len(ids) > 1:
                                    other_ids = ids[1:]
                                else:
                                    other_ids = None
                                label = EntityLinkingLabel(
                                    span=Span(tokens),
                                    id=main_id,
                                    concept_name=entity["text"],
                                    additional_ids=other_ids,
                                    ontology=gold_annotation["db_name"],
                                )
                                sentence["text"].add_complex_label(
                                    typename=label_type + "_GOLD", label=label
                                )
                    # if using a ner dataset without normatilzations
                    if len(entity["normalized"]) == 0 and add_labels:
                        label_type = (
                            label_name if (label_name is not None) else entity["type"]
                        )
                        label = SpanLabel(span=Span(tokens), value=entity["type"])
                        sentence["text"].add_complex_label(
                            label_type + "_GOLD", label=label
                        )
                    break

            # if no matching sentence for the token was found
            else:
                raise Exception("Could not find position of the entity " + entity["id"])

        sentences.append([sentence["text"] for sentence in sentences_with_offsets])

    return sentences


class BigBioDataset:
    def __init__(self, dataset_name, split, use_tokenizer, schema="bigbio_kb"):
        conhelps = BigBioConfigHelpers()
        dataset = conhelps.for_config_name(dataset_name + "_" + schema).load_dataset()
        self.dataset = dataset[split]
        self.tokenizer = use_tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.schema = schema
        self.annotated_sentences = None
        self.sentences = None

    def get_dataset(self):
        return self.dataset

    def get_sentences(self):
        if self.sentences == None:
            self.sentences = bigbio_dataset_to_annotated_flair_sentences(
                self.dataset, tokenizer=self.tokenizer, add_labels=False
            )
        return self.sentences

    def get_annotated_sentences(
        self, ner_only: bool = False, use_label_name: str = None
    ):
        if use_label_name is None:
            self.annotated_sentences = bigbio_dataset_to_annotated_flair_sentences(
                self.dataset, self.tokenizer, add_labels=True, ner_only=ner_only
            )
        else:
            self.annotated_sentences = bigbio_dataset_to_annotated_flair_sentences(
                self.dataset,
                self.tokenizer,
                add_labels=True,
                ner_only=ner_only,
                label_name=use_label_name,
            )
        return self.annotated_sentences

