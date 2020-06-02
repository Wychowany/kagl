import pandas
import os
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
import sys


class Data:
    @classmethod
    def read_train_data_set(cls, sentiment):
        train_set = pandas.read_csv("input/train.csv")
        train_data_by_sentiment = []
        for (_, example) in train_set.iterrows():
            if example.sentiment == sentiment:
                transformed_example = cls._transform_single_example_to_required_format(example)
                train_data_by_sentiment.append(transformed_example)
        return train_data_by_sentiment

    @classmethod
    def _transform_single_example_to_required_format(cls, example):
        start = example.text.find(example.selected_text)
        end = start + len(example.selected_text)
        return example.text, {"entities": [[start, end, 'selected_text']]}


class Models:
    @classmethod
    def train_positive(cls):
        train_data = Data.read_train_data_set("positive")
        print("Start traning positive model")
        cls.train(train_data, 'models/model_pos', n_iter=5)

    @classmethod
    def train_negative(cls):
        train_data = Data.read_train_data_set("negative")
        print("Start traning negative model")
        cls.train(train_data, "models/model_neg", n_iter=5)

    @classmethod
    def train(cls, train_data, output_dir, n_iter=20):

        nlp = spacy.blank("en")
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

        for _ in range(len(train_data)):
            ner.add_label('selected_text')

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            nlp.begin_training()
            for i in range(n_iter):
                print("Iteration: " + str(i + 1) + "/" + str(n_iter))
                random.shuffle(train_data)
                batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, drop=0.5, losses=losses)

                print("Losses", losses)
        cls.save_model(output_dir, nlp, 'st_ner')

    @classmethod
    def save_model(cls, output_dir, nlp, new_model_name):
        output_dir = f'../working/{output_dir}'
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            nlp.meta["name"] = new_model_name
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)


class Predictor:

    @classmethod
    def predict_entities(cls, text, model):
        doc = model(text)
        entities = []
        for entity in doc.ents:
            start = text.find(entity.text)
            end = start + len(entity.text)
            ent = [start, end, entity.label_]
            if ent not in entities:
                entities.append([start, end, entity.label_])

        selected_text = text
        if len(entities) > 0:
            selected_text = text[entities[0][0]: entities[0][1]]
        return selected_text


if len(sys.argv) > 1 and sys.argv[1] == "train":
    print("START TRAINING MODELS")
    Models.train_negative()
    Models.train_positive()

selected_texts = []
test_set = pandas.read_csv('input/test.csv')
df_submission = pandas.read_csv('input/sample_submission.csv')

model_pos = spacy.load('../working/models/model_pos')
model_neg = spacy.load('../working/models/model_neg')

for _, example in test_set.iterrows():
    if example.sentiment == 'neutral' or len(example.text.split()) <= 2:
        selected_texts.append(example.text)
    elif example.sentiment == 'positive':
        selected_texts.append(Predictor.predict_entities(example.text, model_pos))
    else:
        selected_texts.append(Predictor.predict_entities(example.text, model_neg))

test_set['selected_text'] = selected_texts
df_submission['selected_text'] = test_set['selected_text']
df_submission.to_csv("submission.csv", index=False)

print("SUBMISSION FILE GENERATED")
