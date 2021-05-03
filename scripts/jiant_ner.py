"""Named entity labeling Edge Probing task.
Task source paper: https://arxiv.org/pdf/1905.06316.pdf.
Task data prep directions: https://github.com/nyu-mll/jiant/blob/master/probing/data/README.md.
"""
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import edge_probing_single_span
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(edge_probing_single_span.Example):
    @property
    def task(self):
        return NerTask


@dataclass
class TokenizedExample(edge_probing_single_span.TokenizedExample):
    pass


@dataclass
class DataRow(edge_probing_single_span.DataRow):
    pass


@dataclass
class Batch(edge_probing_single_span.Batch):
    pass


class NerTask(edge_probing_single_span.AbstractProbingTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    LABELS = ['B-ACTOR',
            'B-CHARACTER',
            'B-DIRECTOR',
            'B-GENRE',
            'B-PLOT',
            'B-RATING',
            'B-RATINGS_AVERAGE',
            'B-REVIEW',
            'B-SONG',
            'B-TITLE',
            'B-TRAILER',
            'B-YEAR',
            'I-ACTOR',
            'I-CHARACTER',
            'I-DIRECTOR',
            'I-GENRE',
            'I-PLOT',
            'I-RATING',
            'I-RATINGS_AVERAGE',
            'I-REVIEW',
            'I-SONG',
            'I-TITLE',
            'I-TRAILER',
            'I-YEAR',
            'O']
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 1

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (line_num, line) in enumerate(lines):
            for (target_num, target) in enumerate(line["targets"]):
                span = target["span1"]
                examples.append(
                    Example(
                        guid="%s-%s-%s" % (set_type, line_num, target_num),
                        text=line["text"],
                        span=span,
                        labels=[target["label"]] if set_type != "test" else [cls.LABELS[-1]],
                    )
                )
        return examples