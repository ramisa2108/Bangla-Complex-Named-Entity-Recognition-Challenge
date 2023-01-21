import json
import gzip
import itertools
from collections import defaultdict
import sys
import argparse


TOK_COL = 0
TAG_COL = 3
SET_COL = 2 

def build_args():
    """
    Build the argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--true', help='Path to ground truth labels')
    parser.add_argument('--pred', help='Path to predicted labels')
    
    return parser.parse_args()


class EvalMetric():
    def __init__(self, non_entity_labels=['O']):
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0
        self._num_predicted_mentions = 0
        self._TP, self._FP, self._GT = defaultdict(int), defaultdict(int), defaultdict(int)
        self.non_entity_labels = set(non_entity_labels)


    def __call__(self, batched_predicted_spans, batched_gold_spans, sentences=None):
        non_entity_labels = self.non_entity_labels

        for predicted_spans, gold_spans in zip(batched_predicted_spans, batched_gold_spans):
            gold_spans_set = set([x for x, y in gold_spans.items() if y not in non_entity_labels])
            pred_spans_set = set([x for x, y in predicted_spans.items() if y not in non_entity_labels])

            self._num_gold_mentions += len(gold_spans_set)
            self._num_recalled_mentions += len(gold_spans_set & pred_spans_set)
            self._num_predicted_mentions += len(pred_spans_set)

            for ky, val in gold_spans.items():
                if val not in non_entity_labels:
                    self._GT[val] += 1

            for ky, val in predicted_spans.items():
                if val in non_entity_labels:
                    continue
                if ky in gold_spans and val == gold_spans[ky]:
                    self._TP[val] += 1
                else:
                    self._FP[val] += 1

    def get_metric(self, reset=False):
        all_tags = set()
        all_tags.update(self._TP.keys())
        all_tags.update(self._FP.keys())
        all_tags.update(self._GT.keys())
        all_metrics = {}

        for tag in all_tags:
            precision, recall, f1_measure = self.compute_prf_metrics(true_positives=self._TP[tag],
                                                                     false_negatives=self._GT[tag] - self._TP[tag],
                                                                     false_positives=self._FP[tag])
            all_metrics['P@{}'.format(tag)] = precision
            all_metrics['R@{}'.format(tag)] = recall
            all_metrics['F1@{}'.format(tag)] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self.compute_prf_metrics(true_positives=sum(self._TP.values()),
                                                                 false_positives=sum(self._FP.values()),
                                                                 false_negatives=sum(self._GT.values()) - sum(self._TP.values()))
        all_metrics["Precision"] = precision
        all_metrics["Recall"] = recall
        all_metrics["F1"] = f1_measure

        if self._num_gold_mentions == 0:
            entity_recall = 0.0
        else:
            entity_recall = self._num_recalled_mentions / float(self._num_gold_mentions)

        if self._num_predicted_mentions == 0:
            entity_precision = 0.0
        else:
            entity_precision = self._num_recalled_mentions / float(self._num_predicted_mentions)

        all_metrics['MD@R'] = entity_recall
        all_metrics['MD@P'] = entity_precision
        all_metrics['MD@F1'] = 2. * ((entity_precision * entity_recall) / (entity_precision + entity_recall + 1e-13))
        all_metrics['ALLTRUE'] = self._num_gold_mentions
        all_metrics['ALLRECALLED'] = self._num_recalled_mentions
        all_metrics['ALLPRED'] = self._num_predicted_mentions

        return all_metrics

    @staticmethod
    def compute_prf_metrics(true_positives, false_positives, false_negatives):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure


def extract_spans(tags):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        indicator = nt[0]
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans


def _is_divider(line):
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-" or line.startswith('# id'):
        return True

    return False


def read_data(input_file):
    instances = []
    for fields in get_ner_reader(data=input_file):
        # _, ner_tags = fields[TOK_COL], fields[TAG_COL]
        
        if len(fields) != 1:
            sys.stderr.write("Prediction file should only contain tags. Please follow submission instructions.\n")
            exit(-1)

        ner_tags = fields[0]
        # gold_spans = extract_spans(ner_tags)
        instances.append(ner_tags)
    
    return instances


def get_ner_reader(data):
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        fields = [line.strip().split() for line in lines]
        fields = [list(field) for field in zip(*fields)]
        yield fields


def compute_metrics(predictions, ground_truth, id_info=None):

    if type(predictions) is dict:
        results = {}
        for s in predictions:
            pred = [extract_spans(p) for p in predictions[s]]
            gt = [extract_spans(g) for g in ground_truth[s]]
            metric = EvalMetric()
            metric(pred, gt)
            results[s] = metric.get_metric() 
        return results

    else:

        # checking # of sentnences and # of tokens
        if len(predictions) != len(ground_truth):
            sys.stderr.write('Expected # of sentences: {}, received # of sentences: {}!\n'.format(len(predictions), len(ground_truth)))
            sys.stderr.write('Please check your prediction file!\n')
            exit(-1)

        if id_info:
            error_msg = ''
            for id_, pred, gt in zip(id_info, predictions, ground_truth):
                if len(pred) != len(gt):
                    error_msg += 'For sentence {}, expected # of tokens: {}, received # of tokens {}.\n'.format(id_, len(gt), len(pred))
            
            if error_msg:
                error_msg += "Please check the aboved mentioned sentences!\n"
                sys.stderr.write(error_msg)
                exit(-1)

        predictions = [extract_spans(p) for p in predictions]
        ground_truth = [extract_spans(g) for g in ground_truth]
        metric = EvalMetric()
        metric(predictions, ground_truth)
        return metric.get_metric()


if __name__ == '__main__':
    """
    Sample run
    
    
    python eval_main.py --true dev_gt_labels.txt --pred dev_pred_labels.txt
    """
    args = build_args()
    
#    pred_file = 'dev_gt_labels.txt'
#    gt_file = 'dev_pred_labels.txt'
    
    gt_file = args.true
    pred_file = args.pred

    metrics = compute_metrics(read_data(pred_file), read_data(gt_file))
    print(json.dumps(metrics, indent=2))
    
    print(f"Precision {round(metrics['Precision'], 2)}, Recall {round(metrics['Recall'], 2)}, F1 {round(metrics['F1'], 2)}")

