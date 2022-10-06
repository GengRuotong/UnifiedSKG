#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
from .summary_metric import compute_rouges


class EvaluateTool(object):
    """
    The meta evaluator
    """
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        refference_summaries = [item["seq_out"] for item in golds]
        predictions = [pred.replace(' ', '').strip() for pred in preds]
        
        rouge_scores = compute_rouges(predictions, refference_summaries)
        summary.update(rouge_scores)
        return summary
