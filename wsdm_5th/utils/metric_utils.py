# reference: https://www.kaggle.com/code/abdullahmeda/eedi-map-k-metric

import numpy as np


def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.

    This function computes the mean average precision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """

    return round(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]), 4)


def _compute_metrics(true_ids, pred_ids, debug=False):
    """
    fbeta score for one example
    """

    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # calculate the confusion matrix variables
    tp = len(true_ids.intersection(pred_ids))
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    # metrics
    f1 = tp / (tp + 0.5 * fp + 0.5 * fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if debug:
        print("Ground truth count:", len(true_ids))
        print("Predicted count:", len(pred_ids))
        print("True positives:", tp)
        print("False positives:", fp)
        print("F2:", f2)

    to_return = {
        "f1": f1,
        "f2": f2,
        "precision": precision,
        "recall": recall,
    }

    return to_return


def compute_retrieval_metrics(true_ids, pred_ids):
    """
    fbeta metric for learning equality - content recommendation task

    :param true_ids: ground truth content ids
    :type true_ids: List[List[str]]
    :param pred_ids: prediction content ids
    :type pred_ids: List[List[str]]
    """
    assert len(true_ids) == len(pred_ids), "length mismatch between truths and predictions"
    n_examples = len(true_ids)
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(n_examples):
        ex_true_ids = true_ids[i]
        ex_pred_ids = pred_ids[i]
        if len(ex_pred_ids) == 0:
            f1 = 0.0
            f2 = 0.0
            precision = 0.0
            recall = 0.0
        else:
            m = _compute_metrics(ex_true_ids, ex_pred_ids)
            f1 = m["f1"]
            f2 = m["f2"]
            precision = m["precision"]
            recall = m["recall"]

        f1_scores.append(f1)
        f2_scores.append(f2)
        precision_scores.append(precision)
        recall_scores.append(recall)

    to_return = {
        "f1_score": np.mean(f1_scores),
        "f2_score": np.mean(f2_scores),
        "precision_score": np.mean(precision_scores),
        "recall_score": np.mean(recall_scores),
    }

    return to_return
