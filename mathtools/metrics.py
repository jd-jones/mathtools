import logging

try:
    import torch
except ImportError:
    pass
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from . import utils


logger = logging.getLogger(__name__)


def oov_rate(state_seq, state_vocab):
    state_is_oov = ~np.array([s in state_vocab for s in state_seq], dtype=bool)
    prop_state_oov = state_is_oov.sum() / state_is_oov.size
    return prop_state_oov


def confusionMatrix(all_pred_seqs, all_true_seqs, vocab_size):
    """
    Returns
    -------
    confusions: np.ndarray of int, shape (vocab_size, vocab_size)
        Rows represent predicted labels; columns represent true labels.
    """

    confusions = np.zeros((vocab_size, vocab_size), dtype=int)

    for pred_seq, true_seq in zip(all_pred_seqs, all_true_seqs):
        for i_pred, i_true in zip(pred_seq, true_seq):
            confusions[i_pred, i_true] += 1

    return confusions


def scoreConfusionMatrix(all_score_seqs, all_true_seqs, vocab_size):
    """
    Returns
    -------
    confusions: np.ndarray of int, shape (vocab_size, vocab_size)
        Rows represent predicted labels; columns represent true labels.
    """

    confusions = np.full((vocab_size, vocab_size), -np.inf, dtype=float)

    for score_seq, true_seq in zip(all_score_seqs, all_true_seqs):
        for score_row, i_true in zip(score_seq, true_seq):
            for i_pred, score in enumerate(score_row):
                confusions[i_pred, i_true] = np.logaddexp(confusions[i_pred, i_true], score)

    return confusions


def perClassAcc(confusions, return_counts=False):
    class_counts = confusions.sum(axis=0)
    per_class_acc = np.diag(confusions) / class_counts
    if return_counts:
        return per_class_acc, class_counts
    return per_class_acc


def plotConfusions(fn, confusions, vocab, size=24, disp_counts=False):
    plt.figure(figsize=(size, size))

    # vmax = np.abs(confusions).max()
    # plt.matshow(confusions, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    plt.matshow(confusions)

    if disp_counts:
        for i_row, row in enumerate(confusions):
            for i_col, val in enumerate(row):
                if not val:
                    continue
                plt.text(
                    i_col, i_row, val,
                    fontsize=8, color='black', ha='center', va='center'
                )

    plt.xticks(ticks=range(len(vocab)), labels=vocab, rotation='vertical')
    plt.yticks(ticks=range(len(vocab)), labels=vocab)
    plt.colorbar()

    plt.savefig(fn, bbox_inches='tight')
    plt.close()


def plotPerClassAcc(fn, vocab, per_class_acc, class_preds, class_counts):
    macro_acc = per_class_acc.mean()

    f, axes = plt.subplots(3, figsize=(12, 6), sharex=True)
    axes[0].set_title(f"Macro Accuracy: {macro_acc * 100:.2f}%")
    axes[0].bar(range(len(vocab)), per_class_acc)
    for index, val in enumerate(per_class_acc):
        eps = np.sign(val) * 0.01
        axes[0].text(
            x=index, y=val + eps, s=f"{val * 100:.0f}",
            fontdict=dict(fontsize=8), va='center'
        )
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(range(len(vocab)), class_preds / class_counts.sum())
    for index, val in enumerate(class_preds / class_counts.sum()):
        eps = np.sign(val) * 0.01
        axes[1].text(
            x=index, y=val + eps, s=f"{val * 100:.0f}",
            fontdict=dict(fontsize=8), va='center'
        )
    axes[1].set_ylabel("Pred Frequency")

    axes[2].bar(range(len(vocab)), class_counts / class_counts.sum())
    for index, val in enumerate(class_counts / class_counts.sum()):
        eps = np.sign(val) * 0.01
        axes[2].text(
            x=index, y=val + eps, s=f"{val * 100:.0f}",
            fontdict=dict(fontsize=8), va='center'
        )
    axes[2].set_xticks(range(len(vocab)))
    axes[2].set_xticklabels(vocab, rotation='vertical')
    axes[2].set_ylabel("True Frequency")

    plt.savefig(fn, bbox_inches='tight')


def makeMetric(name):
    if name == 'Reciprocal Loss':
        return ReciprocalAverageLoss()
    elif name == 'Loss':
        return AverageLoss()
    elif name == 'Accuracy':
        return Accuracy()
    elif name == 'Precision':
        return Precision()
    elif name == 'Recall':
        return Recall()
    elif name == 'F1':
        return Fmeasure(beta=1)
    else:
        raise AssertionError()


def accuracy_upto(pred_seq, gt_seq, equivalence=None):
    if equivalence is None:
        def equivalence(x, y):
            return x == y

    is_eq = np.array(
        [equivalence(p, gt) for p, gt in zip(pred_seq, gt_seq)],
        dtype=bool
    )

    accuracy = is_eq.sum() / len(is_eq)
    return accuracy


class RationalPerformanceMetric():
    """
    Performance metric with a numerator and a denominator.
    """

    def __init__(self):
        self.initializeCounts()

    def initializeCounts(self):
        self._numerator = 0
        self._denominator = 0

    def evaluate(self):
        return utils.safeDivide(self._numerator, self._denominator)

    @property
    def value(self):
        return float(self.evaluate())

    def __str__(self):
        return f'{self.evaluate():.5f}'

    def accumulate(self, outputs=None, labels=None, loss=None):
        self._numerator += self._count_numerator(outputs, labels, loss)
        self._denominator += self._count_denominator(outputs, labels, loss)

    def _count_numerator(self, outputs=None, labels=None, loss=None):
        raise NotImplementedError()

    def _count_denominator(self, outputs=None, labels=None, loss=None):
        raise NotImplementedError()


class AverageLoss(RationalPerformanceMetric):
    def _count_numerator(self, outputs=None, labels=None, loss=None):
        return loss

    def _count_denominator(self, outputs=None, labels=None, loss=None):
        return 1

    @property
    def name(self):
        return 'loss'

    def __str__(self):
        return self.name + ': ' + super().__str__()


class ReciprocalAverageLoss(AverageLoss):
    def evaluate(self):
        return 1 / super().evaluate()

    @property
    def name(self):
        return 'reciprocal loss'


class ConfusionPerformanceMetric(RationalPerformanceMetric):
    def __init__(self):
        self.initializeCounts()

    def initializeCounts(self):
        self._true_positives = 0
        self._true_negatives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def __str__(self):
        return f'{self.evaluate() * 100:5.2f}%'

    def accumulate(self, predicted=None, true=None, loss=None):
        self._accumulate_confusions(predicted, true)

    def _accumulate_confusions(self, predicted, true):
        self._true_positives += truePositives(predicted, true)
        self._true_negatives += trueNegatives(predicted, true)
        self._false_positives += falsePositives(predicted, true)
        self._false_negatives += falseNegatives(predicted, true)

    @property
    def _numerator(self):
        raise NotImplementedError()

    @property
    def _denominator(self):
        raise NotImplementedError()


class Fmeasure(ConfusionPerformanceMetric):
    def __init__(self, beta=1):
        super().__init__()
        self._beta = beta

    @property
    def _numerator(self):
        return (1 + self._beta ** 2) * self._true_positives

    @property
    def _denominator(self):
        denom = (
            self._false_positives
            + (self._beta ** 2) * self._false_negatives
            + (1 + self._beta ** 2) * self._true_positives
        )
        return denom

    @property
    def name(self):
        return f'F_{self._beta}'

    def __str__(self):
        return self.name + ': ' + super().__str__()


class Recall(ConfusionPerformanceMetric):
    @property
    def _numerator(self):
        return self._true_positives

    @property
    def _denominator(self):
        return self._true_positives + self._false_negatives

    @property
    def name(self):
        return 'rec'

    def __str__(self):
        return self.name + ': ' + super().__str__()


class Precision(ConfusionPerformanceMetric):
    @property
    def _numerator(self):
        return self._true_positives

    @property
    def _denominator(self):
        return self._true_positives + self._false_positives

    @property
    def name(self):
        return 'prc'

    def __str__(self):
        return self.name + ': ' + super().__str__()


class Accuracy(ConfusionPerformanceMetric):
    @property
    def _numerator(self):
        return self._true_positives + self._true_negatives

    @property
    def _denominator(self):
        data_positives = self._true_positives + self._false_negatives
        data_negatives = self._false_positives + self._true_negatives
        return data_positives + data_negatives

    @property
    def name(self):
        return 'acc'

    def __str__(self):
        return self.name + ': ' + super().__str__()


def truePositives(predicted, true, background_index=0):
    if isinstance(predicted, torch.Tensor) and isinstance(true, torch.Tensor):
        pred_is_positive = predicted != background_index
        true_positives = predicted[pred_is_positive] == true[pred_is_positive]
        return true_positives.sum().float()
    if isinstance(predicted, np.ndarray) and isinstance(true, np.ndarray):
        pred_is_positive = predicted != background_index
        true_positives = predicted[pred_is_positive] == true[pred_is_positive]
        return true_positives.sum().astype(float)
    try:
        return sum(p == t for p, t in zip(predicted, true) if t != background_index)
    except TypeError:
        return int(predicted == true) if true != background_index else 0


def trueNegatives(predicted, true, background_index=0):
    if isinstance(predicted, torch.Tensor) and isinstance(true, torch.Tensor):
        pred_is_negative = predicted == background_index
        true_negatives = predicted[pred_is_negative] == true[pred_is_negative]
        return true_negatives.sum().float()
    if isinstance(predicted, np.ndarray) and isinstance(true, np.ndarray):
        pred_is_negative = predicted == background_index
        true_negatives = predicted[pred_is_negative] == true[pred_is_negative]
        return true_negatives.sum().astype(float)
    try:
        return sum(p == t for p, t in zip(predicted, true) if t == background_index)
    except TypeError:
        return int(predicted == true) if true == background_index else 0


def falsePositives(predicted, true, background_index=0):
    if isinstance(predicted, torch.Tensor) and isinstance(true, torch.Tensor):
        pred_is_positive = predicted != background_index
        false_positives = predicted[pred_is_positive] != true[pred_is_positive]
        return false_positives.sum().float()
    if isinstance(predicted, np.ndarray) and isinstance(true, np.ndarray):
        pred_is_positive = predicted != background_index
        false_positives = predicted[pred_is_positive] != true[pred_is_positive]
        return false_positives.sum().astype(float)
    try:
        return sum(p != t for p, t in zip(predicted, true) if t)
    except TypeError:
        return int(predicted != true) if true else 0


def falseNegatives(predicted, true, background_index=0):
    if isinstance(predicted, torch.Tensor) and isinstance(true, torch.Tensor):
        pred_is_negative = predicted == background_index
        false_negatives = predicted[pred_is_negative] != true[pred_is_negative]
        return false_negatives.sum().float()
    if isinstance(predicted, np.ndarray) and isinstance(true, np.ndarray):
        pred_is_negative = predicted == background_index
        false_negatives = predicted[pred_is_negative] != true[pred_is_negative]
        return false_negatives.sum().astype(float)
    try:
        return sum(p != t for p, t in zip(predicted, true) if t == background_index)
    except TypeError:
        return int(predicted != true) if true == background_index else 0


def classAccuracy(true_label_seqs, predicted_label_seqs):
    avg_accuracies = utils.iterate(
        seqClassAccuracy,
        true_label_seqs,
        predicted_label_seqs
    )
    true_label_seq_lens = [len(s) for s in true_label_seqs]

    avg_accuracies = np.array(avg_accuracies)
    true_label_seq_lens = np.array(true_label_seq_lens)

    num_classes = utils.numClasses([label for ls in true_label_seqs for label in ls])

    avg_accy = np.average(avg_accuracies, weights=true_label_seq_lens)
    chance = 1 / num_classes

    return avg_accy, chance


def seqClassAccuracy(true_label_seq, predicted_label_seq):
    num_correct = 0
    for true, predicted in zip(true_label_seq, predicted_label_seq):
        num_correct += int((true == predicted).all())
    total = len(true_label_seq)

    return num_correct / total


def avgAccuracy(true_label_seqs, predicted_label_seqs):
    true_labels = utils.toFlattenedArray(true_label_seqs)
    predicted_labels = utils.toFlattenedArray(predicted_label_seqs)

    metric = metrics.accuracy_score(true_labels.ravel(), predicted_labels.ravel())
    chance = 0.5

    return metric, chance


def avgPrecision(true_label_seqs, predicted_label_seqs):
    true_labels = utils.toFlattenedArray(true_label_seqs)
    predicted_labels = utils.toFlattenedArray(predicted_label_seqs)

    metric = metrics.precision_score(true_labels.ravel(), predicted_labels.ravel())
    chance = true_labels.ravel().sum() / true_labels.size

    return metric, chance


def avgRecall(true_label_seqs, predicted_label_seqs):
    true_labels = utils.toFlattenedArray(true_label_seqs)
    predicted_labels = utils.toFlattenedArray(predicted_label_seqs)

    metric = metrics.recall_score(true_labels.ravel(), predicted_labels.ravel())
    chance = 0.5

    return metric, chance


def edgeDistance(true_label_seqs, predicted_label_seqs):
    avg_distances = utils.iterate(seqEdgeDistance, true_label_seqs, predicted_label_seqs)
    true_label_seq_lens = [len(s) for s in true_label_seqs]

    avg_distances = np.array(avg_distances)
    true_label_seq_lens = np.array(true_label_seq_lens)

    metric = np.average(avg_distances, weights=true_label_seq_lens)
    chance = -1

    return metric, chance


def seqEdgeDistance(true_label_seq, predicted_label_seq):
    sum_dist = 0
    for true, predicted in zip(true_label_seq, predicted_label_seq):
        sum_dist += (true != predicted).sum()
    total = len(true_label_seq)

    return sum_dist / total


def nonempty(assembly_state):
    return assembly_state.any()


def countTrue(true, pred, precision='state', denom_mode='accuracy'):
    if precision == 'block':
        p_blocks = set(pred.blocks.keys())
        t_blocks = set(true.blocks.keys())
        num_true = len(p_blocks & t_blocks)
        if denom_mode == 'accuracy':
            num_total = len(p_blocks | t_blocks)
        elif denom_mode == 'precision':
            num_total = len(p_blocks)
        elif denom_mode == 'recall':
            num_total = len(t_blocks)
    elif precision == 'edge':
        p_edges = pred.connections
        t_edges = true.connections
        num_true = np.sum(p_edges & t_edges)
        if denom_mode == 'accuracy':
            num_total = np.sum(p_edges | t_edges)
        elif denom_mode == 'precision':
            num_total = np.sum(p_edges)
        elif denom_mode == 'recall':
            num_total = np.sum(t_edges)
    elif precision == 'state':
        num_true = int(pred == true)
        if denom_mode == 'accuracy':
            num_total = 1
        elif denom_mode == 'precision':
            num_total = int(nonempty(pred))
        elif denom_mode == 'recall':
            num_total = int(nonempty(true))
        # Don't count empty states in precision/recall mode
        num_true *= num_total
    elif precision == 'topology':
        num_true = int((pred.connections == true.connections).all())
        if denom_mode == 'accuracy':
            num_total = 1
        elif denom_mode == 'precision':
            num_total = int(nonempty(pred))
        elif denom_mode == 'recall':
            num_total = int(nonempty(true))
        # Don't count empty states in precision/recall mode
        num_true *= num_total
    elif precision == 'subset_topo':
        pred_minus_true = pred.connections * ~true.connections
        true_minus_pred = true.connections * ~pred.connections
        pred_subset_true = true_minus_pred.any() and not pred_minus_true.any()
        num_true = int(pred_subset_true)
        if not pred.any() and true.any():
            num_true = 0
        if (pred.connections == true.connections).all():
            num_true = 1
        if denom_mode == 'accuracy':
            num_total = 1
        elif denom_mode == 'precision':
            num_total = int(nonempty(pred))
        elif denom_mode == 'recall':
            num_total = int(nonempty(true))
        # don't count empty states in precision/recall mode
        num_true *= num_total
    elif precision == 'subset_geom':
        num_true = int(pred <= true)
        if not pred.any() and true.any():
            num_true = 0
        if denom_mode == 'accuracy':
            num_total = 1
        elif denom_mode == 'precision':
            num_total = int(nonempty(pred))
        elif denom_mode == 'recall':
            num_total = int(nonempty(true))
        # don't count empty states in precision/recall mode
        num_true *= num_total
    elif precision == 'off by one':
        differences = (pred.symmetrized_connections ^ true.symmetrized_connections).astype(int)
        # since arrays are symmetric, any difference results in at least two changed edges,
        # so divide by two. Comparison is <= 2 because we're allowing one EDIT
        # (i.e. change one edge into another)
        num_true = int(differences.sum() / 2 <= 2)
        if denom_mode == 'accuracy':
            num_total = 1
        elif denom_mode == 'precision':
            num_total = int(nonempty(pred))
        elif denom_mode == 'recall':
            num_total = int(nonempty(true))
        # don't count empty states in precision/recall mode
        num_true *= num_total

    return num_true, num_total


def countSeq(true_seq, pred_seq, precision='states', denom_mode='accuracy'):
    len_true = len(true_seq)
    len_pred = len(pred_seq)
    if len_true != len_pred:
        err_str = f'{len_true} samples in true_seq'
        err_str += f' != {len_pred} samples in pred_seq'
        raise ValueError(err_str)

    num_correct = 0
    num_total = 0
    for true, pred in zip(true_seq, pred_seq):
        cur_correct, cur_total = countTrue(true, pred, precision=precision, denom_mode=denom_mode)
        num_correct += cur_correct
        num_total += cur_total

    return num_correct, num_total


def numberCorrect(
        true_seq, predicted_seq,
        ignore_empty_true=False, ignore_empty_pred=False, precision='states'):

    len_true = len(true_seq)
    len_pred = len(predicted_seq)
    if len_true != len_pred:
        err_str = f'{len_true} samples in true_seq != {len_pred} samples in predicted_seq'
        raise ValueError(err_str)

    if ignore_empty_true:
        indices = tuple(i for i, s in enumerate(true_seq) if nonempty(s))
        predicted_seq = tuple(predicted_seq[i] for i in indices)
        true_seq = tuple(true_seq[i] for i in indices)
        if not len(true_seq):
            warn_str = 'All ground-truth sequences were empty!'
            logger.warning(warn_str)

    if ignore_empty_pred:
        indices = tuple(i for i, s in enumerate(predicted_seq) if nonempty(s))
        predicted_seq = tuple(predicted_seq[i] for i in indices)
        true_seq = tuple(true_seq[i] for i in indices)

    num_correct = 0
    num_total = 0
    for p, t in zip(predicted_seq, true_seq):
        if precision == 'states':
            num_correct += int(p == t)
            num_total += 1
        elif precision == 'edges':
            num_correct += int(np.all(p.connections == t.connections))
            num_total += 1
        elif precision == 'vertices':
            num_correct += int(p.blocks == t.blocks)
            num_total += 1
        elif precision == 'structure':
            if not (nonempty(p) and nonempty(t)):
                num_correct += int(p == t)
            else:
                num_correct += int(p < t or p > t or p == t)
            num_total += 1
        elif precision == 'blocks':
            p_blocks = set(p.blocks.keys())
            t_blocks = set(t.blocks.keys())
            num_correct += len(p_blocks & t_blocks)
            num_total += len(p_blocks | t_blocks)
        elif precision == 'blocks_recall':
            p_blocks = set(p.blocks.keys())
            t_blocks = set(t.blocks.keys())
            num_correct += len(p_blocks & t_blocks)
            num_total += len(t_blocks)
        elif precision == 'blocks_precision':
            p_blocks = set(p.blocks.keys())
            t_blocks = set(t.blocks.keys())
            num_correct += len(p_blocks & t_blocks)
            num_total += len(p_blocks)
        elif precision == 'avg_edge':
            p_edges = p.connections
            t_edges = t.connections
            num_correct += np.sum(p_edges & t_edges)
            num_total += np.sum(p_edges | t_edges)
        elif precision == 'avg_edge_precision':
            p_edges = p.connections
            t_edges = t.connections
            num_correct += np.sum(p_edges & t_edges)
            num_total += np.sum(p_edges)
        elif precision == 'avg_edge_recall':
            p_edges = p.connections
            t_edges = t.connections
            num_correct += np.sum(p_edges & t_edges)
            num_total += np.sum(t_edges)

    return num_correct, num_total


def stateOverlap(true_seq, predicted_seq, ignore_empty=False):
    len_true = len(true_seq)
    len_pred = len(predicted_seq)
    if len_true != len_pred:
        err_str = f'{len_true} samples in true_seq != {len_pred} samples in predicted_seq'
        # raise ValueError(err_str)
        logger.warn(err_str)

    if ignore_empty:
        predicted_seq = tuple(filter(nonempty, predicted_seq))
        true_seq = tuple(filter(nonempty, true_seq))

    size_intersect = 0
    size_union = 0

    for p, t in zip(predicted_seq, true_seq):
        p_blocks = set(p.blocks.keys())
        t_blocks = set(t.blocks.keys())
        size_intersect += len(p_blocks & t_blocks)
        size_union += len(p_blocks | t_blocks)

    return size_intersect, size_union


def vertexOverlap(state1, state2):
    pass


def edgeOverlap(state1, state2):
    pass


def levenshtein(
        reference, candidate, normalized=False, segment_level=False,
        # reduce_reference=True, reduce_candidate=False,
        deletion_cost=1, insertion_cost=1, substitution_cost=1,
        return_num_elems=False, corpus=None, resolution=None):
    """ Compute the Levenshtein (edit) distance between two sequences.

    Parameters
    ----------
    reference : iterable(object)
    candidate : iterable(object)
    normalized : bool, optional
    reduce_true : bool, optional
    deletion_cost : int, optional
        Cost of deleting an element from the `candidate`.
    insertion_cost : int, optional
        Cost of inserting an element from the `reference`.
    substitution_cost : int, optional
        Cost of substituting an element in the `reference` for an element in
        the `candidate`.

    Returns
    -------
    dist : int
        NOTE: `dist` has type `float` if `normalized == True`
    """

    if corpus is None:
        def size(state, resolution):
            return 1

        def difference(state, other, resolution):
            return int(state != other)
    elif corpus == 'airplane':
        def size(state, resolution):
            if resolution == 'state':
                return 1
            elif resolution == 'block':
                return len(state.assembly_state)
            raise NotImplementedError

        def difference(state, other, resolution):
            if resolution == 'state':
                return int(state != other)
            elif resolution == 'block':
                return len(state.assembly_state ^ other.assembly_state)
            raise NotImplementedError
    elif corpus in ('easy', 'child'):
        def size(state, resolution):
            if resolution == 'state':
                return 1
            elif resolution == 'edge':
                return state.connections.sum()
            elif resolution == 'block':
                state_blocks = set(state.blocks.keys())
                return len(state_blocks)
            raise NotImplementedError

        def difference(state, other, resolution):
            if resolution == 'state':
                return int(state != other)
            elif resolution == 'edge':
                edge_diff = state.connections ^ other.connections
                return edge_diff.sum()
            elif resolution == 'block':
                state_blocks = set(state.blocks.keys())
                other_blocks = set(other.blocks.keys())
                return len(state_blocks ^ other_blocks)
            raise NotImplementedError

    if segment_level:
        reference, _ = utils.computeSegments(reference)
        candidate, _ = utils.computeSegments(candidate)

    num_true = 1 if not reference else len(reference)
    num_pred = 1 if not candidate else len(candidate)

    prefix_dists = np.zeros((num_pred, num_true), dtype=int)

    # Cost for deleting all elements of candidate
    for i in range(1, num_pred):
        candidate_size = size(candidate[i], resolution=resolution)
        prefix_dists[i, 0] = prefix_dists[i - 1, 0] + deletion_cost * candidate_size

    # Cost for inserting all elements of reference
    for j in range(1, num_true):
        reference_size = size(reference[j], resolution=resolution)
        prefix_dists[0, j] = prefix_dists[0, j - 1] + insertion_cost * reference_size

    for i in range(1, num_pred):
        for j in range(1, num_true):
            # needs_sub = int(reference[i] != candidate[j])
            candidate_size = size(candidate[i], resolution=resolution)
            reference_size = size(reference[j], resolution=resolution)
            sub_size = difference(candidate[i], reference[j], resolution=resolution)

            prefix_dists[i, j] = min(
                prefix_dists[i - 1, j] + deletion_cost * candidate_size,
                prefix_dists[i, j - 1] + insertion_cost * reference_size,
                prefix_dists[i - 1, j - 1] + substitution_cost * sub_size,
            )

    dist = prefix_dists[num_pred - 1, num_true - 1]

    if normalized:
        size_pred = sum(size(state, resolution) for state in candidate)
        size_true = sum(size(state, resolution) for state in reference)
        dist /= max(size_pred, size_true)

    if return_num_elems:
        return dist, (num_true, num_pred)

    return dist


def avgLevenshtein(true_seqs, predicted_seqs, normalized=False):
    num_true_seqs = len(true_seqs)
    num_pred_seqs = len(predicted_seqs)
    if num_true_seqs != num_pred_seqs:
        err_str = f'{num_true_seqs} ground-truth sequences but {num_pred_seqs} prediction sequences'
        raise ValueError(err_str)

    dist = 0
    for t, p in zip(true_seqs, predicted_seqs):
        dist += levenshtein(t, p, normalized=normalized)

    return dist / num_true_seqs


def blockAccuracy(true_seqs, predicted_seqs, ignore_empty=False):
    total_intersect = 0
    total_union = 0

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        size_intersect, size_union = stateOverlap(p_seq, t_seq, ignore_empty=ignore_empty)
        total_intersect += size_intersect
        total_union += size_union

    return total_intersect / total_union, -1


def stateAccuracy(true_seqs, predicted_seqs, precision='states'):
    total_states = 0
    total_correct = 0

    num_true_seqs = len(true_seqs)
    num_pred_seqs = len(predicted_seqs)
    if num_true_seqs != num_pred_seqs:
        err_str = f'{num_true_seqs} ground-truth sequences but {num_pred_seqs} prediction sequences'
        raise ValueError(err_str)

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        num_correct, num_states = numberCorrect(p_seq, t_seq, precision=precision)
        total_correct += num_correct
        total_states += num_states

    return total_correct / total_states, -1


def statePrecision(true_seqs, predicted_seqs, precision='states'):
    total_states = 0
    total_correct = 0

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        num_correct, num_states = numberCorrect(
            t_seq, p_seq, ignore_empty_pred=True, precision=precision
        )
        total_correct += num_correct
        total_states += num_states

    if total_states:
        return total_correct / total_states, -1
    if total_correct:
        return np.inf, -1
    return np.nan, -1


def stateRecall(true_seqs, predicted_seqs, precision='states'):
    total_states = 0
    total_correct = 0

    for p_seq, t_seq in zip(predicted_seqs, true_seqs):
        num_correct, num_states = numberCorrect(
            t_seq, p_seq, ignore_empty_true=True, precision=precision
        )
        total_correct += num_correct
        total_states += num_states

    if total_states:
        return total_correct / total_states, -1
    if total_correct:
        return np.inf, -1
    return np.nan, -1
