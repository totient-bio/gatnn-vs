import numpy as np
from rdkit.ML.Scoring import Scoring


def metrics_for_target(pred, actual, mask):
    mask = np.array(mask, dtype=np.bool)
    masked_preds = pred.squeeze()[mask]
    order = np.flipud(np.argsort(masked_preds))
    masked_oredered_actual = actual[mask][order]
    return Scoring.CalcEnrichment(
        masked_oredered_actual, 0, [.001, .005, .01, .05]
    ) + [
        Scoring.CalcAUC(masked_oredered_actual, 0),
        Scoring.CalcBEDROC(masked_oredered_actual, 0, 20)
    ]


def get_metrics(predictions, actual, mask, prefix=''):
    num_classes = predictions.shape[1]
    keys = ['ef01', 'ef05', 'ef1', 'ef5', 'auc', 'bedroc20']

    individual_preds = np.split(predictions, num_classes, axis=1)
    individual_actual = np.split(actual, num_classes, axis=1)
    individual_mask = np.split(mask, num_classes, axis=1)
    individual_scores = np.array([
        metrics_for_target(pred, lbl, msk.squeeze())
        for pred, lbl, msk in zip(individual_preds, individual_actual, individual_mask)
    ])
#     target_ord = np.argsort(individual_scores, axis=0)
#     percentiles = {
#         'min_': target_ord[0],
#         '5_': target_ord[int(0.05 * num_classes)],
#         '25_': target_ord[int(0.25 * num_classes)],
#         '50_': target_ord[int(0.5 * num_classes)],
#         '75_': target_ord[int(0.75 * num_classes)],
#         '95_': target_ord[int(0.95 * num_classes)],
#         'max_': target_ord[num_classes - 1]
#     }

# #     histograms = {prefix + 'ef_hist': individual_scores[:, :].flatten()}
#     metrics = {
#         prefix + perc + k: individual_scores[targets[i]][i]
#         for perc, targets in percentiles.items()
#         for i, k in enumerate(keys)
#     }
    means = np.mean(individual_scores, axis=0)
    metrics = {
        prefix + 'mean_' + k: means[i] for i, k in enumerate(keys)
    }
    return metrics, {}
