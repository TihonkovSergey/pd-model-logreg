import numpy as np
import matplotlib.pyplot as plt

from definitions import THRESHOLD_TUNING_N_SAMPLES
from definitions import THRESHOLD_TUNING_N_ITERS
from definitions import REPORT_DIR
from definitions import TARGET_NAME
from definitions import LOGGER
from definitions import SEED


def _get_optimal_threshold_by_accuracy(probs, labels):
    assert len(probs) == len(labels)
    n = len(probs)
    n_positive = sum(labels)

    best_accuracy = 0.0
    best_threshold = None

    counter_pos = 0
    counter_neg = 0

    acc_list = []
    prob_list = []
    for prob, label in sorted((p, l) for p, l in zip(probs, labels)):
        if label:
            counter_pos += 1
        else:
            counter_neg += 1
        acc = (counter_neg + (n_positive - counter_pos)) / n
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = prob
        # print(acc, f'({counter_neg} + ({n} - {counter_pos})/{n}')
        acc_list.append(acc)
        prob_list.append(prob)
    return best_threshold, best_accuracy, prob_list, acc_list


def get_optimal_threshold(df, model, financial_report=False):
    threshold_list = []
    accuracy_list = []

    accuracy_matrix = []
    prob_matrix = []

    for i in range(THRESHOLD_TUNING_N_ITERS):
        val_0 = df[df[TARGET_NAME] == 0].drop(TARGET_NAME, axis=1).sample(n=THRESHOLD_TUNING_N_SAMPLES,
                                                                          random_state=SEED + i)
        val_1 = df[df[TARGET_NAME] == 1].drop(TARGET_NAME, axis=1).sample(n=THRESHOLD_TUNING_N_SAMPLES,
                                                                          random_state=SEED + i)

        prob_0 = model.predict_proba(val_0)[:, 1]
        prob_1 = model.predict_proba(val_1)[:, 1]

        best_threshold, best_accuracy, prob_list, acc_list = _get_optimal_threshold_by_accuracy(
            np.hstack([prob_0, prob_1]),
            [0] * len(prob_0) + [1] * len(prob_1)
        )
        threshold_list.append(best_threshold)
        accuracy_list.append(best_accuracy)
        prob_matrix.append(prob_list)
        accuracy_matrix.append(acc_list)

    optimal_threshold = np.median(threshold_list)

    fig = plt.figure(figsize=(18, 12))
    plt.vlines(optimal_threshold, 0.5, 0.75, colors='black',
               label=f'optimal threshold {optimal_threshold: .3f}')

    for x, y in zip(prob_matrix, accuracy_matrix):
        plt.plot(x, y)
    plt.xlabel('Accuracy')
    plt.ylabel('Threshold')
    plt.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR.joinpath(
        f"threshold_tuning_data_with{'' if financial_report else 'out'}_financial report.png"))

    return optimal_threshold
