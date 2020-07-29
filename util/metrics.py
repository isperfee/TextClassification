from sklearn import metrics


def evaluate(y_data, y_pred, target_names=None):
    original_report = metrics.classification_report(y_data, y_pred, target_names=target_names)
    print(original_report)
    report = {
        'detail': original_report,
        **original_report['weighted avg']
    }
    return report
