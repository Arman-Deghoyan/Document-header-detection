from sklearn import metrics
from sklearn.metrics import accuracy_score


def classification_results(saved_model, y_test):

    y_pred = saved_model.predict(y_test['row'])
    y_pred_prb = saved_model.predict_proba(y_test['row'])

    test_score =  round(accuracy_score(y_test['label'], y_pred), 3)
    print(f'test accuracy {test_score}')

    print(metrics.confusion_matrix(y_test['label'], y_pred))
    print(metrics.classification_report(y_test['label'], y_pred))

    print('ROC AUC Score is' + '\n')
    print(metrics.roc_auc_score(y_test['label'], y_pred_prb[:, 1]))
    

