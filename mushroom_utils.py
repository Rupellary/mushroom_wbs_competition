import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay



def get_scores(X, y, model, thresh = 0.5, name='Model'):
    """Description: generates dataframe of different performance metrics for a model
    INPUTS: X = training or test data to predict with
            y = corresponding training or test data to compare with predictions and evaluate
            model = model to predict with
            thresh = confidence required to label positive, probability between 0.0 and 1.0
            name = name of model to label the data frame with
    OUTPUT: data frame with performance metrics"""
    probs = model.predict_proba(X)[:,1]
    pred = (probs >= thresh).astype(bool)
    acc = accuracy_score(y_true = y,
                         y_pred = pred)
    conf_arry = confusion_matrix(y, pred, labels=None, sample_weight=None, normalize=None)
    fp = conf_arry[0,1]
    fn = conf_arry[1,0]
    pre = precision_score(y_true = y,
                          y_pred = pred)
    rec = recall_score(y_true = y,
                       y_pred = pred)
    f1 = f1_score(y_true = y,
                  y_pred = pred)
    auc = roc_auc_score(y, probs)
    df = pd.DataFrame({'Accuracy': acc,
                      'Precision': pre,
                      'Recall': rec,
                      'AUC': auc,
                      'F1': f1,
                      'Oportunity Cost': fp,
                      'Killed': fn}, index=[name])
    return df


def get_safest_thresh(X, y, model):
    """Description: generates data frame with false and true positive rates with varying thresholds and a score of the highest threshold that leaves no false negatives
    INPUTS: X = training or test data used to predict probabilities
            y = training or test data used to evaluate predictions
            model = model used for predictions
    OUTPUTS: thresh_df = data frame of true and false positive rates for various threshold values
             safest_thresh = highest threshold that still leaves no false negatives
    """
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, thresh = roc_curve(y, probs)
    thresh_df = pd.DataFrame({'Threshold':thresh,
                             'False Positive Rate':fpr,
                             'True Positive Rate':tpr})
    rf_safe_thresh = thresh_df.loc[thresh_df['True Positive Rate']==1]
    safest_thesh = rf_safe_thresh['Threshold'].max()
    return thresh_df, safest_thesh


def full_scoring(X_train, y_train, X_test, y_test, model, name):
    """Description: computes scores for training and testing data with normal predictions and safe threshold predictions
    INPUTS: X_train = predictor features training data
            y_train = target feature training data
            X_test = predictor features test data
            y_test = target feature test data
            model = model used to predict
            name = name of model type
    OUTPUTS: train_pred_scores = score data frame for normal predictions on training data
             test_pred_scores = score data frame for normal predictions on test data
             train_rec_scores = score data frame for safe predictions on training data
             test_pred_scores = score data frame for safe predictions on test data
    """
    thresh_df, safest_thresh = get_safest_thresh(X_train, y_train, model)
    train_pred_scores = get_scores(X_train, y_train, model, thresh = 0.5, name=f'{name} Train Prediction')
    test_pred_scores = get_scores(X_test, y_test, model, thresh = 0.5, name=f'{name} Test Prediction')
    train_rec_scores = get_scores(X_train, y_train, model, thresh = safest_thresh, name=f'{name} Train Recommendation')
    test_rec_scores = get_scores(X_test, y_test, model, thresh = safest_thresh, name=f'{name} Test Recommendation')
    return train_pred_scores, test_pred_scores, train_rec_scores, test_rec_scores


def prep_for_sub(model, thresh, test_df, name='sml'):
    """Description: creates csv file with predictions
    INPUTS: model = model using to predict
            thresh = threshold chosen for cautious prediction
            test_df = data frame with test data
            name = name of model to differentiate submissions
    OUTPUTS: returns nothing
             creates file named mushroom_challenge_answers.csv in the folder in which the code is running
    """
    probs = model.predict_proba(test_df)[:,1]
    pred = (probs >= thresh).astype(bool)
    df = test_df
    df['poisonous'] = pred
    answers_df = df.loc[:,['Id','poisonous']].set_index('Id')
    answers_df.to_csv(f'{name}_mushroom_challenge_answers.csv')