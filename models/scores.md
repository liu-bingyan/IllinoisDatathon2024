For binary classification, 
True Positive = predict 1 for real 1, 
False Positive = predict 1 for real 0,
False Negative = predict 0 for real 1,
True Negative = predict 0 for real 0.

TPR = True Positive Rate (TPR), Sensitivity, or Recall: Proportion of correctly classified positive instances.
    =  TP / (TP + FN)  = discovered positive / all positives
FPR = False Positive Rate (FPR): Proportion of incorrectly classified negative instances.
    = FP /  (FP + TN) =  wrongly labelled negatives / all negatives

ROC curve: the FPR v.s. TPR curve

AUC area under the ROC curve when training. AUC < 0.5 means worth than a random classifier. The closer AUC is to 1 the better. 

```

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc 

    # Generate sample data
    X, y = make_classification(n_samples=1000, random_state=0)

    # Train a classifier
    model = LogisticRegression()
    model.fit(X, y)

    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print('AUC:', roc_auc)

```

AUC generalize to multiclassifications


# questions

model is outputing argmax instead of prob
overfitting

replace original mos and call_key
why the dataset 4 fails
resolved == mos_TR