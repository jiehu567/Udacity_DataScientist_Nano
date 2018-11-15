# Model Evaluation

## Measure Generalization

training vs. testing sets

```{python}
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```

## Confusion Matrix

|        | Diagnosed Sick | Diagnosed Health |
|--------|----------------|------------------|
| Sick   | True Positive  | False Negative   |
| Health | False Positive | True Negative    |

When a patient is sick and the model correctly diagnosed sick, the case is called true positive

When a patient is healthy and the model correctly diagnoses them as healthy, this case is called true negative

When a patient is sick and the model incorrecly diagnosed as health, the case we call false negative

When a patient is health and the model incorrectly diagnosed as sick, the case we call false positive

- Type 1 Error: False Positive - misdiagnosed health patient
- Type 2 Error: False Negative - misdiagnosed sick patient

### Which is more important?

Convenience vs. Danger

We prefer solve the danger side, thought more inconvenient.

Example:

A false positive implies sending a healthy person to get more tests. This is slightly inconvenient, but ok. A false negative implies sending a sick person home, which can be disastrous!



## Accuracy

With a group of patients, how much % we diagnosed correctly.

$$
    Accuracy = \frac{truePositive + trueNegative}{TotalCnt}
$$

```{python}
    from sklearn.metrics import accuracy_score
    accuracy_score(y_true, y_pred)
```

### When accuracy won't work

Imbalance data with extreme ratio among labels


## Precision

Out of diagnosed positive, how many are really positive

Spam email: high precision

## Recall - Sensitivity

Out of all positives, how many are diagnosed positive

Diagnose: high recall required

## F1 Score

Harmonic mean

Always less than arithmetic mean. **Closer to the lower number.**

$$
    F1 = 2\times \frac{precision * recall}{precision + recall}
$$

## F-beta Score

We care more about precision or recall
We will use $F_{0.5}$ or $F_{2}$

$$
    F_{\beta} = (1+\beta^2)\frac{Precision * Recall}{\beta^2Precision + Recall}
$$

- When $\beta->0$, we get precision
- When $\beta ->\infty$, we get recall
- When $\beta=1$, we get F1 score

Examples:
- Spaceship parts: care more about sensitivity, so choose recall
- Notification to phone user who may like: we care both: F1 score
- Promotional Material to right audience: we don't want to waste money on wrong person, so we only care more about precision: among all tested audience, how much % is really target audience


## ROC: Receiver operating characteristics

for binary

Always used to tune the threshold to decide label.
For example, threshold probability we should classify a spam. When this probability change, we'll have different split and get different TP rate and FP rate pairs, so we have ROC curve.

The model with higher Area Under Curve wins. (each model has results of different thresholds)

Measure perfect of a split.

- perfect split get 1
- random split get 0.5

True positive Rate: $\frac{TruePositives}{AllPositives}$

is this one Recall?


False positive Rate: $\frac{FalsePositives}{AllNegatives}$

Move the split and calculate above to get a curve:
TP rate vs. FP rate

We care the area under curve.
Worst: random guess, we care triangle area which is 0.5.

## ROC Curve Extension

a way to visualize performance of binary classifier

This is not a balance test
[Visualize Demo](http://www.navan.name/roc/)

1 pixle = 1 paper

red is admitted paper by model

blue is not admitted by model

> ROC Curve: TP vs. NP for all possible threshold

AUC: area under curve, if it's taking 0.5 of all space, it's a bad model which is not better than random guess

If takes 1, the classifier well split the data.

AUC is insensitive to these 2 label probability distributions and if data is balance.

To extend to 3+ classes, use:

- class 1 vs. non-class 1
- class 2 vs. non-class 2

and similar

ROC curve only care the rank ( label and probability to this label)

## Code

```{python}
def accuracy(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the accuracy as a float
    '''
    return np.sum(preds == actual)/len(actual)

def precision(actual, preds):
    '''
    INPUT
    (assumes positive = 1 and negative = 0)
    preds - predictions as a numpy array or pandas series 
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the precision as a float
    '''
    pred_pos = preds == 1
    true_pos_in_preds = (preds == actual) & (actual == 1)
    
    
    return sum(true_pos_in_preds) / sum(pred_pos) # calculate precision here


def recall(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the recall as a float
    '''
    
    actual_pos = actual == 1
    pred_pos_in_actual_pos = (preds == actual) & (preds == 1)

    return sum(pred_pos_in_actual_pos) / sum(actual_pos) # calculate recall here


def f1(preds, actual):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the f1score as a float
    '''
    
    prec = precision(actual, preds)
    rec  = recall(actual, preds)
    
    
    return 2 * prec * rec / (prec + rec) # calculate f1-score here










# add the letter of the most appropriate metric to each statement
# in the dictionary
a = "recall"
b = "precision"
c = "accuracy"
d = 'f1-score'


seven_sol = {
'We have imbalanced classes, which metric do we definitely not want to use?': c, # letter here,
'We really want to make sure the positive cases are all caught even if that means we identify some negatives as positives': a, # letter here,    
'When we identify something as positive, we want to be sure it is truly positive': b, # letter here, 
'We care equally about identifying positive and negative cases': d, # letter here    
}

t.sol_seven(seven_sol)
```


## Model Preferences of Metrics

- unbalanced data: don't use Naive Bayes
it's depend on prior

- 'We really want to make sure the positive cases are all caught even if that means we identify some negatives as positives': "naive Bayes". ???
example: fraud / crime, we care recall

- 'When we identify something as positive, we want to be sure it is truly positive': random forest, example: spam, we care precision

- 'We care equally about identifying positive and negative cases': naive bayes


## Regression Metrics

- mean absolute error

- mean squared error

Compare models with simplest model:

R2 score:

$$
    R2 = 1 - MSE(full)/MSE(base)
$$

if model is bad, R2 -> 0

???

#match each metric to the model that performed best on it
a = 'decision tree'
b = 'random forest'
c = 'adaptive boosting'
d = 'linear regression'


best_fit = {
    'mse': b,
    'r2': b,
    'mae': b
}

#Tests your answer - don't change this code
t.check_ten(best_fit)



## Review

$$
    Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
    Precision = \frac{TP}{TP + FP}
$$

$$
    Recall = \frac{TP}{TP + FN}
$$