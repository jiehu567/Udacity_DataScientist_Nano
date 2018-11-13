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