# Training and Tuning

## Underfitting vs. Overfitting

- Underfitting: bias, not do well in training set
- Overfitting: memorize, do well in training set but testing set, error due to high variance

## Detect Error

## Cross Balidation

### Model Complexity Graph

underfit -> right -> overfit

high bias error -> high variance

low degree -> high degree (more complex)

## Don't use testing data to train model

3 sets:

- Training
- Validation: make decision
- Testing

### K-Fold Validation

Goal: Not necessary lose training data for validation

- Break data into K buckets
- Train K times: choose one bucket as testing set and the remaining K-1 as traing data
- Avg the results to get final model

```{python}
from sklearn.model_selection import KFold
kf = KFold(12, 3， shuffle = True)

for train_indices, test_indices in kf:
    ......

```

Parameters:
- n_splits
- size of testing set
- shuffle = True, randomly

