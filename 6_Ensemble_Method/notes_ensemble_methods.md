# Ensemble Methods

Goal: join models together to get a better model

- Bagging: bootstrap aggregating
  - get individual model predicitons
  - combine them together

- Boosting
  - similar but tries harder to exploit each model's strength
  - one model (strong learners) might answer well in some part of the data but not well in other parts(weak learner) while the other model might be more powerful in another part. It's not necessary to have all model as strong learners, each of them is just need to be slightly better than random guess


## Why Ensemble

- Bias vs. Variance

> Bias: the model doesn't do a good job of bending to the data. For example, linear regression always has bias. Even with different dataset, we can have same line.

> Variance: the model changes drastically to meet the needs of every point in dataset. Example: decision tree tends to split every point into own branch if possible.

By combining algorithms, we can build models to perform better by meeting in the middle in terms of bias and variance.

## Introduce randomness into ensembles

To improve ensemble methods: introdue randomness into hight variance algorithms before ensembled.

Methods:

- Bootstrap the data - sampling data with replacement and fitting algorithms to sampled data

- Subset the features - in each split of decision tree or with each algorithm used an ensemble only a subset of the total possible features are used

This model: Random Forests

## Random Forests

> Problems of Decision Trees: memorize data: overfitting with high variance

### Idea

- random choose features, vote to get result


## Bagging

> Weak learners: The simplest possible learner, e.g.: one node decision tree. Each is like a horizontal or vertical line.

- take subsets and train a weak learner
- combine by voting

## Boosting

### Adaboost

> Fit the first **simplest** learner in order to maximize accuracy - minimize number of errors

> Second learner needs to fix on the mistakes that this one has made - take misclassified points, and make them bigger - punish the model more if it misses these points

The way to change the weight: sum the weight of correct classified points C and the weight of misclassified points M, then make sure correct / incorrect weights is 50/50. then the constant will be: 

$$C / M$$

and multiply weights of misclassified with this number.

> do the same again

> combine the models

- determine the weight: how well each model is doing

Standard: if half / half, this learner is worst.
We can have weights:
> super positive for good learner, 0 for half/half learner, super negative for liar

$$
    weight = ln(\frac{accuracy}{1-accuracy})
$$

this formula is very negative for x=0, 0 for x=0.5 and very positive for x=1

Theoritically, we might have acc=0 case, which lead to infinity of weight, but in practice this is unlikely to happen.

If happens, just use infinit as the model.


## Calculate the weights of learners

- sum of weights of correctly classified points: 7
- incorrect: 3
- $weight = ln(\frac{3}{7})=0.84$

pick up incorrect points, update their weights to:
$originalWeight * numberOfCorrectPoints / numberOfIncorrectPoints$

Then,
- correct: 11
- incorrect: 3
- weight = 1.3

Vote with weight.
For each decision region, we sum together, if positive, then blue, else red.


## Code

### Basic

```{python}
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
```

Hyperparameters

- base_estimator: default: decision tree
- n_estimators: the maximum number of weak learners used

```{python}
from sklearn.tree import DecisionTreeClassifier
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
```
