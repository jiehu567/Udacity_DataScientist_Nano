# Decision Tree

## Idea
Decision tree ask several questions about the data until narrow the information down well enough to make prediction.

It asks firstly which features are more decisive than other to determine the labels.

## Entropy

Measures how much freedom a particle have to move around.

### Example:

Freedom: Vapor > Water > Ice

Entropy: Vapor > Water > Ice

### Another Example: 3 buckets

4 red balls < 3 red balls + 1 blue ball < 2 blue 2 red

Entropy is the ways we can use balls in one bucket to put into a line. Knowledge is if we pickup random samples from bucket, how much we know about the ball in bucket.

The first bucket only has one possibility. But high knowledge

The 2nd can have 4, but has medium knowledge

The 3rd can have 6. but has low knowledge

The more knowledge, the less entropy.

## Formular

The negatives of the logarithms of the probabilities of picking the balls in a way that we win the game.

Example:

5 red balls + 3 blue balls

$$
    Entropy = - \frac{5}{8}log_2(\frac{5}{8}) - \frac{3}{8}log_2(\frac{3}{8}) = 0.9544
$$

or: m red balls + n blue balls

$$
    Entropy = - \frac{m}{m+n}log_2(\frac{m}{m+n}) - \frac{n}{m+n}log_2(\frac{n}{m+n})
$$

m / (m+n) is the probability.

So entropy is:

$$
    Entropy = -p_1log_2(p_1)-p_2log_2(p_2)
$$

```{python}
import math
def entropy(m, n):
    p1 = m / (m+n)
    p2 = 1 - p1
    return - (p1) * math.log(p1, 2) - p2 * math.log(p2, 2)
```

in multi-class case:

$$
    Entropy = - \sum_{i=1}^{n}p_ilog_2(p_i)
$$

The minimum value is still 0, when all elements are of the same value.

The maximum value is still achieved when the outcome probabilities are the same, but the upper limit increases with the number of different outcomes. (For example, you can verify the maximum entropy is 2 if there are four different possibilites, each with 0.25)

```{python}
def entropy_multiclasses(d):
    entropy = 0
    s = sum(d.values())
    for k in d:
        p = d[k] / s
        entropy += -p * math.log(p, 2)
    return entropy
```

## Information Gain

difference between the entropy of the parent and average of children

maximum information gain: 1
minimum information gain: 0

## Build Decision Tree

| Gender | Occupation | App        |
|--------|------------|------------|
| F      | Study      | Pokemon Go |
| F      | Work       | Whatsapp   |
| M      | Work       | Snapchat   |
| F      | Work       | Whatsapp   |
| M      | Study      | Pokemon Go |
| M      | Study      | Pokemon Go |

**Target**: look at the possible splits that each column gives, calculate the information gain, pick the largest one

> Step 1: Calculate the entropy of labels

* 3 - Pokemon Go
* 2 - Whatsapp
* 1 - Snapchat

$$
    Entropy = - \frac{3}{6}log_2(\frac{3}{6}) - \frac{2}{6}log_2(\frac{2}{6})- \frac{1}{6}log_2(\frac{1}{6}) = 1.46
$$

> Step 2: split by Each column

- by Gender

        * F: 1 Pokemon Go and 2 Whatsapp
        * M: 2 Pokemon Go and 1 Snapchat
        Entropy(F) = Entropy(M) = 0.92
        avg_Entropy = 0.92
        Information Gain by Gender = difference
        1.46 - 0.92 = 0.54
        (Parent - Children)
        Parent purity <= children's

* by Occupation

        * Study: 3 Pokemon Go
        * Work:  2 Whatsapp, 1 Snapchat
        Entropy_avg = (0 + 0.92) /2 = 0.46
        Info Gain = 1.46 - 0.46 = 1

> Step 3: pick the column with highest Info Gain

So Occupation has higher IG than other.

Then split the dataset into two sets:

* with only Pokemon Go, split by Study as Occupation
* we can do better, split by gender

```{python}
def two_group_ent(first, tot):
    return -(first/tot*np.log2(first/tot) +(tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)
g17_ent = 15/24 * two_group_ent(11,15) +
           9/24 * two_group_ent(6,9)

answer = tot_ent - g17_ent  
```

## Hyperparameters

* maximum depth k: $2^k$ leaves max

* minimum number of samples per leaf: keep leave splited in balance, can be specified as int or float (%)

* Maximum number of features

* Minimum number of samples to split a node

## sklearn

> from sklearn.tree import DecisionTreeClassifier

> model = DecisionTreeClassifier()

> model.fit(x_values, y_values)

max_depth: The maximum number of levels in the tree.

min_samples_leaf: The minimum number of samples allowed in a leaf.

min_samples_split: The minimum number of samples required to split an internal node.

max_features : The number of features to consider when looking for the best split.


```{python}
# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier()

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
```

