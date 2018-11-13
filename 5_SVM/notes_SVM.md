# Support Vector Machine

use 2 paralell lines and make margin as larger as possible

> ERROR = classification ERROR + MARGIN ERROR

## Optimization Algorithm

punish according to the distance to the main line

minimize: sum of distance of misclassified points to the main line

## 1. Classification Error

Assume main line:

$$WX + b = 0$$

We want two extra lines to create the margin:

$$
    WX + b = 1
$$
$$
    WX + b = -1
$$

To punish the misclassified points, we don't want anything between the marginal lines.

We split into two parts:

> Blue error starts from $WX + b = -1$ and move above area and red error starts from the above line and going down

> errors start from 0 of baselines

> sum together the errors, which we want to minimize

### 2. Margin Error

Goal: to get as larger margin as possible

Cook up a function that gives a small error for large margin and large error for small error

margins:

$$
    Margin = \frac{2}{|W|}
$$

$$
    Error = |W|^2
$$

### Margin Error Calculation

First recall the nnotation, where $W=(w_1, w_2)$ and $x = (x_1, x_2)$ and $W=w_1x_1+w_2x_2$

Notice we now have 3 lines:

- $Wx+b=-1$
- $Wx+b=0$
- $Wx+b=1$

In order to find the distance between 1st and 3rd lines, we need to find the distance between the first 2 and multiply by 2.

Draw another line intersects the lines and perpendicular with them at point $(p, q)$, then:

- $w_1p + w_2q=1$
- $(p, q)$ is a multiple of $(w_1, w_2)$, $(p,q) = k(w_1, w_2)$

Solve, we get:

$$
    k = \frac{1}{w_1^2 + w_2^2} = 1/|W|^2
$$

So the intercect point represent as $W/|W|^2$
Which is also the distance, then we multiply by 2 and proof.

### Put together

$$Error = marginal + misclassified$$

### C parameter

Flexibility of the space.
It's a constant attached to classification error.

$$Error = C \times Classification Error + MarginError$$

If we have large C, then most error is classification error so we are focusing more on correctly classifying our points than in finding a good margin. But when C is small, most is marginal error.

## Polynomial Kernel - Parabola

We plot points in a plane instead of line, adding y axis for example, draw function: $y=x^2$

Now with y = 4 is a good cut.

The bundary will be $x=+2$ or $x=-2$

## Polynomial Kernel 2

> circle method: sacrifice the linearity, using higher degree polynomial equation

> building mehtod: sacrifice dimension of data

They are actually the same.

The function is:


$$
    x^2 + y^2 = Z
$$

The circle is actually a paraboliod intersects with each floor of building.


lower dimension => higher dimension and use linear function to seperate points, then going back to lower dimension, we have non-linear polynomial boundary.

The hyper parameter is: k - degree of allowed polynomial


## RBF

like sin(x)

with mountains and valleys

> build a mountain on top of every point: radial basis functions

> flip the one with different class

> add together

**Hyper parameter** $\gamma$
the larger it is, we have more specific divisions.

use Gassian distribution:

$$
    \gamma = \frac{1}{2\sigma^2}
$$

So if $\gamma$ is small, we have large variance, so the model will cover large variance, tending to underfit.


```{python}
# Import statements 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf', gamma=29)

# TODO: Fit the model.
model.fit(X, y)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)

```


---

## Recap

3 different ways that SVMs can be implemented:

- Maximum Margin Classifier
- Classification with Inseparable Classes
- Kernel Methods

### Maximum Margin Classifier

When your data can be completely separated, the linear version of SVMs attempts to maximize the distance from the linear boundary to the closest points (called the support vectors). For this reason, we saw that in the picture below, the boundary on the left is better than the one on the right.

### Classification with Inseparable Classes

Unfortunately, data in the real world is rarely completely separable as shown in the above images. For this reason, we introduced a new hyper-parameter called C. The C hyper-parameter determines how flexible we are willing to be with the points that fall on the wrong side of our dividing boundary. The value of C ranges between 0 and infinity. When C is large, you are forcing your boundary to have fewer errors than when it is a small value.

Note: when C is too large for a particular set of data, you might not get convergence at all because your data cannot be separated with the small number of errors allotted with such a large value of C.

### Kernels

Finally, we looked at what makes SVMs truly powerful, kernels. Kernels in SVMs allow us the ability to separate data when the boundary between them is nonlinear. Specifically, you saw two types of kernels:

### polynomial

rbf
By far the most popular kernel is the rbf kernel (which stands for radial basis function). The rbf kernel allows you the opportunity to classify points that seem hard to separate in any space. This is a density based approach that looks at the closeness of points to one another. This introduces another hyper-parameter gamma. When gamma is large, the outcome is similar to having a large value of C, that is your algorithm will attempt to classify every point correctly. Alternatively, small values of gamma will try to cluster in a more general way that will make more mistakes, but may perform better when it sees new data.


Resources
Support Vector Machines are described in Introduction to Statistical Learning starting on page 337.

The wikipedia page related to SVMs

The derivation of SVMs from Stanford's CS229 notes.