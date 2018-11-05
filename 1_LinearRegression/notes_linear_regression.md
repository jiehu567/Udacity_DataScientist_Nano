## Simple Linear Regression

```{python}
# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
y = bmi_life_data[['Life expectancy']]
X = bmi_life_data[['BMI']]

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

print(model.predict)
```

## Polynomial Regression

```{python}
# TODO: Add import statements
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv')
X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y'].values

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X, y)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)
```

## Regularization

Add corresponding panalty to net increases of error for some features.

complex models have more complex terms so easier to get bigger combined error

- L1 Regularization - adding absolute values of coefficients
- L2 Regularization - adding squared values of coefficients

$$
    2x_1^3 - 2x_2^2 - 4x_2^3 + 3x_1^2 + 6x_1x_2 + 4x_2^2 + 5 = 0
$$

Then L1:

$$
    Error = |2| + |2| + |-4| + |3| + |6| + |4| + 5
$$

L2:

$$
    Error = 2^2 + 2^2 + (-4)^2 + 3^2 + 6^2 + 4^2 + 5^2 = 85
$$

To make small extra error, multiply by a small lambda.

### Compare L1 and L2

- computation: L1 is computationally inefficient than L2 even it's simpler but hard to differentiate to get derivatives,
- L1 is faster when input is sparse data
- L1 is good for feature selections, making noise columns to zero
- L2 will treat all columns similarly,   L1-regularization is more likely to create 0-weights.

### Explain to why L1 prefer sparsity:

With a sparse model, we think of a model where many of the weights are 0. Let us therefore reason about how L1-regularization is more likely to create 0-weights.

Consider a model consisting of the weights $(w_1, w_2, \dots, w_m)$.

With L1 regularization, you penalize the model by a loss function $L_1(w)$ = $\Sigma_i |w_i|$.

With L2-regularization, you penalize the model by a loss function $L_2(w)$ = $\frac{1}{2} \Sigma_i w_i^2$

If using gradient descent, you will iteratively make the weights change in the opposite direction of the gradient with a step size $\eta$ multiplied with the gradient. This means that a more steep gradient will make us take a larger step, while a more flat gradient will make us take a smaller step. Let us look at the gradients (subgradient in case of L1):

$\frac{dL_1(w)}{dw} = sign(w)$, where $sign(w) = (\frac{w_1}{|w_1|}, \frac{w_2}{|w_2|}, \dots, \frac{w_m}{|w_m|})$

$\frac{dL_2(w)}{dw} = w$

If we plot the loss function and it's derivative for a model consisting of just a single parameter, it looks like this for L1:

![enter image description here][1]

And like this for L2:

[![enter image description here][2]][2]

Notice that for $L_1$, the gradient is either 1 or -1, except for when $w_1 = 0$. That means that L1-regularization will move any weight towards 0 with the same step size, regardless the weight's value. In contrast, you can see that the $L_2$ gradient is linearly decreasing towards 0 as the weight goes towards 0. Therefore, L2-regularization will also move any weight towards 0, but it will take smaller and smaller steps as a weight approaches 0.

Try to imagine that you start with a model with $w_1 = 5$ and using $\eta = \frac{1}{2}$. In the following picture, you can see how gradient descent using L1-regularization makes 10 of the updates $w_1 := w_1 - \eta \cdot \frac{dL_1(w)}{dw} = w_1 - \frac{1}{2} \cdot 1$, until reaching a model with $w_1 = 0$:

![enter image description here][3]

In constrast, with L2-regularization where $\eta = \frac{1}{2}$, the gradient is $w_1$, causing every step to be only halfway towards 0. That is, we make the update $w_1 := w_1 - \eta \cdot \frac{dL_2(w)}{dw} = w_1 - \frac{1}{2} \cdot w_1$
Therefore, the model never reaches a weight of 0, regardless of how many steps we take:

![enter image description here][4]

Note that L2-regularization **can** make a weight reach zero if the step size $\eta$ is so high that it reaches zero in a single step. Even if L2-regularization on its own over or undershoots 0, it can still reach a weight of 0 when used together with an objective function that tries to minimize the error of the model with respect to the weights. In that case, finding the best weights of the model is a trade-off between regularizing (having small weights) and minimizing loss (fitting the training data), and the result of that trade-off can be that the best value for some weights are 0.


## Code for Regularization

Use sklearn's Lasso class to fit a linear regression model to the data, while also using L1 regularization to control for model complexity.

```{python}
# TODO: Add import statements
from sklearn import linear_model
import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header = None)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
```

## Feature Scaling

Goal: transform data into a common range of values.

Types:

- Standardizing
> df["height_standard"] = (df["height"] - df["height"].mean()) / df["height"].std()

- Normalizing
> df["height_normal"] = (df["height"] - df["height"].min()) / (df["height"].max() - df['height'].min())

**When to use Feature Scaling?**

- model is distance based (SVM, kNN, kMeans)
- incorporate regularization


## Code of Regularization with Feature Scaling

```{python}
# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header = None)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# TODO: Create the standardization scaling object.
scaler = StandardScaler()

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X_scaled, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
```

  [1]: http://i.stack.imgur.com/cmWO0.png
  [2]: https://i.stack.imgur.com/Mkclz.png
  [3]: http://i.stack.imgur.com/XmtF2.png
  [4]: http://i.stack.imgur.com/jlQYp.png