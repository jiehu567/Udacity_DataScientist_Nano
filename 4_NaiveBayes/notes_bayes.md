# Naive Bayes

Based on conditional probability

Easy to implement and fast to train

> Prior: before we get information

> Post: after event occured

Use tree to calculate the post.

$$
    P(B|R) = \frac{P(B)P(R|B)}{P(A)P(R|A) + P(B)P(R|B)}
$$

## Assumption: independent events
But can be OK most of time if correlated

$$
    P(spam |'easy', 'money') -- P('easy', 'money'|spam) P(spam) -- P('easy'|spam)P('money'|spam)P(spam)
$$

Or: 

$$
    P(spam |'easy', ... 'money') -- P('easy'|spam)P('money'|spam)...P(spam)
$$
