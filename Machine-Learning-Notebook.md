# 机器学习入门笔记
## 1. Regression

- Machine learning library: Scikit-learn

### 1.1. Learning Rate
$$
w = w - \alpha \frac{\mathrm{d}}{\mathrm{d} x} J(w) 
$$

$ \alpha $ determines the step size taken along the gradient.

### 1.2. Gradient Descent for Univariate Linear Regression

#### Linear Regression Equation

$$
f{_{w,b}}(x) = wx + b
$$

#### Loss Function

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^{2m}(f_{w,b}(x^{(i)}) - y^{(i)})^2
$$

### 1.3. Gradient Descent for Multivariate Linear Regression

#### Parameter Definitions

$x_j$ – the $j$-th feature in a training example

$n$ – the number of features

${\vec x}^{(i)}$ – the $i$-th training example

${x}^{(i)}_j$ – the $j$-th feature of the $i$-th training example

#### Linear Regression Equation

$$
f{_{w,b}}(x) = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + \dots + w_nx_n+ b
$$

#### Vectorization

$$
f{_{w,b}}(x) = \vec w \cdot \vec x + b
$$

$\vec w = (w_1, w_2, w_3, \dots, w_n)$

$\vec x = (x_1, x_2, x_3, \dots, x_n)$

$b\text{ is a number}$

#### Dot product for variable with vectorization in Python

``` python
f = np.dot(w, x) + b
```

### 1.4. Feature scaling

#### Mean normalization

$$
\frac {{x_1} - {\mu_1}} {x_{max} - x_{min}}
$$

$x \text { is a certain variable}$

$\mu \text{ is the average number}$

#### Z-score normalization

$$
x_1 = \frac {x_1 - \mu_1} {\sigma_1}
$$

$\sigma_1 \text{ is the standard deviation}$

