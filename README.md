## Data Analysis & Visualization in Python

This document summarizes theoretical concepts and practical code examples using Python’s most widely used libraries for data analysis, numerical computing, and visualization — **NumPy, Pandas, Matplotlib, Seaborn,** and **Plotly**.

#### What is NumPy, and why is it widely used in Python?

NumPy (Numerical Python) is a library for efficient handling of large, multi-dimensional arrays and matrices. It supports advanced mathematical functions, vectorized operations, and linear algebra. Its performance, especially with large datasets, and compatibility with libraries like Pandas and Matplotlib make it widely used.


#### How does broadcasting work in NumPy?

Broadcasting allows NumPy to perform operations on arrays of different shapes by automatically expanding the smaller array. This removes the need for explicit looping and improves performance.



#### What is a Pandas DataFrame?

A DataFrame is a 2D, tabular data structure with labeled axes (rows and columns), capable of holding heterogeneous data types. It's similar to an Excel sheet or SQL table.



#### Explain the use of the `groupby()` method in Pandas.

The `groupby()` method is used to group data by one or more keys for aggregation, transformation, or filtering. It's useful for summarizing or analyzing grouped data (e.g., computing mean sales per region).

#### Why is Seaborn preferred for statistical visualizations?

Seaborn offers high-level interfaces for attractive statistical plots, integrates smoothly with Pandas DataFrames, and supports plots like heatmaps, boxplots, and regression plots with minimal code.

#### NumPy Arrays vs Python Lists

| Feature           | NumPy Arrays | Python Lists     |
| ----------------- | ------------ | ---------------- |
| Data Type         | Homogeneous  | Heterogeneous    |
| Performance       | Faster       | Slower           |
| Operations        | Vectorized   | Requires looping |
| Memory Efficiency | High         | Low              |

#### What is a heatmap?

A heatmap displays values in a matrix using color gradients. It’s ideal for visualizing correlations or patterns between variables.

#### What does "vectorized operation" mean in NumPy?

A vectorized operation applies operations over entire arrays without loops. This is faster and more efficient due to internal C-level optimizations.

#### How does Matplotlib differ from Plotly?

* **Matplotlib**: Creates static, publication-quality 2D plots.
* **Plotly**: Interactive, web-based visualizations with support for 3D, zooming, tooltips, and dashboards.


#### What is hierarchical indexing in Pandas?

Hierarchical indexing (MultiIndex) allows multiple levels of indexing in rows or columns, making it easier to handle complex datasets such as multi-group time series.

#### Purpose of `pairplot()` in Seaborn

`pairplot()` creates a grid of scatterplots and histograms for visualizing pairwise relationships between variables in a dataset.


#### Purpose of `describe()` in Pandas

`describe()` generates summary statistics of numerical data, such as mean, min, max, and quartiles. It helps quickly understand data distributions.

#### Why is handling missing data important?

Unaddressed missing data can lead to misleading analysis. Pandas provides tools to drop or fill missing values for cleaner, more accurate datasets.

#### Benefits of Plotly for Data Visualization

* Interactive and responsive plots
* 3D support
* Easy integration with web applications
* Useful for real-time dashboards

#### How does NumPy handle multidimensional arrays?

NumPy uses `ndarray` to handle arrays of any dimensionality. It supports indexing, reshaping, and linear algebra operations across multiple axes.

#### What is Bokeh?

Bokeh is a Python visualization library for creating browser-based interactive plots and dashboards. It supports zooming, tooltips, and real-time data.

#### `apply()` vs `map()` in Pandas

* `apply()`: Used on Series or DataFrame for complex row/column-wise operations.
* `map()`: Used on Series for element-wise transformations, often with dicts or functions.

#### Advanced NumPy Features

* Broadcasting
* Fancy indexing
* Random number generation
* Linear algebra tools
* Fast Fourier transforms
* Multidimensional slicing
* Efficient memory management

#### What is a Pivot Table in Pandas?

Pivot tables summarize data using group-based aggregation and reshaping. They are ideal for multi-dimensional analysis (e.g., total sales by region and quarter).

#### Why is NumPy slicing faster than Python lists?

NumPy stores data in contiguous memory blocks, allowing efficient slicing. Python lists are objects with additional overhead, reducing performance.

#### Common Use Cases for Seaborn

* Distribution plots (`distplot`, `histplot`)
* Correlation heatmaps
* Categorical plots (`boxplot`, `violinplot`)
* Regression plots (`regplot`, `lmplot`)
* Pairwise relationships (`pairplot`)

## Practical Questions & Code

#### Create a 2D NumPy array and calculate row-wise sum

```python
import numpy as np
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_sums = array.sum(axis=1)
print(row_sums)  # Output: [ 6 15 24]
```

#### Find the mean of a column in Pandas

```python
import pandas as pd
data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
df = pd.DataFrame(data)
print(df['A'].mean())  # Output: 2.5
```

#### Scatter plot using Matplotlib

```python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
plt.show()
```

#### Correlation heatmap using Seaborn

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': [2, 3, 4, 5, 6]}
df = pd.DataFrame(data)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
```

#### Bar plot using Plotly

```python
import plotly.graph_objects as go

categories = ['A', 'B', 'C', 'D']
values = [10, 20, 30, 40]
fig = go.Figure([go.Bar(x=categories, y=values)])
fig.update_layout(title='Bar Plot Example', xaxis_title='Categories', yaxis_title='Values')
fig.show()
```

#### Add a new column in a Pandas DataFrame

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3, 4]})
df['B'] = df['A'] * 2
print(df)
```

#### Element-wise multiplication in NumPy

```python
import numpy as np
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print(array1 * array2)  # Output: [ 4 10 18]
```

#### Multiple line plots with Matplotlib

```python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [25, 16, 9, 4, 1]
plt.plot(x, y1, label='y = x^2')
plt.plot(x, y2, label='y = 25 - x^2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multiple Line Plot')
plt.legend()
plt.show()
```

#### Filter DataFrame rows based on column value

```python
import pandas as pd
data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)
print(df[df['B'] > 30])
```

---

#### Histogram with Seaborn

```python
import seaborn as sns
import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
sns.histplot(data, kde=True)
plt.title('Histogram with KDE')
plt.show()
```

#### Matrix multiplication using NumPy

```python
import numpy as np
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.matmul(matrix1, matrix2)
print(result)  # Output: [[19 22], [43 50]]
```

#### Load CSV with Pandas and show first 5 rows

```python
import pandas as pd
df = pd.read_csv('file1.csv')
print(df.head())
```

#### 3D scatter plot using Plotly

```python
import plotly.graph_objects as go

x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
z = [1, 2, 3, 4, 5]
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
fig.update_layout(title='3D Scatter Plot', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.show()
```
