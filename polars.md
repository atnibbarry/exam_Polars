
![](https://github.com/atnibbarry/exam_Polars/blob/4017af82c29388ef70ede69c383c62d7854914cf/img/Introduction-to-Pypolars.jpg)
 # **POLARS**
### Content 
1. [Introduction](#introduction)
1. [Comparison with Pandas](#comparison-with-pandas) 
3. [Installation ](#installation)
4. [Polars Expressions](#polars-expressions)

    * [Expression Contexts](#expression-contexts) 
    * [Numpy Interop](#numpy-interop)
6. [Pratical use case  ](#practical-use-cases)

    * [read & write](#read--write)
    * [selecting & data](#selecting-data)
    * [data handling](#data-handling)
    * [combining data with concat and join](#combining-data-with-concat-and-join)
7. [To go further](#to-go-further) 

### Introduction
[Polars](https://pola-rs.github.io/polars-book/user-guide/introduction.html) is a lightning fast library that: utilizes all available cores on your machine, optimizes queries to reduce unneeded work/memory allocations, handles [datasets](https://datascientest.com/dataset-definition) much larger than your available RAM, has an [API](https://www.redhat.com/fr/topics/api/what-are-application-programming-interfaces) that is consistent and predictable Has a strict schema (data-types should be known before running the query). 
### Comparison with Pandas
[Pandas](https://pandas.pydata.org/docs/user_guide/index.html) is a very versatile tool for small data. Polars is a versatile tool for small and large data with a more predictable, less ambiguous and stricter API.
### Installation 
Installing and using is just a simple ... or away. Refer to the official [documentation](https://pola-rs.github.io/polars-book/user-guide/quickstart/intro.html).
#### Example
*Below we show a simple snippet that parses a CSV file, filters it, and finishes with a groupby operation.*
```python
import polars as pl

df = pl.read_csv("https://j.mp/iriscsv")
print(df.filter(pl.col("sepal_length") > 5)
      .groupby("species", maintain_order=True)
      .agg(pl.all().sum())
)
```
The snippet above will output:
```python
shape: (3, 5)
┌────────────┬──────────────┬─────────────┬──────────────┬─────────────┐
│ species    ┆ sepal_length ┆ sepal_width ┆ petal_length ┆ petal_width │
│ ---        ┆ ---          ┆ ---         ┆ ---          ┆ ---         │
│ str        ┆ f64          ┆ f64         ┆ f64          ┆ f64         │
╞════════════╪══════════════╪═════════════╪══════════════╪═════════════╡
│ setosa     ┆ 116.9        ┆ 81.7        ┆ 33.2         ┆ 6.1         │
├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ versicolor ┆ 281.9        ┆ 131.8       ┆ 202.9        ┆ 63.3        │
├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ virginica  ┆ 324.5        ┆ 146.2       ┆ 273.1        ┆ 99.6        │
└────────────┴──────────────┴─────────────┴──────────────┴─────────────┘
```
As we can see, pretty-prints the output object, including the column name and datatype as headers.
### Polars Expressions
Polars has a powerful concept called expressions that is central to its very fast performance.

**Example 1:**
In this section we will go through some examples, but first let's create a dataset:
```python
import polars as pl
import numpy as np

np.random.seed(12)

df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", None],
        "random": np.random.rand(5),
        "groups": ["A", "A", "B", "C", "B"],
    }
)
print(df)
```
```python
shape: (5, 4)
┌──────┬───────┬──────────┬────────┐
│ nrs  ┆ names ┆ random   ┆ groups │
│ ---  ┆ ---   ┆ ---      ┆ ---    │
│ i64  ┆ str   ┆ f64      ┆ str    │
╞══════╪═══════╪══════════╪════════╡
│ 1    ┆ foo   ┆ 0.154163 ┆ A      │
│ 2    ┆ ham   ┆ 0.74005  ┆ A      │
│ 3    ┆ spam  ┆ 0.263315 ┆ B      │
│ null ┆ egg   ┆ 0.533739 ┆ C      │
│ 5    ┆ null  ┆ 0.014575 ┆ B      │
└──────┴───────┴──────────┴────────┘
```
**Count unique values** :
We can count the unique values in a column. Note that we are creating the same result in different ways. To avoid duplicate column names in the DataFrame, we could use an alias expression that can rename the expression.
```python
out = df.select(
    [
        pl.col("names").n_unique().alias("unique_names_1"),
        pl.col("names").unique().count().alias("unique_names_2"),
    ]
)
print(out)
```
```python
shape: (1, 2)
┌────────────────┬────────────────┐
│ unique_names_1 ┆ unique_names_2 │
│ ---            ┆ ---            │
│ u32            ┆ u32            │
╞════════════════╪════════════════╡
│ 5              ┆ 5              │
└────────────────┴────────────────┘
```

**Filter and conditionals** :
We can also do some pretty complex things. In the next snippet we count all names ending with the string ."am"
```python
out = df.select(
    [
        pl.col("names").filter(pl.col("names").str.contains(r"am$")).count(),
    ]
)
print(out)
```
```python
shape: (1, 1)
┌───────┐
│ names │
│ ---   │
│ u32   │
╞═══════╡
│ 2     │
└───────┘
```
**Various aggregations** : We can do various aggregations. Below are examples of some of them, but there are more such as etc.median mean first
```python
out = df.select(
    [
        pl.sum("random").alias("sum"),
        pl.min("random").alias("min"),
        pl.max("random").alias("max"),
        pl.col("random").max().alias("other_max"),
        pl.std("random").alias("std dev"),
        pl.var("random").alias("variance"),
    ]
)
print(out)
```
```python
shape: (1, 6)
┌──────────┬──────────┬─────────┬───────────┬──────────┬──────────┐
│ sum      ┆ min      ┆ max     ┆ other_max ┆ std dev  ┆ variance │
│ ---      ┆ ---      ┆ ---     ┆ ---       ┆ ---      ┆ ---      │
│ f64      ┆ f64      ┆ f64     ┆ f64       ┆ f64      ┆ f64      │
╞══════════╪══════════╪═════════╪═══════════╪══════════╪══════════╡
│ 1.705842 ┆ 0.014575 ┆ 0.74005 ┆ 0.74005   ┆ 0.293209 ┆ 0.085971 │
└──────────┴──────────┴─────────┴───────────┴──────────┴──────────┘
```
#### Expression contexts
You cannot use an expression anywhere. An expression needs a context, the available contexts are:

* selection: df.select([..]) 
* groupby aggregation: df.groupby(..).agg([..]) 
* hstack/ add columns: df.with_columns([..]) 

**Syntactic sugar** : The reason for such a context, is that you actually are using the Polars lazy API, even if you use it in eager. For instance this snippet:
```python
df.groupby("foo").agg([pl.col("bar").sum()])
```
actually desugars to:
```python
(df.lazy().groupby("foo").agg([pl.col("bar").sum()])).collect()
```
This allows Polars to push the expression into the query engine, do optimizations, and cache intermediate results.

**Select context** :
In the select context the selection applies expressions over columns. The expressions in this context must produce Series that are all the same length or have a length of 1.

A Series of a length of 1 will be broadcasted to match the height of the DataFrame. Note that a select may produce new columns that are aggregations, combinations of expressions, or literals.

```python
out = df.select(
    [
        pl.sum("nrs"),
        pl.col("names").sort(),
        pl.col("names").first().alias("first name"),
        (pl.mean("nrs") * 10).alias("10xnrs"),
    ]
)
print(out)
```
```python
shape: (5, 4)
┌─────┬───────┬────────────┬────────┐
│ nrs ┆ names ┆ first name ┆ 10xnrs │
│ --- ┆ ---   ┆ ---        ┆ ---    │
│ i64 ┆ str   ┆ str        ┆ f64    │
╞═════╪═══════╪════════════╪════════╡
│ 11  ┆ null  ┆ foo        ┆ 27.5   │
│ 11  ┆ egg   ┆ foo        ┆ 27.5   │
│ 11  ┆ foo   ┆ foo        ┆ 27.5   │
│ 11  ┆ ham   ┆ foo        ┆ 27.5   │
│ 11  ┆ spam  ┆ foo        ┆ 27.5   │
└─────┴───────┴────────────┴────────┘
```
**Groupby context** :
In the groupby context expressions work on groups and thus may yield results of any length (a group may have many members).
```python
out = df.groupby("groups").agg(
    [
        pl.sum("nrs"),  # sum nrs by groups
        pl.col("random").count().alias("count"),  # count group members
        # sum random where name != null
        pl.col("random").filter(pl.col("names").is_not_null()).sum().suffix("_sum"),
        pl.col("names").reverse().alias(("reversed names")),
    ]
)
print(out)
```
```python
shape: (3, 5)
┌────────┬──────┬───────┬────────────┬────────────────┐
│ groups ┆ nrs  ┆ count ┆ random_sum ┆ reversed names │
│ ---    ┆ ---  ┆ ---   ┆ ---        ┆ ---            │
│ str    ┆ i64  ┆ u32   ┆ f64        ┆ list[str]      │
╞════════╪══════╪═══════╪════════════╪════════════════╡
│ B      ┆ 8    ┆ 2     ┆ 0.263315   ┆ [null, "spam"] │
│ A      ┆ 3    ┆ 2     ┆ 0.894213   ┆ ["ham", "foo"] │
│ C      ┆ null ┆ 1     ┆ 0.533739   ┆ ["egg"]        │
└────────┴──────┴───────┴────────────┴────────────────┘
```
#### Numpy interop

Polars expressions support NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html).

This means that if a function is not provided by Polars, we can use NumPy and we still have fast columnar operation through the NumPy API.

**Example** :
```python
import polars as pl
import numpy as np

df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

out = df.select(
    [
        np.log(pl.all()).suffix("_log"),
    ]
)
print(out)
```
```python
shape: (3, 2)
┌──────────┬──────────┐
│ a_log    ┆ b_log    │
│ ---      ┆ ---      │
│ f64      ┆ f64      │
╞══════════╪══════════╡
│ 0.0      ┆ 1.386294 │
│ 0.693147 ┆ 1.609438 │
│ 1.098612 ┆ 1.791759 │
└──────────┴──────────┘
```
### Practical Use Cases
#### Read & Write
```python
df = pl.read_csv("path.csv")
```
CSV files come in many different flavors, so make sure to check the [read_csv()](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_csv.html) API.

Writing to a CSV file can be done with the [write_csv()](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.DataFrame.write_csv.html) method.
```python
df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "bak", "baz"]})
df.write_csv("path.csv")
```
**Scan** :
Polars allows you to scan a CSV input. Scanning delays the actual parsing of the file and instead returns a lazy computation holder called a .LazyFrame
```python
df = pl.scan_csv("path.csv")
```
**Dealing with multiple files** :
Polars can deal with multiple files differently depending on your needs and memory strain.

Let's create some files to give use some context:
```python
import polars as pl

df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "ham", "spam"]})

for i in range(5):
    df.write_csv(f"my_many_files_{i}.csv")
```
**Reading into a single DataFrame** : To read multiple files into a single DataFrame, we can use globbing patterns:
```python
df = pl.read_csv("my_many_files_*.csv")
print(df)
```
```python
shape: (15, 2)
┌─────┬──────┐
│ foo ┆ bar  │
│ --- ┆ ---  │
│ i64 ┆ str  │
╞═════╪══════╡
│ 1   ┆ null │
│ 2   ┆ ham  │
│ 3   ┆ spam │
│ 1   ┆ null │
│ ... ┆ ...  │
│ 3   ┆ spam │
│ 1   ┆ null │
│ 2   ┆ ham  │
│ 3   ┆ spam │
└─────┴──────┘
```
To see how this works we can take a look at the query plan. Below we see that all files are read separately and concatenated into a single DataFrame. Polars will try to parallelize the reading.
```python
pl.scan_csv("my_many_files_*.csv").show_graph()
```
![](file:///C:/Users/Fatoumata%20binta/Downloads/single_df_graph.png)
**Reading and processing in parallel** : If your files don't have to be in a single table you can also build a query plan for each file and execute them in parallel on the Polars thread pool.

All query plan execution is embarrassingly parallel and doesn't require any communication.
```python
import polars as pl
import glob

queries = []
for file in glob.glob("my_many_files_*.csv"):
    q = pl.scan_csv(file).groupby("bar").agg([pl.count(), pl.sum("foo")])
    queries.append(q)

dataframes = pl.collect_all(queries)
print(dataframes)
```
```python
[shape: (3, 3)
┌──────┬───────┬─────┐
│ bar  ┆ count ┆ foo │
│ ---  ┆ ---   ┆ --- │
│ str  ┆ u32   ┆ i64 │
╞══════╪═══════╪═════╡
│ spam ┆ 1     ┆ 3   │
│ ham  ┆ 1     ┆ 2   │
│ null ┆ 1     ┆ 1   │
└──────┴───────┴─────┘, shape: (3, 3)
┌──────┬───────┬─────┐
│ bar  ┆ count ┆ foo │
│ ---  ┆ ---   ┆ --- │
│ str  ┆ u32   ┆ i64 │
╞══════╪═══════╪═════╡
│ spam ┆ 1     ┆ 3   │
│ ham  ┆ 1     ┆ 2   │
│ null ┆ 1     ┆ 1   │
└──────┴───────┴─────┘, shape: (3, 3)
┌──────┬───────┬─────┐
│ bar  ┆ count ┆ foo │
│ ---  ┆ ---   ┆ --- │
│ str  ┆ u32   ┆ i64 │
╞══════╪═══════╪═════╡
│ null ┆ 1     ┆ 1   │
│ ham  ┆ 1     ┆ 2   │
│ spam ┆ 1     ┆ 3   │
└──────┴───────┴─────┘, shape: (3, 3)
┌──────┬───────┬─────┐
│ bar  ┆ count ┆ foo │
│ ---  ┆ ---   ┆ --- │
│ str  ┆ u32   ┆ i64 │
╞══════╪═══════╪═════╡
│ null ┆ 1     ┆ 1   │
│ ham  ┆ 1     ┆ 2   │
│ spam ┆ 1     ┆ 3   │
└──────┴───────┴─────┘, shape: (3, 3)
┌──────┬───────┬─────┐
│ bar  ┆ count ┆ foo │
│ ---  ┆ ---   ┆ --- │
│ str  ┆ u32   ┆ i64 │
╞══════╪═══════╪═════╡
│ null ┆ 1     ┆ 1   │
│ spam ┆ 1     ┆ 3   │
│ ham  ┆ 1     ┆ 2   │
└──────┴───────┴─────┘]
```
#### Selecting data
In this section we show how to select rows and/or columns from a DataFrame. We can [ select data with expressions](https://pola-rs.github.io/polars-book/user-guide/howcani/selecting_data/selecting_data_expressions.html) ou [ select data with square bracket indexing.](https://pola-rs.github.io/polars-book/user-guide/howcani/selecting_data/selecting_data_indexing.html)

**Selecting with expressions** : To select data with expressions we use:
* the filter method to select rows
* the select method to select columns 

For simplicity we deal with DataFrame examples throughout. The principles are the same for Series objects except that columns obviously cannot be selected in a Series. To illustrate the filter and select methods we define a simple DataFrame:
```python
df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "color": ["blue", "red", "green"],
        "size": ["small", "medium", "large"],
    }
)
print(df)
```
```python
shape: (3, 3)
┌─────┬───────┬────────┐
│ id  ┆ color ┆ size   │
│ --- ┆ ---   ┆ ---    │
│ i64 ┆ str   ┆ str    │
╞═════╪═══════╪════════╡
│ 1   ┆ blue  ┆ small  │
│ 2   ┆ red   ┆ medium │
│ 3   ┆ green ┆ large  │
└─────┴───────┴────────┘
```
**Selecting rows with the filter method** : We can select rows by using the filter method. In the filter method we pass the condition we are using to select the rows as an expression
```python
filter_df = df.filter(pl.col("id") <= 2)
print(filter_df)
```
```python
shape: (2, 3)
┌─────┬───────┬────────┐
│ id  ┆ color ┆ size   │
│ --- ┆ ---   ┆ ---    │
│ i64 ┆ str   ┆ str    │
╞═════╪═══════╪════════╡
│ 1   ┆ blue  ┆ small  │
│ 2   ┆ red   ┆ medium │
└─────┴───────┴────────┘
```
**Selecting columns with the select method** : We select columns using the select method. In the select method we can specify the columns with:
* a (string) column name 
* a list of (string) column names 
* a boolean list of the same length as the number of columns 
* an expression such as a condition on the column name 
* a Series 

**Select a list of columns**
```python
list_select_df = df.select(["id", "color"])
print(list_select_df)
```
```python
shape: (3, 2)
┌─────┬───────┐
│ id  ┆ color │
│ --- ┆ ---   │
│ i64 ┆ str   │
╞═════╪═══════╡
│ 1   ┆ blue  │
│ 2   ┆ red   │
│ 3   ┆ green │
└─────┴───────┘
```
**Selecting rows and columns** : We can combine the filter and select methods to select rows and columns
```python
expression_df = df.filter(pl.col("id") <= 2).select(["id", "color"])
print(expression_df)
```
```python
shape: (2, 2)
┌─────┬───────┐
│ id  ┆ color │
│ --- ┆ ---   │
│ i64 ┆ str   │
╞═════╪═══════╡
│ 1   ┆ blue  │
│ 2   ┆ red   │
└─────┴───────┘
```
**Selecting with indexing** : Square bracket indexing can be used to select rows and/or columns.

**Comparison with pandas**

![](file:///C:/Users/Fatoumata%20binta/Downloads/Capture%20d’écran%202023-03-03%20113254.png)
#### Data handling
**null and NaN values** : Polars also allows NotaNumber or NaN values for float columns. These NaN values are considered to be a type of floating point data rather than missing data. We discuss NaN values separately below.

You can manually define a missing value with the python None value:
```python
df = pl.DataFrame(
    {
        "value": [1, None],
    },
)
print(df)
```
```python
shape: (2, 1)
┌───────┐
│ value │
│ ---   │
│ i64   │
╞═══════╡
│ 1     │
│ null  │
└───────┘
```
In Pandas the value for missing data depends on the dtype of the column. In Polars missing data is always represented as a null value.

**Filling missing data** : Missing data in a Series can be filled with the fill_null method. You have to specify how you want the fill_null method to fill the missing data. The main ways to do this are filling with:
* a literal such as 0 or "0" 
* a strategy such as filling forwards 
* an expression such as replacing with values from another column 
* interpolation 

We illustrate each way to fill nulls by defining a simple DataFrame with a missing value in col2:
```python
df = pl.DataFrame(
    {
        "col1": [1, 2, 3],
        "col2": [1, None, 3],
    },
)
print(df)
```
```python
shape: (3, 2)
┌──────┬──────┐
│ col1 ┆ col2 │
│ ---  ┆ ---  │
│ i64  ┆ i64  │
╞══════╪══════╡
│ 1    ┆ 1    │
│ 2    ┆ null │
│ 3    ┆ 3    │
└──────┴──────┘
```
**Fill with an expression** : 
For more flexibility we can fill the missing data with an expression. For example, to fill nulls with the median value from that column
```python
fill_median_df = df.with_columns(
    pl.col("col2").fill_null(pl.median("col2")),
)
print(fill_median_df)
```
```python
shape: (3, 2)
┌──────┬──────┐
│ col1 ┆ col2 │
│ ---  ┆ ---  │
│ i64  ┆ f64  │
╞══════╪══════╡
│ 1    ┆ 1.0  │
│ 2    ┆ 2.0  │
│ 3    ┆ 3.0  │
└──────┴──────┘
```
In this case the column is cast from integer to float because the median is a float statistic.

**Fill with interpolation** :
In addition, we can fill nulls with interpolation (without using the fill_null function)
```python
fill_interpolation_df = df.with_columns(
    pl.col("col2").interpolate(),
)
print(fill_interpolation_df)
```
```python
shape: (3, 2)
┌──────┬──────┐
│ col1 ┆ col2 │
│ ---  ┆ ---  │
│ i64  ┆ i64  │
╞══════╪══════╡
│ 1    ┆ 1    │
│ 2    ┆ 2    │
│ 3    ┆ 3    │
└──────┴──────┘
```
#### Combining data with concat and join
You can combine data from different DataFrames using:
 [the concat function](https://pola-rs.github.io/polars-book/user-guide/howcani/combining_data/concatenating.html) or [the join method ](https://pola-rs.github.io/polars-book/user-guide/howcani/combining_data/joining.html)  on a DataFrame
 
 **Vertical concatenation - getting longer** :
In a vertical concatenation you combine all of the rows from a list of DataFrames into a single longer DataFrame.

```python
df_v1 = pl.DataFrame(
    {
        "a": [1],
        "b": [3],
    }
)
df_v2 = pl.DataFrame(
    {
        "a": [2],
        "b": [4],
    }
)
df_vertical_concat = pl.concat(
    [
        df_v1,
        df_v2,
    ],
    how="vertical",
)
print(df_vertical_concat)
```
```python
shape: (2, 2)
┌─────┬─────┐
│ a   ┆ b   │
│ --- ┆ --- │
│ i64 ┆ i64 │
╞═════╪═════╡
│ 1   ┆ 3   │
│ 2   ┆ 4   │
└─────┴─────┘
```
Vertical concatenation fails when the dataframes do not have the same column names.

**Diagonal concatenation - getting longer, wider and nullier** :
In a diagonal concatenation you combine all of the row and columns from a list of DataFrames into a single longer and/or wider DataFrame.
```python
df_d1 = pl.DataFrame(
    {
        "a": [1],
        "b": [3],
    }
)
df_d2 = pl.DataFrame(
    {
        "a": [2],
        "d": [4],
    }
)

df_diagonal_concat = pl.concat(
    [
        df_d1,
        df_d2,
    ],
    how="diagonal",
)
print(df_diagonal_concat)
```

```python
shape: (2, 3)
┌─────┬──────┬──────┐
│ a   ┆ b    ┆ d    │
│ --- ┆ ---  ┆ ---  │
│ i64 ┆ i64  ┆ i64  │
╞═════╪══════╪══════╡
│ 1   ┆ 3    ┆ null │
│ 2   ┆ null ┆ 4    │
└─────┴──────┴──────┘
```
Diagonal concatenation generates nulls when the column names do not overlap.

Polars is a very large library with many possibilities that make it much easier in terms of speed and performance. We've covered some basics in this course. Refer to the [official documentation](https://pola-rs.github.io/polars-book/user-guide/) to go further. 

#### To go further
1. [Polars, the Fastest Dataframe Library You Never Heard of. - Ritchie Vink | PyData Global 2021](https://www.youtube.com/watch?v=iwGIuGk5nCE&t=2s) 
2. [Loading CSV files with Polars in Python](https://www.youtube.com/watch?v=nGritAo-71o)
3. [Polars: The Next Big Python Data Science Library... written in RUST?](https://www.youtube.com/watch?v=VHqn7ufiilE) 























