# Pandas Cheatsheet

---

## Installation

```bash
pip install pandas

# Check version
python -c "import pandas as pd; print(pd.__version__)"
```

---

## Import Convention

```python
import pandas as pd
import numpy as np
```

---

## Core Data Structures

| Structure | Description | Analogy |
|-----------|-------------|---------|
| `Series` | 1D labeled array | Single column |
| `DataFrame` | 2D labeled table | Spreadsheet / SQL table |
| `Index` | Immutable labels for rows/columns | Row/column headers |

---

## Series

```python
# Create
s = pd.Series([10, 20, 30, 40])
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s = pd.Series({'a': 10, 'b': 20, 'c': 30})
s = pd.Series(5, index=range(5))               # Scalar broadcast

# Properties
s.values        # NumPy array of values
s.index         # Index object
s.dtype         # Data type
s.shape         # (n,)
s.name          # Series name
s.size          # Number of elements

# Access
s[0]            # By position
s['a']          # By label
s[['a', 'c']]   # Multiple labels
s[s > 15]       # Boolean mask
```

---

## DataFrame

### Creating DataFrames
```python
# From dict of lists
df = pd.DataFrame({
    'name':  ['Alice', 'Bob', 'Carol'],
    'age':   [25, 30, 35],
    'score': [88.5, 92.0, 79.3]
})

# From list of dicts
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob',   'age': 30},
])

# From NumPy array
df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])

# From another DataFrame
df2 = df.copy()

# Empty DataFrame
df = pd.DataFrame(columns=['A', 'B', 'C'])
```

### Reading Data
```python
# CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv',
    sep=',',
    header=0,
    index_col='id',
    usecols=['col1', 'col2'],
    dtype={'age': int, 'score': float},
    parse_dates=['date'],
    na_values=['NA', 'N/A', ''],
    nrows=1000,
    skiprows=[1, 2],
    encoding='utf-8',
    chunksize=10000,       # Iterator for large files
)

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)

# JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')

# SQL
import sqlalchemy
engine = sqlalchemy.create_engine('sqlite:///db.sqlite')
df = pd.read_sql('SELECT * FROM table', engine)
df = pd.read_sql_table('table_name', engine)
df = pd.read_sql_query('SELECT * FROM table WHERE age > 30', engine)

# Parquet (fast columnar format)
df = pd.read_parquet('data.parquet')

# HTML tables
dfs = pd.read_html('https://example.com/table.html')

# Clipboard
df = pd.read_clipboard()
```

### Writing Data
```python
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
df.to_json('output.json', orient='records', indent=2)
df.to_parquet('output.parquet', index=False)
df.to_sql('table_name', engine, if_exists='replace', index=False)
df.to_clipboard()
df.to_markdown()           # Markdown table string
```

---

## Inspection

```python
df.head(5)           # First 5 rows (default)
df.tail(5)           # Last 5 rows
df.sample(5)         # Random 5 rows
df.sample(frac=0.1)  # Random 10%

df.shape             # (rows, cols)
df.ndim              # 2
df.size              # rows × cols
df.dtypes            # dtype of each column
df.info()            # Shape, dtypes, non-null counts, memory
df.describe()        # Stats for numeric columns
df.describe(include='all')       # All columns
df.describe(include=['object'])  # Categorical columns

df.columns           # Column Index
df.index             # Row Index
df.values            # NumPy array (avoid for mixed types)
df.to_numpy()        # Preferred over .values

df.memory_usage(deep=True)       # Memory per column
```

---

## Selecting Data

### Columns
```python
df['col']                  # Single column → Series
df[['col1', 'col2']]       # Multiple columns → DataFrame
df.col                     # Attribute access (avoid for ambiguous names)
```

### Rows & Cells
```python
# loc — label-based
df.loc[0]                         # Row by label
df.loc[0:3]                       # Row slice (inclusive!)
df.loc[0, 'col']                  # Single value
df.loc[0:3, 'col1':'col3']        # Row & col slices
df.loc[[0, 2, 4], ['A', 'B']]    # Lists of labels
df.loc[df['age'] > 30]            # Boolean condition

# iloc — position-based
df.iloc[0]                        # First row
df.iloc[-1]                       # Last row
df.iloc[0:3]                      # First 3 rows (exclusive end)
df.iloc[0, 1]                     # Row 0, col 1
df.iloc[0:3, 0:2]                 # Slices by position
df.iloc[[0, 2, 4], [0, 1]]        # Lists of positions

# at / iat — single value (fast)
df.at[0, 'col']                   # Label-based single cell
df.iat[0, 1]                      # Position-based single cell
```

### Boolean Filtering
```python
df[df['age'] > 30]
df[(df['age'] > 25) & (df['score'] < 90)]    # AND
df[(df['age'] > 30) | (df['score'] > 95)]    # OR
df[~(df['age'] > 30)]                        # NOT

df[df['name'].isin(['Alice', 'Bob'])]
df[~df['name'].isin(['Carol'])]

df[df['score'].between(80, 95)]
df[df['name'].str.startswith('A')]
df[df['col'].isna()]                          # NaN rows
df[df['col'].notna()]                         # Non-NaN rows

# query() — readable string syntax
df.query('age > 30 and score < 90')
df.query('name in ["Alice", "Bob"]')
df.query('@threshold < score')                # Reference Python variable
```

---

## Adding & Modifying Data

```python
# Add / modify column
df['new_col'] = 0
df['new_col'] = df['a'] + df['b']
df['category'] = pd.cut(df['score'], bins=[0, 60, 80, 100],
                         labels=['C', 'B', 'A'])

# assign() — method chaining friendly
df = df.assign(
    full_name=df['first'] + ' ' + df['last'],
    age_squared=df['age'] ** 2,
)

# Modify single value
df.at[0, 'col'] = 99
df.loc[df['name'] == 'Alice', 'score'] = 100

# Rename columns
df.rename(columns={'old': 'new', 'a': 'b'}, inplace=True)
df.columns = ['A', 'B', 'C']                  # Rename all

# Rename index
df.index.name = 'id'
df = df.rename_axis('row_id')

# Add row
new_row = pd.DataFrame([{'name': 'Dave', 'age': 28}])
df = pd.concat([df, new_row], ignore_index=True)

# Drop
df.drop(columns=['col1', 'col2'], inplace=True)
df.drop(index=[0, 1], inplace=True)
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['name', 'age'], keep='first')
```

---

## Index Operations

```python
df.set_index('col')                     # Set column as index
df.set_index(['col1', 'col2'])          # Multi-level index
df.reset_index()                        # Move index back to column
df.reset_index(drop=True)              # Discard index

df.sort_index()                         # Sort by index
df.sort_values('col')                   # Sort by column
df.sort_values(['col1', 'col2'], ascending=[True, False])

df.reindex([2, 0, 1])                   # Reorder rows by label
df.reindex(columns=['B', 'A', 'C'])    # Reorder columns

pd.IndexSlice                           # For MultiIndex slicing
```

---

## Missing Data

```python
# Detect
df.isna()                  # Boolean DataFrame
df.isnull()                # Alias of isna()
df.isna().sum()            # Count NaN per column
df.isna().sum().sum()      # Total NaN count
df.isna().mean()           # Fraction missing per column
df.notna()

# Drop
df.dropna()                          # Drop rows with any NaN
df.dropna(axis=1)                    # Drop columns with any NaN
df.dropna(how='all')                 # Drop rows where ALL are NaN
df.dropna(subset=['col1', 'col2'])   # Drop rows with NaN in subset
df.dropna(thresh=3)                  # Keep rows with ≥ 3 non-NaN

# Fill
df.fillna(0)                         # Fill all NaN with 0
df.fillna({'col1': 0, 'col2': 'N/A'})  # Per-column fill
df.fillna(method='ffill')            # Forward fill
df.fillna(method='bfill')            # Backward fill
df.ffill()                           # Shorthand forward fill
df.bfill()                           # Shorthand backward fill
df['col'].fillna(df['col'].median()) # Fill with median

# Interpolate
df.interpolate(method='linear')
df.interpolate(method='time')        # Time-aware (DatetimeIndex)

# Replace
df.replace(np.nan, 0)
df.replace(-999, np.nan)
df.replace({'M': 'Male', 'F': 'Female'})
```

---

## Data Types & Conversion

```python
df['col'].astype(int)
df['col'].astype(float)
df['col'].astype(str)
df['col'].astype('category')          # Memory-efficient for low cardinality

pd.to_numeric(df['col'], errors='coerce')   # Non-numeric → NaN
pd.to_datetime(df['date'])
pd.to_datetime(df['date'], format='%Y-%m-%d')
pd.to_datetime(df['date'], errors='coerce') # Invalid → NaT

# Efficient downcasting
pd.to_numeric(df['col'], downcast='integer')  # int64 → int32/int16
pd.to_numeric(df['col'], downcast='float')

# Category dtype (saves memory, faster groupby)
df['col'] = df['col'].astype('category')
df['col'].cat.categories
df['col'].cat.codes                  # Integer codes
```

---

## String Operations (`.str`)

```python
s = df['name']

s.str.lower(), s.str.upper(), s.str.title()
s.str.strip(), s.str.lstrip(), s.str.rstrip()
s.str.len()
s.str.replace('old', 'new', regex=False)
s.str.replace(r'\d+', '', regex=True)
s.str.contains('pattern', na=False)
s.str.startswith('A'), s.str.endswith('e')
s.str.split(',')                      # → Series of lists
s.str.split(',', expand=True)         # → DataFrame of columns
s.str.join('-')                       # Join list elements
s.str.get(0)                          # First element of list
s.str.extract(r'(\d+)')               # Extract first match group
s.str.extractall(r'(\d+)')            # Extract all matches
s.str.findall(r'\d+')                 # Find all matches → list
s.str.count(r'\d')                    # Count pattern matches
s.str.pad(10, side='left', fillchar='0')
s.str.zfill(5)                        # Zero-pad
s.str.slice(0, 3)                     # s[0:3] for each
s.str.cat(sep=', ')                   # Concatenate all to one string
```

---

## DateTime Operations (`.dt`)

```python
df['date'] = pd.to_datetime(df['date'])

df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.hour, df['date'].dt.minute, df['date'].dt.second
df['date'].dt.dayofweek            # 0=Monday, 6=Sunday
df['date'].dt.day_name()           # 'Monday', 'Tuesday', ...
df['date'].dt.quarter
df['date'].dt.is_month_end
df['date'].dt.is_year_start
df['date'].dt.date                 # Python date object
df['date'].dt.floor('H')           # Floor to hour
df['date'].dt.ceil('D')            # Ceil to day
df['date'].dt.normalize()          # Truncate to midnight

# Arithmetic
df['date'] + pd.Timedelta(days=7)
df['date2'] - df['date1']          # → Timedelta Series
(df['date2'] - df['date1']).dt.days

# Date ranges
pd.date_range('2024-01-01', periods=12, freq='ME')  # Month ends
pd.date_range('2024-01-01', '2024-12-31', freq='D') # Daily
```

---

## GroupBy

```python
g = df.groupby('category')
g = df.groupby(['cat1', 'cat2'])
g = df.groupby('category', observed=True)   # For Categorical dtype

# Aggregation
g['col'].mean()
g['col'].agg(['mean', 'std', 'count'])
g.agg({'col1': 'mean', 'col2': ['min', 'max']})

# Named aggregation (clean column names)
df.groupby('cat').agg(
    avg_score=('score', 'mean'),
    total=('amount', 'sum'),
    n=('id', 'count'),
)

# Common aggregations
g.sum(), g.mean(), g.median()
g.min(), g.max(), g.std(), g.var()
g.count()                          # Non-NaN count
g.size()                           # Group sizes (including NaN)
g.first(), g.last()
g.nth(0)                           # First row of each group
g.nunique()                        # Unique value count

# Transform — same shape as input
df['zscore'] = df.groupby('cat')['score'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['rank'] = df.groupby('cat')['score'].transform('rank')

# Filter — keep groups matching condition
df.groupby('cat').filter(lambda x: x['score'].mean() > 80)

# Apply — arbitrary function per group
df.groupby('cat').apply(lambda x: x.nlargest(3, 'score'))

# Iteration
for name, group in df.groupby('cat'):
    print(name, group.shape)
```

---

## Merging & Joining

```python
# merge — SQL-style joins
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, on=['k1', 'k2'])
pd.merge(df1, df2, left_on='id', right_on='user_id')
pd.merge(df1, df2, on='key', how='inner')   # inner, left, right, outer
pd.merge(df1, df2, on='key', how='left', suffixes=('_x', '_y'))
pd.merge(df1, df2, on='key', indicator=True)  # Adds _merge column

# join — index-based
df1.join(df2)                         # Join on index
df1.join(df2, how='left')
df1.join(df2, lsuffix='_l', rsuffix='_r')

# concat
pd.concat([df1, df2])                 # Stack rows
pd.concat([df1, df2], ignore_index=True)
pd.concat([df1, df2], axis=1)         # Stack columns
pd.concat([df1, df2], keys=['a','b']) # Hierarchical index

# combine_first — fill NaN with other DataFrame's values
df1.combine_first(df2)

# update — modify in-place with non-NaN values from other
df1.update(df2)
```

---

## Reshaping

```python
# Pivot table
df.pivot_table(
    values='score',
    index='student',
    columns='subject',
    aggfunc='mean',
    fill_value=0,
    margins=True,            # Add row/col totals
)

# Simple pivot (no aggregation — values must be unique)
df.pivot(index='date', columns='variable', values='value')

# Melt — wide to long
pd.melt(df, id_vars=['id','name'], value_vars=['jan','feb','mar'],
        var_name='month', value_name='sales')

# Stack / Unstack (MultiIndex)
df.stack()                    # Columns → innermost row index
df.unstack()                  # Innermost row index → columns
df.unstack(level=0)

# Crosstab
pd.crosstab(df['gender'], df['category'])
pd.crosstab(df['gender'], df['category'], normalize='index')  # Row %

# Wide to long
pd.wide_to_long(df, stubnames=['score'], i='id', j='year')
```

---

## Apply & Map

```python
# apply — row or column function
df.apply(np.sum, axis=0)          # Column-wise
df.apply(np.sum, axis=1)          # Row-wise
df['col'].apply(lambda x: x * 2)
df.apply(lambda row: row['a'] + row['b'], axis=1)

# map — element-wise on Series
df['col'].map({'A': 1, 'B': 2, 'C': 3})
df['col'].map(lambda x: x ** 2)

# applymap / map on DataFrame (element-wise)
df.map(lambda x: round(x, 2))     # pandas 2.1+
df.applymap(lambda x: round(x, 2))  # legacy

# pipe — method chaining with functions
df.pipe(clean_fn).pipe(transform_fn, arg=value)

# vectorized alternatives (much faster)
df['col'] * 2                     # Instead of apply(lambda x: x*2)
np.where(df['col'] > 0, 1, -1)    # Instead of apply(if/else)
df['col'].clip(lower=0, upper=100)
```

---

## Window Functions

```python
# Rolling
df['col'].rolling(window=7).mean()          # 7-day rolling mean
df['col'].rolling(window=7, min_periods=1).sum()
df['col'].rolling(7).agg(['mean', 'std'])

# Expanding (cumulative)
df['col'].expanding().mean()                # Running mean
df['col'].cumsum()
df['col'].cumprod()
df['col'].cummax()
df['col'].cummin()

# Exponential weighted
df['col'].ewm(span=10).mean()
df['col'].ewm(alpha=0.3).mean()

# Shift & diff
df['col'].shift(1)                          # Lag by 1
df['col'].shift(-1)                         # Lead by 1
df['col'].diff(1)                           # x[t] - x[t-1]
df['col'].pct_change()                      # (x[t] - x[t-1]) / x[t-1]
```

---

## Sorting & Ranking

```python
df.sort_values('col', ascending=False)
df.sort_values(['col1', 'col2'], ascending=[True, False])
df.sort_values('col', na_position='last')

df['col'].rank()                          # Average rank
df['col'].rank(method='first')            # Rank by first occurrence
df['col'].rank(method='min')              # Min rank for ties
df['col'].rank(ascending=False)           # Descending
df['col'].rank(pct=True)                  # Percentile rank

df.nlargest(5, 'score')                   # Top 5 rows
df.nsmallest(5, 'score')                  # Bottom 5 rows
```

---

## Categorical Data

```python
df['col'] = pd.Categorical(df['col'])
df['col'] = pd.Categorical(df['col'],
    categories=['low','mid','high'], ordered=True)
df['col'] = df['col'].astype('category')

df['col'].cat.categories                  # Category labels
df['col'].cat.codes                       # Integer encoding
df['col'].cat.ordered                     # Is ordered?
df['col'].cat.rename_categories({'low': 'L'})
df['col'].cat.add_categories('new')
df['col'].cat.remove_unused_categories()
df['col'].cat.set_categories(['a','b','c'], ordered=True)

# Useful for:
# - Faster groupby & sorting
# - Correct ordering in plots
# - Memory savings on low-cardinality columns
```

---

## Performance Tips

```python
# Use vectorized ops — avoid loops
df['result'] = df['a'] + df['b']          # Good
df['result'] = df.apply(lambda r: r.a + r.b, axis=1)  # Slow

# Use query() for filtering (faster for large DataFrames)
df.query('age > 30 & score < 90')

# Use categorical for low-cardinality strings
df['status'] = df['status'].astype('category')

# Read only needed columns
pd.read_csv('big.csv', usecols=['a', 'b', 'c'])

# Process large files in chunks
for chunk in pd.read_csv('big.csv', chunksize=10_000):
    process(chunk)

# Use Parquet instead of CSV
df.to_parquet('data.parquet')
pd.read_parquet('data.parquet')

# Avoid chained indexing (triggers SettingWithCopyWarning)
df.loc[mask, 'col'] = val          # Good
df[mask]['col'] = val              # Bad — may not work

# Use .copy() when slicing for independent DataFrames
subset = df[df['age'] > 30].copy()

# eval() for fast column arithmetic
df.eval('c = a + b * 2', inplace=True)
```

---

## MultiIndex

```python
# Create
arrays = [['A','A','B','B'], [1, 2, 1, 2]]
idx = pd.MultiIndex.from_arrays(arrays, names=['letter','number'])
df = pd.DataFrame({'val': [10,20,30,40]}, index=idx)

# Access
df.loc['A']
df.loc[('A', 1)]
df.loc['A':'B']
idx = pd.IndexSlice
df.loc[idx['A', 1:2], :]
df.xs(1, level='number')           # Cross-section

# Manipulate
df.swaplevel()                     # Swap level order
df.sort_index(level=0)
df.reset_index()                   # Flatten to columns
df.droplevel(0)                    # Drop a level
```

---

## Useful Utilities

```python
# Value counts
df['col'].value_counts()
df['col'].value_counts(normalize=True)     # Proportions
df['col'].value_counts(dropna=False)       # Include NaN

# Unique values
df['col'].unique()                         # Array of unique values
df['col'].nunique()                        # Count unique values

# Clip
df['col'].clip(lower=0, upper=100)

# Between
df['col'].between(10, 50)

# Where / mask
df['col'].where(df['col'] > 0, other=0)   # Keep where True, else 0
df['col'].mask(df['col'] < 0, other=0)    # Replace where True

# Binning
pd.cut(df['score'], bins=3)
pd.cut(df['score'], bins=[0,60,80,100], labels=['C','B','A'])
pd.qcut(df['score'], q=4)                 # Quantile-based bins

# Duplicate detection
df.duplicated()
df.duplicated(subset=['name', 'age'], keep='first')

# Cross-tab of missing values
df.isna().sum()

# Pipe for clean chaining
result = (
    df
    .dropna(subset=['score'])
    .query('age >= 18')
    .assign(grade=pd.cut(df['score'], bins=[0,60,80,100],
                          labels=['C','B','A']))
    .groupby('grade')
    .agg(count=('name','count'), avg_score=('score','mean'))
    .sort_values('avg_score', ascending=False)
)

# Display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 120)
pd.reset_option('all')
```

---

## Common Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Chained indexing `df[mask]['col'] = val` | May silently fail | Use `df.loc[mask, 'col'] = val` |
| Modifying a slice | `SettingWithCopyWarning` | Call `.copy()` on slice |
| `==` with NaN | NaN != NaN → always False | Use `.isna()` / `.notna()` |
| Mixed types in column | Slows operations, unexpected behavior | Enforce dtype with `.astype()` |
| `inplace=True` | Can't method-chain, future deprecation | Reassign: `df = df.dropna()` |
| Integer index confusion | `.loc` vs `.iloc` differ on integer index | Be explicit: always use `.loc` or `.iloc` |
| Large CSV in memory | Runs out of RAM | Use `chunksize`, Parquet, or Dask |
| String `object` dtype | Slow string ops | Use `pd.StringDtype()` |
| `.apply()` with axis=1 | Very slow on large frames | Vectorize with NumPy / built-in ops |
| Timezone-naive datetimes | Mixing tz-aware and tz-naive | Use `tz_localize` / `tz_convert` consistently |

---

*Last updated: 2025 · Pandas 2.x · For larger-than-memory data consider Polars, Dask, or DuckDB.*
