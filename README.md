# mnist-extractor

This crate is a simple utility to download and extract MNIST dataset.\
As it is not very optimized, it is not recommended in production but only during testing.

## How to use

To use it, do :

```rust
extern crate mnist-extractor;
use mnist-extractor::*;

let (test_lbl, test_img, train_lbl, train_img) = get_all();

// use datas the way you want
```

The returned datas are `ndarray::Array2<f64>` arrays. Each row of the arrays are either `hot_ones` encoded labels (`10` numbers long) or `flat_encoded` images (`784` numbers long).

If you want to clean the downloaded datas, use :

```rust
mnist-extractor::clean_all_extracted();
```

or

```rust
mnist-extractor::();
```

## Contribution

Contributions are welcome, although this crate is really not intended to be used a lot. It is used in the neural network project [spitz](https://github.com/aunetx/spitz) during its tests.
