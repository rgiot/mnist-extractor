extern crate libflate;
extern crate ndarray;
extern crate reqwest;
use libflate::gzip::Decoder;
use ndarray::prelude::*;

use std::fs;
use std::io;

// ! MNIST
//	DATAs :
// data/t10k-images-idx3-ubyte.gz
// data/t10k-labels-idx1-ubyte.gz
// data/train-images-idx3-ubyte.gz
// data/train-labels-idx1-ubyte.gz

const URL: &str = "http://yann.lecun.com/exdb/mnist/";
const PATH: &str = "./data/";
const TE_LBL: &str = "t10k-labels-idx1-ubyte";
const TE_IMG: &str = "t10k-images-idx3-ubyte";
const TR_LBL: &str = "train-labels-idx1-ubyte";
const TR_IMG: &str = "train-images-idx3-ubyte";

/// Returns a tuple of four Array2<f64> :
///
/// ```
/// use mnist_extractor::*;
///
/// let (test_lbl, test_img, train_lbl, train_img) = get_all();
///
/// // As many images than labels :
/// // Each images is 784 `f64`'s long (for each pixel)
/// // Each label is 10 `f64`'s long (for each number, hot_ones vector)
/// assert_eq!(test_lbl.len() * 784, test_img.len() * 10);
/// assert_eq!(train_lbl.len() * 784, train_img.len() * 10);
///
/// // 10_000 testing datas, 60_000 training datas
/// assert_eq!(10_000 * 784, test_img.len());
/// assert_eq!(60_000 * 784, train_img.len());
/// ```
pub fn get_all() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let te_lbl = read_deflated_labels(maybe_read(TE_LBL)).unwrap();
    let te_img = read_deflated_images(maybe_read(TE_IMG)).unwrap();
    let tr_lbl = read_deflated_labels(maybe_read(TR_LBL)).unwrap();
    let tr_img = read_deflated_images(maybe_read(TR_IMG)).unwrap();
    (te_lbl, te_img, tr_lbl, tr_img)
}

/// Clean every extracted datas, usefull in case of errors :
///
/// ```
/// use mnist_extractor::*;
///
/// clean_all_extracted();
/// ```
pub fn clean_all_extracted() -> io::Result<()> {
    fs::remove_file(PATH.to_owned() + TE_LBL)?;
    fs::remove_file(PATH.to_owned() + TE_IMG)?;
    fs::remove_file(PATH.to_owned() + TR_LBL)?;
    fs::remove_file(PATH.to_owned() + TR_IMG)?;
    Ok(())
}

/// Clean everything, extracted datas and downloaded files, usefull in case of errors.
/// Try first `clean_all_extracted()` as you won't download again everything.
///
/// ```
/// use mnist_extractor::*;
///
/// clean_everything();
/// ```
pub fn clean_everything() -> io::Result<()> {
    clean_all_extracted().unwrap_or(());
    fs::remove_file(PATH.to_owned() + TE_LBL + ".gz")?;
    fs::remove_file(PATH.to_owned() + TE_IMG + ".gz")?;
    fs::remove_file(PATH.to_owned() + TR_LBL + ".gz")?;
    fs::remove_file(PATH.to_owned() + TR_IMG + ".gz")?;
    Ok(())
}

fn maybe_read(name: &str) -> Vec<u8> {
    match fs::read(PATH.to_owned() + name) {
        Ok(f) => f,
        Err(_) => maybe_download(name),
    }
}

fn uncompress_file(file: &mut fs::File, name: &str) -> Vec<u8> {
    println!("UNZIPPING {}", name);
    let mut decoder = Decoder::new(file).unwrap();
    let mut unzipped =
        fs::File::create(PATH.to_owned() + name).expect("failed to create unzipped file");
    io::copy(&mut decoder, &mut unzipped).expect("failed to copy data to unzipped file");
    fs::read(PATH.to_owned() + name)
        .expect("failed to read data from upzipped file we just created")
}

fn maybe_download(name: &str) -> Vec<u8> {
    // check if files exists
    match fs::File::open(PATH.to_owned() + name + ".gz") {
        Ok(mut f) => uncompress_file(&mut f, name),
        Err(_) => {
            download(name);
            uncompress_file(
                &mut fs::File::open(PATH.to_owned() + name + ".gz")
                    .expect("file downloaded but can't read it"),
                name,
            )
        }
    }
}

fn download(name: &str) {
    println!("DOWNLOADING {}", name);
    let url = URL.to_owned() + name + ".gz";

    let mut resp = reqwest::get(url.as_str()).expect("request failed");
    let mut out = fs::File::create(PATH.to_owned() + name + ".gz").expect("failed to create file");

    io::copy(&mut resp, &mut out).expect("failed to copy content");
}

fn read_deflated_labels(file: Vec<u8>) -> Result<Array2<f64>, std::io::Error> {
    println!("PASSED with labels file {:?}", file.len());

    let file: Vec<u8> = Vec::from(&file[8..]);
    let labels = hot_ones(file);
    Ok(labels)
}

fn read_deflated_images(file: Vec<u8>) -> Result<Array2<f64>, std::io::Error> {
    println!("PASSED with images file {:?}", file.len());

    let file: Vec<f64> = file[16..].iter().map(|&e| e as f64).collect();
    let images: Array2<f64> = Array::from_shape_vec((file.len() / 784, 784), file).unwrap();
    Ok(images)
}

fn hot_ones(data: Vec<u8>) -> Array2<f64> {
    let mut hot_vec: Vec<f64> = Vec::new();

    for element in data {
        for i in 0..10 {
            if element == i {
                hot_vec.push(1.);
            } else {
                hot_vec.push(0.);
            }
        }
    }
    Array::from_shape_vec((hot_vec.len() / 10, 10), hot_vec).unwrap()
}
