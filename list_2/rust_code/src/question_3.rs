use nalgebra::{DMatrix, RowDVector};
use rand::Rng;
use rand::{rngs::StdRng, seq::index::sample, SeedableRng};

use crate::handle_csv::{load_csv, to_csv};
use crate::question_1::calculate_correlation;
use crate::question_2::preprocess_data;

/// Fixed seed for randomization
pub static SEED: u64 = 242104677;

/// Shuffles the given dataset, then splits it in two: one for training and one for testing.
/// The seed in the randomization is fixed to `SEED`.
///
/// Parameters
/// ---
/// - `sample_matrix` (`&DMatrix<f64>`): Sample matrix of original data.
/// - `label_matrix` (`&DMatrix<f64>`): Label matrix of original data.
/// - `test_split` (`f64`): Proportion of the number of samples that must be used in test stage, between 0 and 1.
///
/// Returns
/// ---
/// - `train_samples` (`DMatrix<f64>`): Train samples matrix, given row by row.
/// - `train_labels` (`DMatrix<f64>`): Actual labels of the training set.
/// - `test_samples` (`DMatrix<f64>`): Train samples matrix, given row by row.
/// - `test_labels` (`DMatrix<f64>`): Actual labels of the test set.
fn split_data(
    sample_matrix: &DMatrix<f64>,
    label_matrix: &DMatrix<f64>,
    test_split: f64,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let n = sample_matrix.nrows();

    let mut rng = StdRng::seed_from_u64(SEED);
    let n_random_indeces = sample(&mut rng, n, n).into_vec();

    let train_end: usize = (n as f64 * (1.0 - test_split)) as usize;

    let train_indeces = &n_random_indeces[0..train_end];
    let test_indeces = &n_random_indeces[train_end..n];

    let train_samples = sample_matrix.select_rows(train_indeces.iter());
    let test_samples = sample_matrix.select_rows(test_indeces.iter());
    let train_labels = label_matrix.select_rows(train_indeces.iter());
    let test_labels = label_matrix.select_rows(test_indeces.iter());

    return (train_samples, test_samples, train_labels, test_labels);
}

/// For given weight matrix W and samples X, calculates the regressor's predictions.
/// This is done by softmaxing the product W[1 X]^T,
/// where 1 is  a column vector of 1's added to X to take into account
/// the linear coefficient of the discriminant.
///
/// Parameters
/// ---
/// - `weight_matrix` (`&DMatrix<f64>`): Weight matrix. Each row is associated with a class.
/// - `samples` (`&DMatrix<f64>`): Sample matrix, given row by row.
///
/// Returns
/// ---
/// - `predictions` (`DMatrix<f64>`): Regressor's label predictions.
fn calculate_predictions(weight_matrix: &DMatrix<f64>, samples: &DMatrix<f64>) -> DMatrix<f64> {
    // Handy variables
    let k = weight_matrix.nrows();
    let x = samples.clone().insert_column(0, 1.0);
    let broadcaster = DMatrix::<f64>::from_fn(k, 1, |_, _| 1.0);

    // Prediction softmaxes regression
    let mut regression = weight_matrix * x.transpose();
    regression.apply(|x| *x = x.exp());

    let predictions = regression
        .component_div(&(broadcaster * regression.row_sum()))
        .transpose();

    return predictions;
}

/// Trains the logistic regressor using gradient descent.
///
/// Parameters
/// ---
/// - `train_samples` (`&DMatrix<f64>`): Train samples matrix, given row by row.
/// - `train_labels` (`&DMatrix<f64>`): Groundtruth labels on training set.
/// - `epochs` (`usize`): How many times gradient descent will iterate during training.
/// - `eta` (`f64`): Learning rate of the gradient descent.
///
/// Returns
/// ---
/// - `predictions` (`DMatrix<f64>`): Labels on the training set given by the regressor.
/// - `w`: (`DMatrix<f64>`): Weight matrix used to determine the regressor's predictions.
fn train(
    train_samples: &DMatrix<f64>,
    train_labels: &DMatrix<f64>,
    epochs: usize,
    eta: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    // Handy variables
    let (n, k) = train_labels.shape(); // number of samples, classes
    let d = train_samples.ncols(); // number of attributes per sample

    let mut predictions = DMatrix::zeros(n, k);

    // Fixed seed for random number generators
    let mut rng = StdRng::seed_from_u64(SEED);
    // Weight initialization: between -0.01 e 0.01
    let mut w = 0.01 * DMatrix::from_fn(k, d + 1, |_, _| rng.random::<f64>());

    // Training loop
    for epoch in 0..epochs {
        println!("Running epoch {:?}...", &epoch);

        predictions = calculate_predictions(&w, train_samples);
        let dw: DMatrix<f64> = train_labels
            .row_iter()
            .enumerate()
            .map(|(i, l)| {
                (l - predictions.row(i)).transpose() * train_samples.row(i).insert_column(0, 1.0)
            })
            .sum();
        w += eta * dw;
    }

    return (predictions, w);
}

/// Applies logistic regression to data, both training and testing the model.
/// Original dataset is normalized, shuffled, and split in train and test sets before regression.
///
/// Parameters
/// ---
/// - `sample_matrix` (`&DMatrix<f64>`):
///     Original, assumedly ordered, sample matrix. Rows are samples and columns are attributes.
/// - `label_matrix` (`&DMatrix<f64>`):
///     Label of each sample. Each row is sparse with only one "1", indicating the label given.
/// - `test_split` (`f64`):
///     Proportion of the number of samples that must be used in test stage, between 0 and 1.
/// - `epochs` (`usize`):
///     How many times gradient descent will iterate during training.
/// - `eta` (`f64`):
///     Learning rate of the gradient descent.
///
/// Returns
/// ---
/// - `weight_matrix` (`DMatrix<f64>`): Weight matrix obtained after the training.
/// - `train_predictions` (`DMatrix<f64>`): Labels the regressor assigned to the training set.
/// - `train_labels` (`DMatrix<f64>`): Actual labels of the training set.
/// - `test_predictions` (`DMatrix<f64>`): Labels the regressor assigned to the test set.
/// - `test_labels` (`DMatrix<f64>`): Actual labels of the test set.
fn logistic_regression(
    sample_matrix: &DMatrix<f64>,
    label_matrix: &DMatrix<f64>,
    test_split: f64,
    epochs: usize,
    eta: f64,
) -> (
    DMatrix<f64>,
    DMatrix<f64>,
    DMatrix<f64>,
    DMatrix<f64>,
    DMatrix<f64>,
) {
    let (zero_mean_normalized_samples, _, _) = preprocess_data(sample_matrix);
    let (train_samples, test_samples, train_labels, test_labels) =
        split_data(&zero_mean_normalized_samples, label_matrix, test_split);

    let (train_predictions, weight_matrix) = train(&train_samples, &train_labels, epochs, eta);

    let test_predictions = calculate_predictions(&weight_matrix, &test_samples);

    return (
        weight_matrix,
        train_predictions,
        train_labels,
        test_predictions,
        test_labels,
    );
}

/// Runs logistic regression on the given dataset.
/// Select highly correlated attributes are hard-codedly removed.
/// Logistic regression hyperparameters are also hard coded, with
/// - test split euqal to 20% of original amount of samples;
/// - number of epochs equal to 100;
/// - learning rate equal to 0.1.
///
/// Parameters
/// ---
/// - `path` (`&str`): Path to the CSV file with the dataset and labels.
/// - `has_headers` (`bool`): Whether CSV file has headers.
/// - `save_results` (`bool`):
///     Whether to save results, overwriting previous results if existing. All results are CSV files.
///     If folder "`logistic_regression`" does not exist inside "`results/`", saving will fail.
///
/// Returns
/// ---
/// None, but the following files may be saved/overwritten in path `resuls/logistic_regression/`:
/// - `correlation.csv`: Correlation matrix of original data.
/// - `weight_matrix.csv`: Weight matrix resulting from training.
/// - `train_predictions.csv`: Regressor's predicitons on training set.
/// - `train_labels.csv`:  Actual training set labels.
/// - `test_predictions.csv`: Regressor's predicitons on test set.
/// - `test_labels.csv`: Actual test set labels.
pub fn run(path: &str, has_headers: bool, save_results: bool) {
    println!("\n---\nRunning Logistic regression algorithm...\n");

    // Samples and labels are loaded in the same matrix
    let raw_data = load_csv(path, has_headers);

    // Separating labels from features
    let l = raw_data.ncols();
    let label_column = raw_data.column(l - 1);
    let label_matrix = DMatrix::from_rows(
        &label_column
            .iter()
            .map(|r| {
                RowDVector::from_fn(label_column.max() as usize + 1, |_, j| {
                    // Column 0 is 1 if label is 0 (female),
                    // column 1 is 1 if label is 1 (male)
                    if j == *r as usize {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect::<Vec<_>>(),
    );
    let mut sample_matrix = raw_data.remove_column(l - 1); // label is not an attribute

    // Histograms: done externally

    // Correlation between attributes (same manipulations as in question 1)
    let (correlation_matrix, _, _, _) = calculate_correlation(&sample_matrix);

    // Remove highly correlated attributes
    sample_matrix = sample_matrix.remove_column(17); // maxdom -> column 17 in original data
    sample_matrix = sample_matrix.remove_column(8); // kurt -> column 8 in original data
    sample_matrix = sample_matrix.remove_column(0); // meanfreq -> column 0 in original data

    // Logistic regression
    let test_split: f64 = 0.2;
    let epochs: usize = 100;
    let eta = 0.1;
    let (weight_matrix, train_predictions, train_labels, test_predictions, test_labels) =
        logistic_regression(&sample_matrix, &label_matrix, test_split, epochs, eta);

    // Saving results
    let r: String = "./results/logistic_regression/".to_owned();
    if save_results {
        to_csv(&(r.clone() + "correlation.csv"), &correlation_matrix).unwrap();
        to_csv(&(r.clone() + "weight_matrix.csv"), &weight_matrix).unwrap();
        to_csv(&(r.clone() + "train_predictions.csv"), &train_predictions).unwrap();
        to_csv(&(r.clone() + "train_labels.csv"), &train_labels).unwrap();
        to_csv(&(r.clone() + "test_predictions.csv"), &test_predictions).unwrap();
        to_csv(&(r.clone() + "test_labels.csv"), &test_labels).unwrap();
    }

    // RoC, F1-score, confusion matrix: discussed externally (python_code, report)

    // End of code
    println!("\nLogistic Regression algorithm done.\n---\n");

    return;
}
