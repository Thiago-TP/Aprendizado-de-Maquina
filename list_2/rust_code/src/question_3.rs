use nalgebra::{DMatrix, RowDVector};
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};

use crate::handle_csv::{load_csv, to_csv};
use crate::question_1::calculate_correlation;
use crate::question_2::preprocess_data;

static SEED: u64 = 242104677;

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
    for _ in 0..epochs {
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

fn logistic_regression(
    sample_matrix: &DMatrix<f64>,
    label_matrix: &DMatrix<f64>,
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
        split_data(&zero_mean_normalized_samples, label_matrix, 0.2);

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

pub fn run() {
    println!("\n---\nRunning Logistic regression algorithm...\n");
    let save_results = true;

    // Samples and labels are loaded in the same matrix
    let raw_data = load_csv("./data/logistic_regression/data_gender_voice.csv", true);

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
                    if j == *r as usize { 1.0 } else { 0.0 }
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
    let epochs: usize = 100;
    let eta = 0.1;
    let (weight_matrix, train_predictions, train_labels, test_predictions, test_labels) =
        logistic_regression(&sample_matrix, &label_matrix, epochs, eta);

    // Saving results
    if save_results {
        to_csv(
            "./results/logistic_regression/correlation.csv",
            &correlation_matrix,
        )
        .unwrap();
        to_csv(
            "./results/logistic_regression/weight_matrix.csv",
            &weight_matrix,
        )
        .unwrap();
        to_csv(
            "./results/logistic_regression/train_predictions.csv",
            &train_predictions,
        )
        .unwrap();
        to_csv(
            "./results/logistic_regression/train_labels.csv",
            &train_labels,
        )
        .unwrap();
        to_csv(
            "./results/logistic_regression/test_predictions.csv",
            &test_predictions,
        )
        .unwrap();
        to_csv(
            "./results/logistic_regression/test_labels.csv",
            &test_labels,
        )
        .unwrap();
    }

    // RoC, F1-score, confusion matrix: discussed externally

    // End of code
    println!("\nLogistic Regression algorithm done.\n---\n");

    return;
}
