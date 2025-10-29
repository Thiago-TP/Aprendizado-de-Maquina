use crate::handle_csv::{load_csv, to_csv};

use nalgebra::{DMatrix, SymmetricEigen};

pub fn calculate_correlation(
    data: &DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let broadcaster = DMatrix::from_fn(data.nrows(), 1, |_, _| 1f64);
    let mean_matrix = &broadcaster * data.row_mean();
    let std_matrix = &broadcaster * (data.row_variance()).map(|x| x.sqrt());

    // Pointwise arithmethic
    let preprocessed_data = (data - &mean_matrix).component_div(&std_matrix);

    // Covariance matrix of preprocessed data (simple product with transpose since data has 0 mean)
    let covariance_matrix = preprocessed_data.transpose() * &preprocessed_data
        / ((preprocessed_data.nrows() - 1) as f64);

    // Original data correlation matrix is the same as the covariance of zero-mean, deviation normalized data
    return (
        covariance_matrix,
        preprocessed_data,
        mean_matrix,
        std_matrix,
    );
}

pub fn run() {
    println!("\n---\nRunning PCA algorithm...");
    let save_results = false;
    // let post_process = false;

    // Data loading
    let sample_matrix = load_csv("./data/pca/data_pca.csv", false);

    // Covariance matrix of zero mean, unit deviation preprocessed data is equal to correlation matrix of original samples
    let (covariance_matrix, preprocessed_data, mean_matrix, std_matrix) =
        calculate_correlation(&sample_matrix);

    // Eigenvalues and eigenvectors of covariance matrix, ordered by magnitude (biggest to smallest)
    let SymmetricEigen {
        eigenvalues: lambdas,
        eigenvectors: vecs,
    } = SymmetricEigen::new(covariance_matrix.clone());
    let weights = &lambdas / lambdas.sum();

    // 1-D and 2-D Reconstructions from projections
    let projections = &preprocessed_data * &vecs;
    // Reconstructions: Z = [(X-m)/d] * W => X = d*(ZW^T) + m
    // Reconstruction from 1D: x1d = d.(z1*w1^T) + m
    let reconstructions_1d = (&projections.column(0) * vecs.column(0).transpose())
        .component_mul(&std_matrix)
        + &mean_matrix;
    // Reconstruction from 2D: x2d = d.([z1 z2] * w1^T) + m
    // or, equivalently: x2d = x1d + d.z2*w2^T
    let reconstructions_2d = &reconstructions_1d
        + (&projections.column(1) * vecs.column(1).transpose()).component_mul(&std_matrix);

    // Saving results
    if save_results {
        to_csv("./results/pca/means.csv", &mean_matrix).unwrap();
        to_csv("./results/pca/standard_deviations.csv", &std_matrix).unwrap();
        to_csv("./results/pca/covariance.csv", &covariance_matrix).unwrap();
        to_csv("./results/pca/eigenvalues.csv", &lambdas).unwrap();
        to_csv("./results/pca/weights.csv", &weights).unwrap();
        to_csv("./results/pca/eigenvectors.csv", &vecs).unwrap();
        to_csv("./results/pca/reconstructions_1d.csv", &reconstructions_1d).unwrap();
        to_csv("./results/pca/reconstructions_2d.csv", &reconstructions_2d).unwrap();
    }

    println!("\nPCA algorithm done.\n---");
}
