use nalgebra::{DMatrix, SymmetricEigen};

use crate::handle_csv::{load_csv, to_csv};

/// Given a matrix of samples X, returns its covariance matrix C.
/// It is done C = Y^T * Y / (N-1), with N the number of samples and
/// Y = (X-M)./S a zero mean, unit standard deviation matrix.
/// Division is pointwise with M, S being the matrices of attributes means and deviations, respectively.
///
/// Parameters
/// ---
/// - data (`&DMatrix<f64>`): Nxd matrix of samples. Each row of the matrix is assumed to be a sample.
///
/// Returns
/// ---
/// - `covariance_matrix` (`DMatrix<f64>`): dxd covariance matrix of samples
/// - `zero_mean_unit_sd` (`DMatrix<f64>`): Nxd matrix with zero mean, unit standard deviation attributes derived from original data.
/// - `mean_matrix` (`DMatrix<f64>`): Nxd matrix with all rows equal to attributes means.
/// - `std_matrix` (`DMatrix<f64>`): Nxd matrix with all rows equal to attributes standard deviation.
pub fn calculate_correlation(
    data: &DMatrix<f64>,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let broadcaster = DMatrix::from_fn(data.nrows(), 1, |_, _| 1f64);
    let mean_matrix = &broadcaster * data.row_mean();
    let std_matrix = &broadcaster * (data.row_variance()).map(|x| x.sqrt());

    // Pointwise arithmethic to get zero mean, unit standard deviation samples
    let zero_mean_unit_sd = (data - &mean_matrix).component_div(&std_matrix);

    // Covariance matrix of preprocessed data (simple product with transpose since data has 0 mean)
    let covariance_matrix = zero_mean_unit_sd.transpose() * &zero_mean_unit_sd
        / ((zero_mean_unit_sd.nrows() - 1) as f64);

    // Original data correlation matrix is the same as the covariance of zero-mean, deviation normalized data
    return (
        covariance_matrix,
        zero_mean_unit_sd,
        mean_matrix,
        std_matrix,
    );
}

/// Runs PCA on the given CSV dataset.
///
/// Parameters
/// ---
/// - `path` (`&str`): Path to the CSV, given relative to this project's root.
/// - `has_headers` (`bool`): Whether the CSV file has headers.
/// - `save_results` (`bool`):
///     Whether to save results, overwriting previous results if existing. All results are CSV files.
///     If folder "`pca`" does not exist inside "`results/`", saving will fail.
///
/// Returns
/// ---
/// None, but the following files may be saved/overwritten in path `resuls/pca/`:
/// - `means.csv`: matrix of attribute averages.
/// - `standard_deviations.csv`: matrix of attribute deviations.
/// - `covariance.csv`: covariance matrix of attributes.
/// - `eigenvalues.csv`: eigenvalues of covariance matrix.
/// - `weights.csv`: weights of eigenvalues of covariance matrix.
/// - `eigenvectors.csv`: eigenvectors of the covariance matrix.
/// - `reconstructions_1d.csv`: reconstructions in original sample space of PCA unidimentional projections.
/// - `reconstructions_2d.csv`: reconstructions in original sample space of PCA bidimentional projections.
pub fn run(path: &str, has_headers: bool, save_results: bool) {
    println!("\n---\nRunning PCA algorithm...\n");

    // Data loading
    let sample_matrix = load_csv(path, has_headers);

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

    println!("\nPCA algorithm done.\n---\n");
}
