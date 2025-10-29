use image;
use nalgebra::DMatrix;
use rand::{rngs::StdRng, seq::index::sample, SeedableRng};

use crate::handle_csv::to_csv;

/// Fixed seed for randomization
pub static SEED: u64 = 242104677;

/// Preprocess a given sample matrix X, making its attributes zero-mean and normalized (between -1 and 1).
/// Attributes are assumed to be columns of the matrix, and its rows the samples.
/// What is done are two pointwise operations: Z = (X-M)./Amax,
/// where M is the matrix of attributes averages and Amax the matrix of attributes maxima, in absolute value.
///
/// Parameters
/// ---
/// - `sample_matrix` (`&DMatrix<f64>`):
///     Sample matrix.
///
/// Returns
/// ---
/// - `preprocessed_data` (`DMatrix<f64>`): Zero-mean, normalized attributes data matrix derived from original sample matrix.
/// - `mean_matrix` (`DMatrix<f64>`): Matrix of attributes averages used to remove the mean.
/// - `max_matrix` (`DMatrix<f64>`): Matrix of attributes maximum (in absolute value) used to normalize the data.
pub fn preprocess_data(sample_matrix: &DMatrix<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let broadcaster = DMatrix::from_fn(sample_matrix.nrows(), 1, |_, _| 1f64);
    let mean_matrix = &broadcaster * sample_matrix.row_mean();
    let row_max = DMatrix::from_iterator(
        1,
        sample_matrix.ncols(),
        sample_matrix.column_iter().map(|col| col.amax()),
    );
    let max_matrix = &broadcaster * row_max;
    let preprocessed_data = (sample_matrix - &mean_matrix).component_div(&max_matrix);
    return (preprocessed_data, mean_matrix, max_matrix);
}

/// Given an iteration's centroids, finds the label of each sample.
///
/// Parameters
/// ---
/// - `x`: (`&DMatrix<f64>`): Data matrix which rows are samples and columns are attributes.
/// - `c`: (`&DMatrix<f64>`): Centroid matrix which rows are centroids and columns are attributes.
///
/// Returns
/// ---
/// - `labels` (`Vec<f64>`): Label of each sample.
/// - `error` (`f64`): Total error in using the assigned labels to reconstuct data.
fn reconstruction_labels(x: &DMatrix<f64>, c: &DMatrix<f64>) -> (Vec<f64>, f64) {
    let reconstruction = DMatrix::from_row_slice(
        x.nrows(),
        c.nrows(),
        &x.row_iter()
            .flat_map(|xi| c.row_iter().map(move |cj| (&xi - &cj).norm()))
            .collect::<Vec<f64>>(),
    );

    let mut labels = Vec::<f64>::new();
    let mut min_errors = Vec::<f64>::new();

    for distances in reconstruction.row_iter() {
        // Transpose necessary since nalgebra's row objects don't implement argmin
        let (label, error) = distances.transpose().argmin();
        labels.push(label as f64);
        min_errors.push(error);
    }

    let error = min_errors.iter().sum();

    return (labels, error);
}

/// Returns the indices of all samples labeled with a given value.
///
/// Parameters
/// ---
/// - `label` (`f64`): Target label value
/// - `labels` (`Vec<f64>`): Samples labels, given sample by sample
///
/// Returns
/// ---
/// - `indices` (`Vec<usize>`): Indices in `labels` that had value `label`.
fn samples_labeled(label: f64, labels: &Vec<f64>) -> Vec<usize> {
    let indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(
            |(index, &value)| {
                if value == label {
                    Some(index)
                } else {
                    None
                }
            },
        )
        .collect();
    return indices;
}

/// Implements the k-means algorithm.
/// Centroids are randomly initialized with seed `SEED`.
///
/// Parameters
/// ---
/// - `sample_matrix` (`&DMatrix<f64>`):
///     Matrix which rows are samples and columns are attributes.
///     Attributes are assumed to be zero-mean and unit standard deviation.
/// - `k` (`usize`):
///     Number of clusters.
/// - `max_iter` (`usize`):
///     Maximum number of iterations if convergence of centroids is not achieved.
/// - `tol` (`f64`):
///     Tolerance on the change of centroids after maximization step.
///     If the change is falls below `tol`, kmeans stops.
///
/// Returns
/// ---
/// - `centroids` (`Vec<DMatrix<f64>>`): The sets of centroids observed at each iteration's expectation step.
/// - `labels` (`Vec<Vec<f64>>`): The sets of labels observed at each iteration's expectation step.
/// - `errors` (`Vec<f64>`): The reconstruction errors associated with the labels history.
fn k_means(
    sample_matrix: &DMatrix<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> (Vec<DMatrix<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    // Handy variables
    let n = sample_matrix.nrows();

    // Centroid initialization from fixed seed
    let mut rng = StdRng::seed_from_u64(SEED);
    let k_random_indeces = sample(&mut rng, n, k).into_vec();
    let mut new_centroids = sample_matrix.select_rows(k_random_indeces.iter());

    // kmeans loop
    let mut centroids = Vec::<DMatrix<f64>>::new(); // centroids at each iteration
    let mut labels = Vec::<Vec<f64>>::new(); // labels at each iteration
    let mut errors = Vec::<f64>::new(); // reconstruction errors ate each iteration
    let mut converged: bool = false;
    let mut iteration: usize = 0;

    while !converged && iteration < max_iter {
        println!("Running iteration {:?}...", &iteration);

        // ====> Expectation step
        centroids.push(new_centroids);
        let (iteration_labels, error) =
            reconstruction_labels(&sample_matrix, &centroids[iteration]);
        labels.push(iteration_labels);
        errors.push(error);

        // ====> Maximization step
        new_centroids = DMatrix::from_rows(
            &(0..k)
                .map(|label| {
                    sample_matrix
                        .select_rows(samples_labeled(label as f64, &labels[iteration]).iter())
                        .row_mean()
                })
                .collect::<Vec<_>>(),
        );

        // Stop conditions: number of iterations maxed out and/or centroids positions update is negligible.
        let delta = &new_centroids - &centroids[iteration];
        let max_change = delta.component_mul(&delta).max().sqrt();
        converged = max_change < tol;
        iteration += 1;
    }

    return (centroids, labels, errors);
}

/// Runs kmeans on the given image, assumed to be of format JPG.
/// Code may not execute properly if image is not JPG.
/// Original dataset's attributes are normalized between -1 and 1 before running the algorithm.
/// Kmeans parameters of stoppage are hard-coded:
/// 50 as a maximum number of iterations,
/// 0.001 as a minimum of centroids change to run another iteration.
///
/// Parameters
/// ---
/// - `case_study` (`&str`): Name to the JPG, minus the extension. Expected to be in folder "`./data/kmeans/".
/// - number_of_clusters (`int`): Number of clusters kmeans will use.
/// - `save_results` (`bool`):
///     Whether to save results, overwriting previous results if existing. All results are CSV files.
///     If folder "`kmeans/{case_study}`" does not exist inside "`results/`", saving will fail.
///
/// Returns
/// ---
/// None, but the following files may be saved/overwritten in path `resuls/pca/`:
/// - `centroids.csv`: centroids of the last expectation step, given row by row.
/// - `labels.csv`: labels of each pixel, given in order as a column.
/// - `errors.csv`: reconstruction error (in normalized, zero mean sample space) of each iteration's expectation step, given as a column.
pub fn run(case_study: &str, number_of_clusters: usize, save_results: bool) {
    println!("K-means case study: {}\n---\n", case_study);

    // Data loading: 3D dataset (i.e., three features per sample)
    let img = image::open(format!("./data/kmeans/{}.jpg", case_study))
        .expect("Image could not be loaded!");
    let bytes = img.into_bytes();
    let img_matrix = DMatrix::from_row_slice(bytes.len() / 3, 3, &bytes);
    let sample_matrix_3d = img_matrix.cast::<f64>();
    let (zero_mean_normalized_samples, means, maxs) = preprocess_data(&sample_matrix_3d);

    // Results of the k-means algorithm
    let (centroid_history, label_history, errors) = k_means(
        &zero_mean_normalized_samples,
        number_of_clusters,
        50,      // default maximum number of iterations
        1e-3f64, // default tolerance on centroids change
    );

    // Saving error results (centroids and labels are saved across iterations of kmeans loop)
    if save_results {
        let (last_centroids, last_labels) = (
            &centroid_history[centroid_history.len() - 1],
            &label_history[label_history.len() - 1],
        );
        // Bring centroids back to original scale and average
        let irows: Vec<usize> = (0..number_of_clusters).collect();
        let rescaled_centroids = &(last_centroids.component_mul(&maxs.select_rows(irows.iter()))
            + &means.select_rows(irows.iter()));

        let r = "./results/kmeans/".to_owned() + case_study;
        if save_results {
            to_csv(&(r.clone() + "/centroids.csv"), &rescaled_centroids).unwrap();
            to_csv(
                &(r.clone() + "/labels.csv"),
                &DMatrix::from_column_slice(last_labels.len(), 1, &last_labels),
            )
            .unwrap();
            to_csv(
                &(r.clone() + "/errors.csv"),
                &DMatrix::from_iterator(errors.len(), 1, errors),
            )
            .unwrap();
        }
    }
    println!("\nkmeans algorithm done.\n---\n");
}
