use image;
use nalgebra::DMatrix;
use rand::{SeedableRng, rngs::StdRng, seq::index::sample};

use crate::handle_csv::to_csv;

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

    return (labels, min_errors.iter().sum());
}

fn samples_labeled(label: f64, labels: &Vec<f64>) -> Vec<usize> {
    let indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(
            |(index, &value)| {
                if value == label { Some(index) } else { None }
            },
        )
        .collect();
    return indices;
}

fn k_means(
    sample_matrix: &DMatrix<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> (Vec<DMatrix<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    // Handy variables
    let n = sample_matrix.nrows();

    // Centroid initialization from fixed seed
    let mut rng = StdRng::seed_from_u64(242104677);
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

        if max_change < tol {
            converged = true; // new centroids after convergence are not included
            println!("Centroids converged after {:?} iterations!", iteration);
        }

        iteration += 1;

        if iteration == max_iter && !converged {
            println!(
                "Centroids did not converge even after {:?} iterations!",
                iteration
            );
        }
    }

    return (centroids, labels, errors);
}

pub fn run() {
    let save_results = false;

    let args = vec![
        // ("cat-10", 15),
        // ("cat-101", 15),
        // ("cat-110", 15),
        ("flower-6", 10),
        // ("flower-14", 10),
        // ("flower-23", 10),
        // ("horse-137", 10),
        // ("horse-139", 15),
        // ("horse-170", 15),
    ];

    for (case_study, number_of_clusters) in args {
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
            50,
            1e-3f64,
        );

        // Saving error results (centroids and labels are saved across iterations of kmeans loop)
        if save_results {
            let (last_centroids, last_labels) = (
                &centroid_history[centroid_history.len() - 1],
                &label_history[label_history.len() - 1],
            );
            // Bring centroids back to original scale and average
            let irows: Vec<usize> = (0..number_of_clusters).collect();
            let rescaled_centroids = &(last_centroids
                .component_mul(&maxs.select_rows(irows.iter()))
                + &means.select_rows(irows.iter()));
            to_csv(
                &format!("./results/kmeans/{}/centroids.csv", case_study),
                &rescaled_centroids,
            )
            .unwrap();
            to_csv(
                &format!("./results/kmeans/{}/labels.csv", case_study),
                &DMatrix::from_column_slice(last_labels.len(), 1, &last_labels),
            )
            .unwrap();
            to_csv(
                &format!("./results/kmeans/{}/errors.csv", case_study),
                &DMatrix::from_iterator(errors.len(), 1, errors),
            )
            .unwrap();
        }
        println!("\nkmeans algorithm done.\n---\n");
    }
}
