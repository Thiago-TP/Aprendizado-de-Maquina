// Utils
pub mod handle_csv;

// Question modules
pub mod question_1;
pub mod question_2;
pub mod question_3;

/// Runs PCA, k-means, and logistic regression algorithms to the given datasets, saving the results or not.
/// Default:
/// - all results are saved/overwritten.
/// - PCA is applied to `data_pca.csv` (in `./data/pca/`).
/// - k-means is applied to `flower-6.png` (in `./data/kmeans/`) with 10 clusters.
/// - Logistic regression is applied to `data_gender_voice.csv` (in `./data/logistic_regression/`).
pub fn main() {
    let save_results = true;
    question_1::run("./data/pca/data_pca.csv", false, save_results);
    question_2::run("flower-6", 10, save_results);
    question_3::run(
        "./data/logistic_regression/data_gender_voice.csv",
        true,
        save_results,
    );
}
