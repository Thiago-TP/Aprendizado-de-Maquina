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
    // Whether to save results
    let save_results = true;

    // Quick and easy PCA application
    question_1::run("./data/pca/data_pca.csv", false, save_results);

    // Kmeans application, some case studies may take a while!
    for (image, k) in vec![
        ("cat-10", 15),
        ("cat-101", 15),
        ("cat-110", 15),
        ("flower-6", 10),
        ("flower-14", 10),
        ("flower-23", 10),
        ("horse-137", 10),
        ("horse-139", 15),
        ("horse-170", 15),
    ] {
        question_2::run(image, k, save_results);
    }

    // Quick and easy logictic regression application
    question_3::run(
        "./data/logistic_regression/data_gender_voice.csv",
        true,
        save_results,
    );
}
