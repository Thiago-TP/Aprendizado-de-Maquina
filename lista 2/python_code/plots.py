import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def pca_postprocessing(output_folder: str = "./plots/pca/") -> None:
    """
    Plots PCA reconstructions from 1D and 2D projections.
    """
    # Results loading
    features_names = ["x1", "x2", "x3"]
    results_path = "./../rust_code/results/pca/"
    reconstructions_1d = pd.read_csv(
        results_path + "reconstructions_1d.csv", names=features_names
    )
    reconstructions_2d = pd.read_csv(
        results_path + "reconstructions_2d.csv", names=features_names
    )

    # Original dataset loading
    data_path = "./../rust_code/data/pca/"
    data_set = pd.read_csv(data_path + "data_pca.csv", names=features_names)

    # 3D plot: lower dimentional reconstructions versus original data
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    for data, color in zip(
        [data_set, reconstructions_1d, reconstructions_2d],
        ["black", "tab:blue", "tab:red"],
    ):
        ax.scatter(
            xs=data["x1"],
            ys=data["x2"],
            zs=data["x3"],
            edgecolors=color,
            facecolors="white",
            alpha=0.35,
        )
    for elev in [20, 40, 60]:
        for azim in [-60, -30, 0, 30, 60, 90]:
            ax.view_init(elev=elev, azim=azim)
            fig.savefig(
                output_folder + f"pca_reconstructions_elev{elev}_azim{azim}.png",
                bbox_inches="tight",
                pad_inches=0,
            )


def _reconstruct_image(
    centroids: pd.DataFrame,
    labels: pd.DataFrame,
    original_image: Image.Image,
    output_folder: str,
    case_study: str,
) -> None:
    """
    Reconstructs and saves image from k-means centroids and labels.
    """

    reconstructed_image = centroids.iloc[labels["label"].values, :]
    width, height = original_image.size
    img_shape = (height, width, 3)
    img_array = np.array(reconstructed_image.values.reshape(img_shape).astype(np.uint8))
    img = Image.fromarray(img_array)
    img.save(output_folder + case_study + "/reconstruction.png")


def _plot_elbow_kmeans(
    errors: pd.DataFrame, output_folder: str, case_study: str
) -> None:
    """
    Plots elbow curve of k-means reconstruction errors
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(1, len(errors) + 1),
        errors["error"],
        marker="o",
        linestyle="-",
        color="tab:blue",
    )
    plt.title(f"Case Study: {case_study}", fontsize=24)
    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("Error on $\\mathbf{X}^*$", fontsize=24)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(rotation=45, ha="right", fontsize=20)
    plt.savefig(
        output_folder + case_study + "/elbow_curve.png",
        bbox_inches="tight",
        pad_inches=0,
    )


def _plot_clusters_in_sample_space(
    data_set: pd.DataFrame,
    centroids: pd.DataFrame,
    labels: pd.DataFrame,
    output_folder: str,
    case_study: str,
) -> None:
    """
    Plots k-means clusters in sample space (3D)
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    # Plot samples colored by their labels
    for label in range(len(centroids)):
        # Grabs all samples labeled "label"
        cluster_samples = data_set[(labels == label).values]
        ax.scatter(
            xs=cluster_samples["R"],
            ys=cluster_samples["G"],
            zs=cluster_samples["B"],
            edgecolors=plt.get_cmap("tab20")(label % 20),
            facecolors="white",
            alpha=0.2,
        )
    # Gneerate several views of the same plot
    for elev in [20, 40, 60]:
        for azim in [-60, -30, 0, 30, 60]:
            ax.view_init(elev=elev, azim=azim)
            fig.savefig(
                output_folder + case_study + f"/clusters_elev{elev}_azim{azim}.png",
                bbox_inches="tight",
                pad_inches=0,
            )


def kmeans_postprocessing(
    output_folder: str = "./plots/kmeans/",
    case_study: str = "cat-10",
    fmt: str = ".jpg",
) -> None:
    """
    Plots k-means image reconstructions,
    feature space clusters, and
    reconstruction error elbow curve
    """
    # Results loading
    features_names = ["R", "G", "B"]
    results_path = f"./../rust_code/results/kmeans/{case_study}/"
    labels = pd.read_csv(results_path + "labels.csv", names=["label"]).astype(int)
    centroids = pd.read_csv(results_path + "centroids.csv", names=features_names)
    errors = pd.read_csv(results_path + "errors.csv", names=["error"])

    # Original dataset loading
    data_path = "./../rust_code/data/kmeans/"
    original_image = Image.open(data_path + case_study + fmt)
    data_set = pd.DataFrame(
        np.array(original_image).reshape(-1, 3), columns=features_names
    )

    # Image reconstruction
    _reconstruct_image(centroids, labels, original_image, output_folder, case_study)

    # Plot elbow curve of errors
    _plot_elbow_kmeans(errors, output_folder, case_study)

    # Plot in sample space
    _plot_clusters_in_sample_space(
        data_set, centroids, labels, output_folder, case_study
    )


def _plot_correlation_logistic_regression(output_folder: str) -> None:
    """
    Plots logistic regression dataset correlation matrix.
    """
    attributes_names = (
        pd.read_csv(
            "./../rust_code/data/logistic_regression/attributes_gender_voice.csv",
            index_col="name",
        )
        .drop(index="label", axis=0)  # label is not an attribute
        .transpose()  # attributes come in a column, we want them in a row
    )
    correlations = pd.read_csv(
        "./../rust_code/results/logistic_regression/correlation.csv",
        header=None,
    )
    correlations.columns = attributes_names.columns
    correlations = correlations.T  # plot labels on y axis
    correlations.index.name = ""  # remove y axis name for better display
    g = sns.clustermap(
        correlations,
        cmap="RdBu",
        annot=True,
        # Disabling clustermap goodies since we're just interested in the heatmap
        row_cluster=False,
        col_cluster=False,
        cbar_pos=None,
        dendrogram_ratio=(0, 0),
        xticklabels=False,
        # yticklabels=False,
    )
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=20)
    plt.savefig(
        output_folder + "correlations.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def _plot_histograms_logistic_regression(
    output_folder: str = "./plots/logistic_regression/",
) -> None:
    """
    Plots logistic regression dataset features' histograms.
    """
    attributes_values = pd.read_csv(
        "./../rust_code/data/logistic_regression/data_gender_voice.csv",
    ).drop(columns=["label"])

    color_list = [
        (
            "tab:orange" * (col in ["meanfreq", "centroid"])
            + "tab:green" * (col in ["skew", "kurt"])
            + "tab:red" * (col in ["maxdom", "dfrange"])
        )
        for col in attributes_values.columns
    ]

    for column, color in zip(attributes_values.columns, color_list):
        attributes_values[column].hist(
            figsize=(5, 4),
            color=color if color else "tab:blue",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title(column, fontsize=20)
        plt.tick_params(labelsize=20)
        plt.savefig(
            output_folder + f"histogram-{column}.png",
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close()


def logistic_regression_preprocessing(
    output_folder: str = "./plots/logistic_regression/",
) -> None:
    """
    Plots logistic regression dataset features' histograms,
    correlation matrix.
    """
    _plot_correlation_logistic_regression(output_folder)
    _plot_histograms_logistic_regression(output_folder)


def logistic_regression_postprocessing(
    output_folder: str = "./plots/logistic_regression/",
    thresholds: np.ndarray = np.arange(0.1, 0.91, 0.01),
) -> None:
    """
    Plots logistic regression weight matrix, ROC curve and F1-score.
    """
    _plot_weight_matrix_logistic_regression(output_folder=output_folder)
    _plot_roc_f1_logistic_regression(output_folder=output_folder, thresholds=thresholds)


def _plot_weight_matrix_logistic_regression(
    output_folder: str = "./plots/logistic_regression/",
) -> None:
    """
    Plots logistic regression weight matrix as a colormap.
    """
    weights = pd.read_csv(
        "./../rust_code/results/logistic_regression/weight_matrix.csv",
        header=None,
    )
    attributes_names = (
        pd.read_csv(
            "./../rust_code/data/logistic_regression/attributes_gender_voice.csv",
            index_col="name",
        )
        .transpose()
        .drop(  # weights do not include these
            columns=["kurt", "maxdom", "meanfreq", "label"], axis=0
        )
        # .reset_index(drop=True)
    )

    weights.columns = attributes_names.columns.insert(0, "bias")  # add bias to columns
    weights.columns.name = ""  # remove x axis name for better display
    sns.clustermap(
        weights,
        cmap="RdBu",
        annot=True,
        figsize=(11, 3),
        # Disabling clustermap goodies since we're just interested in the heatmap
        row_cluster=False,
        col_cluster=False,
        cbar_pos=None,
        dendrogram_ratio=(0, 0),
        xticklabels=True,
        yticklabels=False,
    )
    plt.xticks(rotation=45, ha="right", fontsize=15)
    plt.savefig(
        output_folder + "weight_matrix.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def _plot_roc_f1_logistic_regression(
    output_folder: str = "./plots/logistic_regression/",
    thresholds: np.ndarray = np.arange(0.1, 0.91, 0.01),
    th_optimal: float = 0.35,
) -> None:
    """
    Plots logistic regression ROC curve and F1-score for both training and testing..
    """
    for mode, color in zip(["train", "test"], ["tab:blue", "tab:orange"]):
        # Predicted labels (i.e., if sample is male or not)
        predictions = pd.read_csv(
            f"./../rust_code/results/logistic_regression/{mode}_predictions.csv",
            names=["probability_female", "probability_male"],
            usecols=["probability_male"],  # original labels are 1 if male
        )
        groundtruth = pd.read_csv(
            f"./../rust_code/results/logistic_regression/{mode}_labels.csv",
            names=["female_label", "male_label"],
            usecols=["male_label"],  # original labels are 1 if male
        ).astype(int)

        # Build DataFrames
        roc = dict(zip(["threshold", "tpr", "fpr"], [[], [], []]))
        f1 = dict(zip(["threshold", "score"], [[], []]))

        for th in thresholds:

            tp = np.sum(
                (predictions["probability_male"] >= th)
                & (groundtruth["male_label"] == 1)
            )
            fn = np.sum(
                (predictions["probability_male"] < th)
                & (groundtruth["male_label"] == 1)
            )
            fp = np.sum(
                (predictions["probability_male"] >= th)
                & (groundtruth["male_label"] == 0)
            )
            tn = np.sum(
                (predictions["probability_male"] < th)
                & (groundtruth["male_label"] == 0)
            )

            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            if mode == "test" and abs(th - th_optimal) < 0.01:
                confusion_matrix = np.array([[tp, fn], [fp, tn]])

            f1_score = (2 * tp) / (2 * tp + fp + fn)

            f1["threshold"].append(th)
            f1["score"].append(100 * f1_score)

            roc["threshold"].append(th)
            roc["tpr"].append(100 * tpr)
            roc["fpr"].append(100 * fpr)

        # Plots
        fig, axs = plt.subplots(2, 1, figsize=(7, 14), sharex=False, sharey=False)
        style = {
            "legend": None,
            "grid": True,
            "marker": "o",
            "linestyle": "-",
            "color": color,
            "fontsize": 18,
            # "labelsize": 18,
            # "tick_params": {"labelsize": 15},
        }

        f1_df = pd.DataFrame(f1)
        roc_df = pd.DataFrame(roc)

        # F1-Score
        a = f1_df.plot(
            "threshold",
            "score",
            ax=axs[0],
            **style,
        )
        a.set_ylabel("Value (%)", fontsize=18)
        a.set_xlabel("Threshold", fontsize=18)
        a.set_title("F1-Score", fontsize=18)
        # ROC
        b = roc_df.plot(
            "fpr",
            "tpr",
            ax=axs[1],
            **style,
        )
        b.set_ylabel("True Positive Rate (%)", fontsize=18)
        b.set_xlabel("False Positive Rate (%)", fontsize=18)
        b.set_title("ROC", fontsize=18)
        fig.savefig(
            output_folder + f"results-{mode}.png",
            bbox_inches="tight",
            pad_inches=0,
        )

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Actual", fontsize=16)
    plt.savefig(
        output_folder + f"results-confusion_matrix.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(
        confusion_matrix
    )
    print(f"Overall accuracy for threshold {th_optimal}: {accuracy*100:.2f} %")


if __name__ == "__main__":
    # pca_postprocessing()

    # args = [ # uncomment study cases to process
    #     ("cat-10", ".jpg"),
    #     ("cat-101", ".jpg"),
    #     ("cat-110", ".jpg"),
    #     ("flower-6", ".jpg"),
    #     ("flower-14", ".jpg"),
    #     ("flower-23", ".jpg"),
    #     ("horse-137", ".jpg"),
    #     ("horse-139", ".jpg"),
    #     ("horse-170", ".jpg"),
    # ]

    # for arg in args:
    #     case_study, fmt = arg
    #     kmeans_postprocessing(case_study=case_study, fmt=fmt)

    # logistic_regression_preprocessing()
    logistic_regression_postprocessing()
