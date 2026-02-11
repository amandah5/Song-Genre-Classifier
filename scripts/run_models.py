
from operator import itemgetter

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.naive_bayes import MultinomialNB

import feature_extraction

def main():
    """Prepare feature sets and labels, and return them along with model hyperparameters."""

    # hyperparameters for each model: smoothing (alpha) for NB, L2 inverse reg strength for LR
    possible_k = [0.001, 0.01]
    possible_L2 = [0.002, 0.0035, 0.005]

    # unpacking the vectorized data
    results = feature_extraction.run_feature_extraction()
    features_bin_train = results["features_bin_train"]
    features_bin_dev = results["features_bin_dev"]
    features_bin_test = results["features_bin_test"]
    features_tfidf_train = results["features_tfidf_train"]
    features_tfidf_dev = results["features_tfidf_dev"]
    features_tfidf_test = results["features_tfidf_test"]
    features_mix_train = results["features_mix_train"]
    features_mix_dev = results["features_mix_dev"]
    features_mix_test = results["features_mix_test"]
    features_mega_train = results["features_mega_train"]
    features_mega_dev = results["features_mega_dev"]
    features_mega_test = results["features_mega_test"]

    # lists of dicts that contain the TRUE labels as one key/val pair
    train_songs = results["train_songs"]
    dev_songs = results["dev_songs"]
    test_songs = results["test_songs"]

    # turn these into actual lists of labels
    train_labels_true = [song["genre"] for song in train_songs]
    dev_labels_true = [song["genre"] for song in dev_songs]
    test_labels_true = [song["genre"] for song in test_songs]

    feature_sets = [
        ("binary", features_bin_train, features_bin_dev, features_bin_test),
        ("tfidf", features_tfidf_train, features_tfidf_dev, features_tfidf_test),
        ("mixed", features_mix_train, features_mix_dev, features_mix_test),
        ("mega", features_mega_train, features_mega_dev, features_mega_test),
    ]
    return feature_sets, train_labels_true, dev_labels_true, test_labels_true, possible_k, possible_L2


def run_grid_search(configs, train_labels_true, dev_labels_true, test_labels_true, k_vals, L2_vals):
    """Performs grid search over Naive Bayes and Logistic Regression configurations, evaluates on dev and test sets, and displays the best model's metrics and confusion matrix."""

    results = []

    # only feature set being used for naive bayes: binary n-gram counts
    for smoother in k_vals:
        model = MultinomialNB(alpha=smoother)
        model.fit(configs[0][1], train_labels_true)
        predicted_labels = model.predict(configs[0][2])
        acc = accuracy_score(dev_labels_true, predicted_labels)
        print("binary features; NB model; k = " + str(smoother) + ": accuracy = " + str(acc))
        current_result = {"features": "binary", "model_name": "NB", "model": model, "k_val": str(smoother),
                          "L2": "not used", "dev_features": configs[0][2], "test_features": configs[0][3],
                          "accuracy": acc}
        results.append(current_result)


    # below: everything is for logistic regression
    for feature_set, train_features, dev_features, test_features in configs:
        for reg in L2_vals:
            model = LogisticRegression(C = reg, max_iter = 2000) # default solver is lbfgs
            model.fit(train_features, train_labels_true)
            predicted_labels = model.predict(dev_features)
            acc = accuracy_score(dev_labels_true, predicted_labels)
            print(str(feature_set) + " features; LR model; L2 = " + str(reg) + ": accuracy = " + str(acc))
            current_result = {"features": feature_set, "model_name": "LR", "model": model, "k_val": "not used",
                              "L2": str(reg), "dev_features": dev_features, "test_features": test_features,
                              "accuracy": acc}
            results.append(current_result)


    results_list_sorted = sorted(results, key = itemgetter("accuracy"), reverse = True)
    # ^ reverse to sort largest to smallest accuracy

    # find accuracy of the top 4 configurations, applied to the test set:
    print("\nTEST SET RESULTS ON DEV SET TOP CONFIGURATIONS:\n")
    top_4 = results_list_sorted[:4]
    top_test_acc = 0
    best_config = None
    for config in top_4:
        accuracy = accuracy_score(test_labels_true, config["model"].predict(config["test_features"]))
        print(config["features"] + " features; " + config["model_name"] + "; L2 = " + config["L2"] +
              "; smoothing = " + config["k_val"] + f"; accuracy = {accuracy:.4f}")
        if accuracy > top_test_acc:
            top_test_acc = accuracy
            best_config = config

    final_test_pred = best_config["model"].predict(best_config["test_features"])

    print("\n\nBEST MODEL TEST SET REPORT:\n")
    print("features: " + str(best_config["features"]) + ", model: " + str(best_config["model_name"])
          + ", k_val: " + str(best_config["k_val"]) + ", L2: " + str(best_config["L2"]) + "\n")
    print(classification_report(test_labels_true, final_test_pred, digits = 4))

    # confusion matrix for the best model:
    cm = confusion_matrix(test_labels_true, final_test_pred, labels=["rap", "pop", "rock", "rb", "country"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["rap", "pop", "rock", "rb", "country"])
    disp.plot()
    plt.show()


if __name__ == "__main__":
    feature_sets, train_labels_true, dev_labels_true, test_labels_true, possible_k, possible_L2 = main()
    run_grid_search(feature_sets, train_labels_true, dev_labels_true, test_labels_true,
        possible_k, possible_L2)