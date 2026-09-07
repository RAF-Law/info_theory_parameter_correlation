import numpy as np
from npeet import entropy_estimators as ee
from scipy.optimize import minimize
from itertools import combinations

def ee_objective(coeffs, X, y):
    coeffs = np.asarray(coeffs, dtype=float)
    coeffs /= np.linalg.norm(coeffs)
    u = X @ coeffs
    return ee.mi(y, u)

def scipy_objective(coeffs, X, y):
    return -ee_objective(coeffs, X, y)

def build_function_library(X, param_names=None):
    X = np.asarray(X, dtype=float)

    n_samples, n_params = X.shape

    if param_names is None:
        param_names = [f"x{i}" for i in range(n_params)]

    features = []
    feature_names = []

    # Linear
    for i in range(n_params):
        features.append(X[:, i])
        feature_names.append(param_names[i])

    # Square
    for i in range(n_params):
        features.append(X[:, i] ** 2)
        feature_names.append(f"{param_names[i]}^2")

    # Log
    for i in range(n_params):
        features.append(np.log(X[:, i]))
        feature_names.append(f"log({param_names[i]})")

    # Pairwise interaction
    for i in range(n_params):
        for j in range(i + 1, n_params):
            features.append(X[:, i] * X[:, j])
            feature_names.append(
                f"{param_names[i]}*{param_names[j]}"
            )

    Theta = np.column_stack(features)

    return Theta, feature_names

def power_features(
    X,
    y,
    param_names,
    min_size=2,
    max_size=None,
    verbose=True
):
    X = np.asarray(X, dtype=float)

    n_params = X.shape[1]

    if max_size is None:
        max_size = n_params

    features = []
    names = []
    results = []

    for size in range(min_size, max_size + 1):

        for indices in combinations(range(n_params), size):

            X_subset = X[:, indices]

            log_X_subset = np.log(X_subset)

            result = minimize(
                scipy_objective,
                x0=np.ones(size),
                args=(log_X_subset, y),
                method="Powell"
            )

            coeff = result.x.astype(float)

            norm = np.linalg.norm(coeff)

            if norm == 0:
                continue

            coeff /= norm

            idx = np.argmax(np.abs(coeff))

            if coeff[idx] < 0:
                coeff *= -1

            power_feature = np.prod(
                X_subset ** coeff,
                axis=1
            )

            mi = ee.mi(
                power_feature,
                y
            )

            selected_names = [
                param_names[i]
                for i in indices
            ]

            terms = [
                f"{name}^{exponent:.3f}"
                for name, exponent
                in zip(selected_names, coeff)
            ]

            feature_name = " * ".join(terms)

            features.append(power_feature)
            names.append(feature_name)

            results.append({
                "column": len(features) - 1,
                "size": size,
                "indices": indices,
                "parameters": selected_names,
                "exponents": coeff,
                "mi": mi,
                "search_mi": -result.fun,
                "feature_name": feature_name
            })

    results_sorted = sorted(
        results,
        key=lambda x: x["mi"],
        reverse=True
    )

    if verbose:

        print("===== Discovered Power Features =====\n")

        for rank, r in enumerate(
            results_sorted,
            start=1
        ):
            print(
                f"{rank:2d}. "
                f"[{r['size']} vars] "
                f"{r['feature_name']:70s} "
                f"MI = {r['mi']:.6f}"
            )

    return (
        np.column_stack(features),
        names,
        results_sorted
    )

def select_top_power_features(
    Theta_power,
    power_results,
    top_k=10,
    redundancy_threshold=0.98,
):

    selected_columns = []
    selected_results = []

    for r in power_results:

        col = r["column"]
        candidate = Theta_power[:, col]

        keep = True

        for old_col in selected_columns:

            redundancy = ee.mi(
                candidate,
                Theta_power[:, old_col]
            )

            if redundancy >= redundancy_threshold:

                keep = False
                break

        if keep:

            selected_columns.append(col)
            selected_results.append(r)

        if len(selected_results) >= top_k:
            break

    print("\n===== Selected Power Features =====\n")

    for i, r in enumerate(selected_results, 1):

        print(
            f"{i:2d}. "
            f"{r['feature_name']:70s}"
            f"MI = {r['mi']:.6f}"
        )

    return selected_results

def build_power_library(
    X,
    selected_results,
):

    Theta_power = []

    power_names = []

    for r in selected_results:

        feature = np.prod(
            X[:, r["indices"]] ** r["exponents"],
            axis=1,
        )

        Theta_power.append(feature)

        power_names.append(
            r["feature_name"]
        )

    Theta_power = np.column_stack(
        Theta_power
    )

    return Theta_power, power_names