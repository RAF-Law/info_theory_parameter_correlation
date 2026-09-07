import numpy as np
import matplotlib.pyplot as plt
import pickle
from npeet import entropy_estimators as ee
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from .function_library import build_function_library


def visualize_sparse_prediction(
    Theta,
    y,
    model_file="sparse_result_lv_max.pkl",
    bins=60,
    alpha=0.35,
):
    """
    Load sparse model and compare prediction with ground truth.

    Parameters
    ----------
    Theta : ndarray
        Function library.

    y : ndarray
        Ground truth.

    model_file : str
        Saved sparse_result.

    Returns
    -------
    u : ndarray
    """

    # ----------------------------
    # load model
    # ----------------------------
    with open(model_file, "rb") as f:
        sparse_result = pickle.load(f)

    coeff = sparse_result["coeff"]

    if Theta.shape[1] != len(coeff):
        raise ValueError(
            f"Theta has {Theta.shape[1]} columns "
            f"but model expects {len(coeff)}."
        )

    # ----------------------------
    # prediction
    # ----------------------------
    u = Theta @ coeff

    # ----------------------------
    # metrics
    # ----------------------------
    mi = ee.mi(u, y)
    corr = np.corrcoef(u, y)[0, 1]

    print(f"MI          : {mi:.6f}")

    # ----------------------------
    # plotting
    # ----------------------------
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        y,
        u,
        s=3,
        alpha=alpha
    )

    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted u")
    ax.set_title("Scatter")

    plt.tight_layout()
    plt.show()

    return u

def build_theta_from_model(
    X,
    sparse_result,
    param_names,
):
    """
    Rebuild the complete function library exactly as used in training.
    """

    # ---------- Base library ----------
    Theta_base, _ = build_function_library(
        X,
        param_names
    )

    return Theta_base

def analyze_feature_cmi(
    sparse_result,
    Theta,
    y,
    print_result=True,
):

    coeff = sparse_result["coeff"]
    active_indices = sparse_result["active_indices"]
    feature_names = sparse_result["feature_names"]

    Theta_active = Theta[:, active_indices]

    results = []

    print("\n===== Conditional Mutual Information =====\n")

    for i in range(len(active_indices)):

        xi = Theta_active[:, i]

        other_idx = [j for j in range(len(active_indices)) if j != i]

        if len(other_idx) == 0:
            cmi = ee.mi(xi, y)
        else:
            z = Theta_active[:, other_idx]
            cmi = ee.cmi(xi, y, z)

        mi = ee.mi(xi, y)

        results.append({
            "feature": feature_names[i],
            "coefficient": coeff[i],
            "mi": mi,
            "cmi": cmi,
            "redundancy": mi - cmi
        })

    results = sorted(results, key=lambda r: r["cmi"], reverse=True)

    if print_result:

        print(
            f"{'Feature':45s}"
            f"{'Coeff':>10s}"
            f"{'MI':>12s}"
            f"{'CMI':>12s}"
            f"{'Redundant':>12s}"
        )

        for r in results:

            print(
                f"{r['feature'][:45]:45s}"
                f"{r['coefficient']:10.4f}"
                f"{r['mi']:12.4f}"
                f"{r['cmi']:12.4f}"
                f"{r['redundancy']:12.4f}"
            )

    return results

def analyze_feature_importance(
    sparse_result,
    Theta,
    y,
    print_result=True,
):
    coeff = np.asarray(sparse_result["coeff"])
    active_indices = np.asarray(sparse_result["active_indices"])
    feature_names = sparse_result["feature_names"]

    Theta_active = Theta[:, active_indices]

    # full model
    u_full = Theta_active @ coeff
    full_mi = ee.mi(u_full, y)

    results = []

    for i in range(len(coeff)):

        keep = np.ones(len(coeff), dtype=bool)
        keep[i] = False

        # only one feature
        if keep.sum() == 0:
            mi_without = 0.0
            delta = full_mi

        else:

            u_without = Theta_active[:, keep] @ coeff[keep]

            mi_without = ee.mi(
                u_without,
                y
            )

            delta = full_mi - mi_without

        results.append({

            "feature": feature_names[i],

            "coefficient": coeff[i],

            "full_mi": full_mi,

            "mi_without": mi_without,

            "delta_mi": delta,

        })

    results.sort(
        key=lambda x: x["delta_mi"],
        reverse=True
    )

    if print_result:

        print("\n===== Feature Importance (Leave-One-Out MI) =====\n")

        print(
            f"{'Feature':45s}"
            f"{'Coeff':>10s}"
            f"{'ΔMI':>12s}"
            f"{'MI(-f)':>12s}"
        )

        print("-"*82)

        for r in results:

            print(
                f"{r['feature'][:45]:45s}"
                f"{r['coefficient']:10.4f}"
                f"{r['delta_mi']:12.4f}"
                f"{r['mi_without']:12.4f}"
            )

        print("-"*82)
        print(f"Full model MI : {full_mi:.6f}")

    return results