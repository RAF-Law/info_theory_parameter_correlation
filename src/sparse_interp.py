import numpy as np
import pickle
import os
from npeet import entropy_estimators as ee
from scipy.optimize import minimize

def ee_objective(coeffs, X, y):
    coeffs = np.asarray(coeffs, dtype=float)
    coeffs /= np.linalg.norm(coeffs)
    u = X @ coeffs
    return ee.mi(y, u)

def scipy_objective(coeffs, X, y):
    return -ee_objective(coeffs, X, y)

def sparse_ee_interpretation(
    Theta,
    y,
    feature_names,
    scaler,
    library_power_results=None,
    threshold=0.1,
    max_iter=10,
    method="Powell",
    checkpoint_file="sparse_checkpoint.pkl",
    resume=False,
    save_every_iteration=True,
):
    Theta = np.asarray(Theta)
    y = np.asarray(y)

    if y.ndim not in (1, 2):
        raise ValueError(
            f"y must have shape (N,) or (N, M), got {y.shape}"
        )

    if Theta.shape[0] != y.shape[0]:
        raise ValueError(
            f"Theta and y have inconsistent number of samples: "
            f"{Theta.shape[0]} vs {y.shape[0]}"
        )

    n_outputs = 1 if y.ndim == 1 else y.shape[1]

    if n_outputs > 1:
        print(f"Multi-output target detected: y shape = {y.shape}")
    else:
        print(f"Single-output target detected: y shape = {y.shape}")

    if library_power_results is None:
        library_power_results = []

    if resume and os.path.exists(checkpoint_file):

        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        iteration_start = checkpoint["iteration"] + 1
        active_indices = checkpoint["active_indices"]

        # Restore history if available
        history = checkpoint.get("history", [])

        print(f"Resume from iteration {iteration_start}")

    else:

        iteration_start = 0
        history = []
        active_indices = np.arange(Theta.shape[1])

    def calculate_gamma(z, y):

        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)

        # Center z
        z_centered = z - np.mean(z)

        # Single output
        if y.ndim == 1:

            y_centered = y - np.mean(y)

            denominator = np.sum(z_centered ** 2)

            if denominator == 0:
                return 0.0

            gamma = (
                np.sum(z_centered * y_centered)
                / denominator
            )

            return gamma

        # Multi-output
        else:

            y_centered = y - np.mean(y, axis=0)

            denominator = np.sum(z_centered ** 2)

            if denominator == 0:
                return 0.0

            # Sum covariance-like contributions from all outputs
            numerator = np.sum(
                z_centered[:, None] * y_centered
            )

            gamma = numerator / denominator

            return gamma

    for iteration in range(iteration_start, max_iter):

        Theta_active = Theta[:, active_indices]

        try:

            result = minimize(
                scipy_objective,
                x0=np.ones(len(active_indices)),
                args=(Theta_active, y),
                method=method
            )

        except KeyboardInterrupt:

            checkpoint = {
                "iteration": iteration - 1,
                "active_indices": active_indices,
                "threshold": threshold,
                "method": method,
                "history": history,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            print("\nTraining interrupted.")
            print(f"Checkpoint saved to {checkpoint_file}")

            return

        coeff = result.x.astype(float)

        coeff_norm = np.linalg.norm(coeff)

        if coeff_norm == 0:
            raise ValueError(
                "Optimisation returned zero coefficient vector."
            )

        coeff /= coeff_norm

        z = Theta_active @ coeff

        gamma = calculate_gamma(z, y)

        coeff *= gamma

        idx = np.argmax(np.abs(coeff))

        if coeff[idx] < 0:
            coeff *= -1

        history.append({
            "iteration": iteration + 1,
            "mi": -result.fun,
            "coeff": coeff.copy(),
            "active_indices": active_indices.copy(),
            "feature_names":
                np.array(feature_names)[active_indices].tolist(),
            "u": z.copy(),
        })

        print(f"\n===== Iteration {iteration + 1} =====")
        print(f"MI              : {-result.fun:.6f}")
        print(f"Active features : {len(active_indices)}")

        keep = np.abs(coeff) >= threshold

        # Prevent removing all features
        if not np.any(keep):
            keep[np.argmax(np.abs(coeff))] = True

        for name, c, selected in zip(
            np.array(feature_names)[active_indices],
            coeff,
            keep
        ):

            status = "KEEP" if selected else "REMOVE"

            print(
                f"{name:35s}: "
                f"{c: .6f}  {status}"
            )

        if np.all(keep):
            break

        active_indices = active_indices[keep]

        if save_every_iteration:

            checkpoint = {
                "iteration": iteration,
                "active_indices": active_indices,
                "threshold": threshold,
                "method": method,
                "history": history,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

    Theta_final = Theta[:, active_indices]

    final_result = minimize(
        scipy_objective,
        x0=np.ones(len(active_indices)),
        args=(Theta_final, y),
        method=method
    )

    final_coeff = final_result.x.astype(float)

    final_norm = np.linalg.norm(final_coeff)

    if final_norm == 0:
        raise ValueError(
            "Final optimisation returned zero coefficient vector."
        )

    final_coeff /= final_norm

    z = Theta_final @ final_coeff

    gamma = calculate_gamma(z, y)

    final_coeff *= gamma

    idx = np.argmax(np.abs(final_coeff))

    if final_coeff[idx] < 0:
        final_coeff *= -1

    final_names = [
        feature_names[i]
        for i in active_indices
    ]

    print("\n===== Final Sparse Combination =====")
    print(f"Maximum MI : {-final_result.fun:.6f}")
    print(f"Terms      : {len(active_indices)}")

    terms = []

    for name, c in zip(final_names, final_coeff):

        print(
            f"{name:35s}: "
            f"{c:.6f}"
        )

        terms.append(
            f"{c:.4f}*{name}"
        )

    print("\nu ∝")
    print(" + ".join(terms))

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return {
        "coeff": final_coeff,
        "library_feature_names": feature_names,
        "feature_names": final_names,
        "active_indices": active_indices,
        "library_power_results": library_power_results,
        "scaler": scaler,
        "mi": -final_result.fun,
        "result": final_result,
        "history": history,
    }

def refine_sparse_result(
    sparse_result,
    Theta,
    y,
    coefficient_threshold=0.1,
    importance_threshold=0,
    method="Powell",
    print_result=True,
):
    """
    Refine a sparse EE model using coefficient pre-screening
    followed by iterative leave-one-out feature pruning.

    The procedure is:

        1. Start from an existing sparse regression result.
        2. Remove features with small absolute coefficients.
        3. Re-optimise the remaining features.
        4. Calculate leave-one-out MI contribution for every
           remaining feature without re-optimisation.
        5. If the smallest contribution is below
           importance_threshold, remove only that feature.
        6. Re-optimise the remaining features and repeat.
        7. Stop when all remaining features have contribution
           >= importance_threshold.

    Parameters
    ----------
    sparse_result : dict
        Result returned by sparse_ee_interpretation.

    Theta : array-like
        Scaled full function library used for the sparse model.

    y : array-like
        Target variable.

    coefficient_threshold : float
        Features with abs(coefficient) below this value are
        removed during the initial screening.

    importance_threshold : float
        Iterative refinement stops when every remaining feature
        has delta_mi >= this value.

    method : str
        Optimisation method passed to scipy.optimize.minimize.

    print_result : bool
        Whether to print the refinement results.

    Returns
    -------
    refined_result : dict
        Updated sparse result.
    """

    Theta = np.asarray(Theta)
    y = np.asarray(y)

    # ==========================================================
    # Get original sparse model
    # ==========================================================

    coeff = np.asarray(
        sparse_result["coeff"],
        dtype=float
    )

    original_active_indices = np.asarray(
        sparse_result["active_indices"]
    ).copy()

    active_indices = original_active_indices.copy()

    feature_names = sparse_result[
        "library_feature_names"
    ]

    scaler = sparse_result["scaler"]

    library_power_results = sparse_result.get(
        "library_power_results",
        []
    )

    # ==========================================================
    # Helper: optimise current active features
    # ==========================================================

    def optimise_active(active_indices,x0=None):

        Theta_active = Theta[:, active_indices]

        if x0 is None:
            x0 = np.ones(len(active_indices))

        result = minimize(
            scipy_objective,
            x0=x0,
            args=(Theta_active, y),
            method=method
        )

        coeff = result.x.astype(float)

        coeff /= np.linalg.norm(coeff)

        u = Theta_active @ coeff

        gamma = (
            np.cov(u, y, bias=True)[0, 1]
            / np.var(u)
        )

        coeff *= gamma

        # Fix sign
        idx = np.argmax(np.abs(coeff))

        if coeff[idx] < 0:
            coeff *= -1

        mi = ee.mi(u, y)

        return {
            "result": result,
            "coeff": coeff,
            "u": u,
            "mi": mi,
            "Theta_active": Theta_active,
        }

    # ==========================================================
    # Step 1: Coefficient pre-screening
    # ==========================================================

    coefficient_keep = (
        np.abs(coeff) >= coefficient_threshold
    )

    # Always keep at least one feature
    if not np.any(coefficient_keep):

        coefficient_keep[
            np.argmax(np.abs(coeff))
        ] = True

    active_indices = active_indices[
        coefficient_keep
    ]

    if print_result:

        print(
            "\n===== Coefficient Pre-screening ====="
        )

        print(
            f"Original features : "
            f"{len(coeff)}"
        )

        print(
            f"Remaining features: "
            f"{len(active_indices)}"
        )

        print()

        for name, c, keep in zip(
            np.array(feature_names)[
                original_active_indices
            ],
            coeff,
            coefficient_keep
        ):

            status = "KEEP" if keep else "REMOVE"

            print(
                f"{name:35s}: "
                f"{c: .6f}  {status}"
            )

    # ==========================================================
    # Step 2: Initial re-optimisation
    # ==========================================================

    coeff = coeff[coefficient_keep]

    current_model = optimise_active(
        active_indices,
        x0=coeff
    )

    full_mi = current_model["mi"]

    # ==========================================================
    # Step 3: Iterative leave-one-out refinement
    # ==========================================================

    refinement_history = []

    iteration = 0

    while len(active_indices) > 1:

        iteration += 1

        Theta_active = Theta[:, active_indices]

        feature_results = []

        # ------------------------------------------------------
        # Calculate leave-one-out contribution
        # ------------------------------------------------------

        for i in range(len(active_indices)):

            keep = np.ones(
                len(active_indices),
                dtype=bool
            )

            keep[i] = False

            u_without = (
                Theta_active[:, keep]
                @ coeff[keep]
            )

            mi_without = ee.mi(
                u_without,
                y
            )

            delta_mi = (
                full_mi
                - mi_without
            )

            feature_results.append({
                "feature":
                    feature_names[
                        active_indices[i]
                    ],
                "library_index":
                    active_indices[i],
                "coefficient":
                    coeff[i],
                "mi_without":
                    mi_without,
                "delta_mi":
                    delta_mi,
            })

        # ------------------------------------------------------
        # Find feature with smallest contribution
        # ------------------------------------------------------

        worst_idx = np.argmin([
            r["delta_mi"]
            for r in feature_results
        ])

        worst_feature = feature_results[
            worst_idx
        ]

        min_delta_mi = worst_feature[
            "delta_mi"
        ]

        if print_result:

            print(
                f"\n===== Refinement Iteration "
                f"{iteration} ====="
            )

            print(
                f"Current MI : "
                f"{full_mi:.6f}"
            )

            print()

            for i, r in enumerate(
                feature_results
            ):

                marker = (
                    "  <-- lowest"
                    if i == worst_idx
                    else ""
                )

                print(
                    f"{r['feature']:35s}: "
                    f"coeff={r['coefficient']: .6f}  "
                    f"MI(-f)={r['mi_without']: .6f}  "
                    f"ΔMI={r['delta_mi']: .6f}"
                    f"{marker}"
                )

        # ------------------------------------------------------
        # Stop if every feature is important enough
        # ------------------------------------------------------

        if min_delta_mi >= importance_threshold:

            if print_result:

                print(
                    "\nAll remaining features have "
                    f"ΔMI >= {importance_threshold:.6f}."
                )

                print(
                    "Refinement stopped."
                )

            break

        # ------------------------------------------------------
        # Remove only the worst feature
        # ------------------------------------------------------

        removed_feature = (
            feature_names[
                active_indices[worst_idx]
            ]
        )

        refinement_history.append({
            "iteration":
                iteration,
            "full_mi":
                full_mi,
            "feature_results":
                feature_results,
            "removed_feature":
                removed_feature,
            "removed_index":
                active_indices[worst_idx],
            "removed_delta_mi":
                min_delta_mi,
        })

        if print_result:

            print()

            print(
                f"Removing: "
                f"{removed_feature}"
            )

            print(
                f"ΔMI = "
                f"{min_delta_mi:.6f}"
            )

        keep = np.ones(
            len(active_indices),
            dtype=bool
        )
        keep[worst_idx] = False

        x0 = coeff[keep]

        active_indices = active_indices[
            keep
        ]

        # ------------------------------------------------------
        # Re-optimise after removing one feature
        # ------------------------------------------------------

        current_model = optimise_active(
            active_indices,
            x0=x0
        )

        coeff = current_model["coeff"]

        full_mi = current_model["mi"]

        if print_result:

            print(
                f"MI after re-optimisation: "
                f"{full_mi:.6f}"
            )

    # ==========================================================
    # Final model
    # ==========================================================

    final_result = current_model["result"]

    final_coeff = current_model["coeff"]

    final_mi = current_model["mi"]

    final_names = [
        feature_names[i]
        for i in active_indices
    ]

    # ==========================================================
    # Final feature importance
    # ==========================================================

    final_feature_results = []

    Theta_final = Theta[:, active_indices]

    if len(active_indices) == 1:

        final_feature_results.append({
            "feature":
                final_names[0],
            "library_index":
                active_indices[0],
            "coefficient":
                final_coeff[0],
            "mi_without":
                0.0,
            "delta_mi":
                final_mi,
        })

    else:

        for i in range(len(active_indices)):

            keep = np.ones(
                len(active_indices),
                dtype=bool
            )

            keep[i] = False

            u_without = (
                Theta_final[:, keep]
                @ final_coeff[keep]
            )

            mi_without = ee.mi(
                u_without,
                y
            )

            delta_mi = (
                final_mi
                - mi_without
            )

            final_feature_results.append({
                "feature":
                    final_names[i],
                "library_index":
                    active_indices[i],
                "coefficient":
                    final_coeff[i],
                "mi_without":
                    mi_without,
                "delta_mi":
                    delta_mi,
            })

    # ==========================================================
    # Print final result
    # ==========================================================

    if print_result:

        print(
            "\n===== Refined Sparse Combination ====="
        )

        print(
            f"Initial terms : "
            f"{len(original_active_indices)}"
        )

        print(
            f"Final terms   : "
            f"{len(active_indices)}"
        )

        print(
            f"Final MI      : "
            f"{final_mi:.6f}"
        )

        print()

        terms = []

        for name, c in zip(
            final_names,
            final_coeff
        ):

            print(
                f"{name:35s}: "
                f"{c:.6f}"
            )

            terms.append(
                f"{c:.4f}*{name}"
            )

        print("\nu ∝")
        print(" + ".join(terms))

    # ==========================================================
    # Return refined result
    # ==========================================================

    return {
        "coeff":
            final_coeff,
        "library_feature_names":
            feature_names,
        "feature_names":
            final_names,
        "active_indices":
            active_indices,
        "library_power_results":
            library_power_results,
        "scaler":
            scaler,
        "mi":
            final_mi,
        "result":
            final_result,
        "feature_importance":
            final_feature_results,
        "coefficient_keep":
            coefficient_keep,
        "refinement_history":
            refinement_history,
        "original_sparse_result":
            sparse_result,
    }

def save_sparse_result(sparse_result, filename="sparse_result.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(sparse_result, f)

    print(f"Saved to {filename}")
    print(f"MI = {sparse_result['mi']:.6f}")
    print(f"Active features = {len(sparse_result['coeff'])}")
    
def load_sparse_result(filename="sparse_result.pkl"):
    with open(filename, "rb") as f:
        sparse_result = pickle.load(f)

    print(f"Loaded from {filename}")
    print(f"MI = {sparse_result['mi']:.6f}")
    print(f"Active features = {len(sparse_result['coeff'])}")

    return sparse_result

def sparse_predict(Theta, sparse_result):
    coeffs = sparse_result["coeff"]
    if Theta.shape[1] != len(coeffs):
        raise ValueError(
            f"Theta has {Theta.shape[1]} columns, "
            f"but model expects {len(coeffs)}."
        )

    return Theta @ coeffs