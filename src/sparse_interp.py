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
    library_power_results=[],
    threshold=0.1,
    max_iter=10,
    method="Powell",
    checkpoint_file="sparse_checkpoint.pkl",
    resume=False,
    save_every_iteration=True,
):
    Theta = np.asarray(Theta)
    y = np.asarray(y)

    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        iteration_start = checkpoint["iteration"] + 1
        active_indices = checkpoint["active_indices"]
        print(f"Resume from iteration {iteration_start}")
    else:
        iteration_start = 0
        history = []
        active_indices = np.arange(Theta.shape[1])

    for iteration in range(iteration_start, max_iter):

        # Current active library
        Theta_active = Theta[:, active_indices]

        # MI optimisation
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
                "method": method
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            print("\nTraining interrupted.")
            print(f"Checkpoint saved to {checkpoint_file}")

            return

        # Recover direction + gamma scaling
        coeff = result.x.astype(float)
        coeff /= np.linalg.norm(coeff)
        
        z = Theta_active @ coeff
        gamma = np.cov(z, y, bias=True)[0,1] / np.var(z)
        coeff *= gamma

        # Fix sign ambiguity
        idx = np.argmax(np.abs(coeff))
        if coeff[idx] < 0:
            coeff *= -1
            
        history.append({
            "iteration": iteration+1,
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

        # Threshold
        keep = np.abs(coeff) >= threshold

        # Prevent removing all features
        if not np.any(keep):
            keep[np.argmax(np.abs(coeff))] = True

        # Display
        for name, c, selected in zip(
            np.array(feature_names)[active_indices],
            coeff,
            keep
        ):
            status = "KEEP" if selected else "REMOVE"
            print(f"{name:35s}: {c: .6f}  {status}")

        # Stop if no feature removed
        if np.all(keep):
            break

        active_indices = active_indices[keep]
        
        if save_every_iteration:
            checkpoint = {
                "iteration": iteration,
                "active_indices": active_indices,
                "threshold": threshold,
                "method": method
            }
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

    # Final re-optimisation
    Theta_final = Theta[:, active_indices]

    final_result = minimize(
        scipy_objective,
        x0=np.ones(len(active_indices)),
        args=(Theta_final, y),
        method=method
    )

    final_coeff = final_result.x.astype(float)
    final_coeff /= np.linalg.norm(final_coeff)
    
    z = Theta_final @ final_coeff
    gamma = np.cov(z, y, bias=True)[0,1] / np.var(z)
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
        print(f"{name:35s}: {c:.6f}")
        terms.append(f"{c:.4f}*{name}")

    print("\nu ∝")
    print(" + ".join(terms))

    #clear
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
        "history": history
    }

def refine_sparse_result(
    sparse_result,
    Theta,
    y,
    coefficient_threshold=0.1,
    importance_threshold=0.1,
    method="Powell",
    print_result=True,
):
    """
    Refine a sparse EE model using coefficient pre-screening
    followed by leave-one-out feature importance analysis.

    The procedure is:

        1. Start from an existing sparse regression result.
        2. Remove features with small absolute coefficients.
        3. Re-optimise the remaining features.
        4. For each remaining feature, calculate its leave-one-out
           MI without re-optimisation.
        5. Remove features whose MI contribution is below
           importance_threshold.
        6. Re-optimise the remaining features once more.

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
        Features with delta_mi below this value are removed
        after leave-one-out analysis.

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

    active_indices = np.asarray(
        sparse_result["active_indices"]
    ).copy()

    feature_names = sparse_result[
        "library_feature_names"
    ]

    scaler = sparse_result["scaler"]

    library_power_results = sparse_result.get(
        "library_power_results",
        []
    )

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
                sparse_result["active_indices"]
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
    # Step 2: Re-optimise after coefficient screening
    # ==========================================================

    Theta_active = Theta[:, active_indices]

    result = minimize(
        scipy_objective,
        x0=np.ones(len(active_indices)),
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

    full_mi = ee.mi(u, y)

    # ==========================================================
    # Step 3: Leave-one-out feature importance
    # ==========================================================

    feature_results = []

    for i in range(len(active_indices)):

        keep = np.ones(
            len(active_indices),
            dtype=bool
        )

        keep[i] = False

        if keep.sum() == 0:

            mi_without = 0.0

        else:

            # IMPORTANT:
            # No re-optimisation here.
            # Use the current coefficients directly.
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
            "coefficient":
                coeff[i],
            "mi_without":
                mi_without,
            "delta_mi":
                delta_mi,
        })

    # ==========================================================
    # Step 4: Importance-based pruning
    # ==========================================================

    importance_keep = np.array([
        r["delta_mi"] >= importance_threshold
        for r in feature_results
    ])

    # Always keep at least one feature
    if not np.any(importance_keep):

        best_idx = np.argmax([
            r["delta_mi"]
            for r in feature_results
        ])

        importance_keep[best_idx] = True

    if print_result:

        print(
            "\n===== Leave-one-out Feature Importance ====="
        )

        print(
            f"Full MI : {full_mi:.6f}"
        )

        print()

        for r, keep in zip(
            feature_results,
            importance_keep
        ):

            status = (
                "KEEP"
                if keep
                else "REMOVE"
            )

            print(
                f"{r['feature']:35s}: "
                f"coeff={r['coefficient']: .6f}  "
                f"MI(-f)={r['mi_without']: .6f}  "
                f"ΔMI={r['delta_mi']: .6f}  "
                f"{status}"
            )

    active_indices = active_indices[
        importance_keep
    ]

    # ==========================================================
    # Step 5: Final re-optimisation
    # ==========================================================

    Theta_final = Theta[:, active_indices]

    final_result = minimize(
        scipy_objective,
        x0=np.ones(len(active_indices)),
        args=(Theta_final, y),
        method=method
    )

    final_coeff = (
        final_result.x.astype(float)
    )

    final_coeff /= np.linalg.norm(
        final_coeff
    )

    u_final = (
        Theta_final @ final_coeff
    )

    gamma = (
        np.cov(
            u_final,
            y,
            bias=True
        )[0, 1]
        / np.var(u_final)
    )

    final_coeff *= gamma

    # Fix sign
    idx = np.argmax(
        np.abs(final_coeff)
    )

    if final_coeff[idx] < 0:
        final_coeff *= -1

    final_mi = ee.mi(
        u_final,
        y
    )

    final_names = [
        feature_names[i]
        for i in active_indices
    ]

    # ==========================================================
    # Print final result
    # ==========================================================

    if print_result:

        print(
            "\n===== Refined Sparse Combination ====="
        )

        print(
            f"Initial terms : "
            f"{len(sparse_result['active_indices'])}"
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
        "coeff": final_coeff,
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
            feature_results,
        "coefficient_keep":
            coefficient_keep,
        "importance_keep":
            importance_keep,
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