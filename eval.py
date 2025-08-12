"""
supervisors edition
import numpy as np
from collections import defaultdict

def run_eval(
    transport_maps,          # dict[(L,k,j_policy)] -> (T:[d,d], b:[d])
    chosen_layers, k_list, j_policy_list,
    residual_stream_pairs,   # dict[(L,k,j_policy)] -> (X_up:[N,d], Y_down:[N,d])
    feature_list,            # list of feature names OR indices
    SAE_decoders,            # dict[layer] -> dict[feature] -> d_f:[d]
    score_latent,            # fn(a_true, a_pred) -> (r2, mse, r)
    score_residual=None,     # optional fn(Y_true, Y_pred) -> dict
    decoder_normalize=True
):
    results = {}

    for L in chosen_layers:
        for k in k_list:
            for j_policy in j_policy_list:

                key = (L, k, j_policy)
                if key not in transport_maps or key not in residual_stream_pairs:
                    continue  # skip missing configs

                T, b = transport_maps[key]
                X_up, Y_dn = residual_stream_pairs[key]     # shapes [N,d], [N,d]
                # predict downstream residuals
                Y_hat = X_up @ T.T + b                      # [N,d]

                # residual-level metrics (optional but recommended)
                res_metrics = {}
                if score_residual is not None:
                    res_metrics = score_residual(Y_dn, Y_hat)  # e.g., {"r2_res":..., "mse_res":..., "cos_res":...}

                # prepare decoders for the target layer L+k
                D_layer = {}
                for feat in feature_list:
                    d = np.asarray(SAE_decoders[L + k][feat], dtype=Y_dn.dtype)
                    if decoder_normalize:
                        n = np.linalg.norm(d)
                        if n > 0: d = d / n
                    D_layer[feat] = d

                # score each feature
                for feat in feature_list:
                    d_f = D_layer[feat]                      # [d]
                    a_true = Y_dn @ d_f                      # [N]
                    a_pred = Y_hat @ d_f                     # [N]

                    r2_lat, mse_lat, r_pearson = score_latent(a_true, a_pred)

                    # optional: calibration (slope, intercept)
                    X = np.vstack([a_pred, np.ones_like(a_pred)]).T
                    # lstsq for stability
                    beta, *_ = np.linalg.lstsq(X, a_true, rcond=None)
                    slope, intercept = float(beta[0]), float(beta[1])

                    results[(L, k, j_policy, feat)] = {
                        "r2_lat": float(r2_lat),
                        "mse_lat": float(mse_lat),
                        "r_pearson": float(r_pearson),
                        "slope": slope,
                        "intercept": intercept,
                        **res_metrics
                    }

    return results


"""


def run_eval(
    transport_maps, chosen_layers, k_list,
    j_policy_list, residual_stream_pairs,
    feature_list, SAE_decoders, score_latent
):
    results = {}
    for L in chosen_layers:
        for k in k_list:
            for j_policy in j_policy_list:

                # Get the transport map for the current (L, k, j_policy)
                T, b = transport_maps[(L, k, j_policy)]
                # Prepare the residuals for evaluation
                upstream_gt, downstream_gt = residual_stream_pairs[(
                    L, k, j_policy)]
                # Evaluate the transport map
                downstream_pred = upstream_gt @ T.T + b  # [N, d]

                for feature in feature_list:
                    decoder_direction = SAE_decoders[L + k][feature]  # [d]
                    # Compute the latent activations
                    a_true = downstream_gt @ decoder_direction  # [N]
                    a_pred = downstream_pred @ decoder_direction  # [N]

                    # Compute evaluation metrics
                    r2_lat, mse_lat, r_pearson = score_latent(a_true, a_pred)

                    # Store the results
                    results[(L, k, j_policy, feature)] = {
                        "r2_lat": r2_lat,
                        "mse_lat": mse_lat,
                        "r_pearson": r_pearson
                    }

    return results
