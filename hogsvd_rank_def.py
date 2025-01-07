# https://github.com/kmpape/HO-GSVD

import warnings

import numpy as np


def hogsvd(A, m, verbose: bool = False, **kwargs):
    default_args = {
        'RANK_TOL_A': 1e-14,
        'ppi': 1e-3,
        'ZEROTOL': 1e-14,
        'EPS_REL_ISO': 1e-6,
        'DISABLE_WARNINGS': False
    }
    args = {**default_args, **kwargs}

    n = A.shape[1]
    N = len(m)
    if verbose:
        print("A.shape=", A.shape, "m=", m)

    if A.shape[0] != np.sum(m):
        raise ValueError(f'A.shape[0]={A.shape[0]} != sum(m)={np.sum(m)}')

    SA = np.linalg.svd(A, compute_uv=False)
    rank_def_A = n - np.sum(SA > args['RANK_TOL_A'])
    if rank_def_A == 0:
        Apad = A
    else:
        if not args['DISABLE_WARNINGS']:
            warnings.warn(f"Warning: Provided rank-deficient A w. rank(A)={n - rank_def_A}<n={n}. Padding A.")
        _, _, VA = np.linalg.svd(A)
        Apad = np.vstack([A, VA[:, -rank_def_A:].T])
        m = np.append(m, rank_def_A)
    m = np.array(m).astype(int)

    Q, R = np.linalg.qr(Apad, mode='reduced')
    U, S, Z, Tau, taumin, taumax, iso_classes = hocsd(Q, m)
    V = R.T @ Z

    if rank_def_A > 0:
        U = U[:-rank_def_A, :]
        S = S[:-n, :]

    return U, S, V, Tau, taumin, taumax, iso_classes


def hocsd(Q, m, ppi=1e-3, ZEROTOL=1e-14, EPS_REL_ISO=1e-8, DISABLE_WARNINGS=False):
    WARN_EPS_ISO = 1e-8
    WARN_COND = 1e8
    N = len(m)
    n = Q.shape[1]
    print("Q.shape=", Q.shape, "m=", m)

    if np.sum(m) < n:
        raise ValueError(f"sum(m)={np.sum(m)} < n={n}. Rank(Q)={n} required.")

    if Q.shape[0] != np.sum(m):
        raise ValueError(f"Q.shape[0]={Q.shape[0]} != sum(m)={np.sum(m)}.")

    Rhat = np.zeros((n, N, n))
    sqrt_ppi = np.sqrt(ppi)

    for i in range(N):
        Qi = get_mat_from_stacked(Q, m, i)
        _, Rhati = np.linalg.qr(np.vstack((Qi, np.eye(n) * sqrt_ppi)), mode='reduced')
        Rhat[:, i, :] = np.linalg.inv(Rhati)

        if np.linalg.cond(Rhati) >= WARN_COND:
            print(f"For i={i}, cond(Rhati)={np.linalg.cond(Rhati)}")
    Rhat = Rhat.reshape(n, N * n)

    Z, sqrt_Tau, _ = np.linalg.svd(Rhat, full_matrices=False)
    Tau = np.diag(np.square(np.diag(sqrt_Tau)) / N)
    taumin = 1 / (1 / N + ppi)  # theoretical min. of Tau
    taumax = (N - 1) / N / ppi + 1 / N / (1 + ppi)  # theoretical max. of Tau

    # % Indices corresponding to the isolated subspace (eq. (6.6))
    ind_iso = np.abs(taumax * np.ones((n,)) - Tau) <= (taumax - taumin) * EPS_REL_ISO
    iso_classes = []
    if np.any(ind_iso):  # align Z_iso to standard RSVs of Qi
        Z_iso = Z[:, ind_iso]
        Z_iso_new = np.zeros(Z_iso.shape)
        n_iso = np.sum(ind_iso)
        iso_classes = np.zeros((n_iso))

        Z_iter = Z_iso
        for i in range(n_iso - 1):
            all_S = np.zeros((N,))
            for j in range(N):
                Qj = get_mat_from_stacked(Q, m, j)
                all_S[j] = np.linalg.norm(Qj @ Z_iter, 2)

            ind_sorted = np.argsort(all_S)[::-1]
            iso_classes[i] = ind_sorted[0]
            Qiso = get_mat_from_stacked(Q, m, int(iso_classes[i]))
            _, _, Xiso = np.linalg.svd(Qiso @ Z_iter, full_matrices=False)
            Z_iso_new[:, i] = Z_iter @ Xiso[:, 0]
            Z_iter = Z_iter @ Xiso[:, 1:]

        all_S = np.zeros(N)
        for j in range(N):
            Qj = get_mat_from_stacked(Q, m, j)
            all_S[j] = np.linalg.norm(Qj @ Z_iter, 2)
        ind_sorted = np.argsort(all_S)[::-1]
        iso_classes[-1] = ind_sorted[0]
        Z_iso_new[:, -1] = Z_iter.squeeze()

        Z[:, ind_iso] = Z_iso_new
        if not DISABLE_WARNINGS:
            if np.linalg.norm(np.eye(n) - Z.T @ Z) > WARN_EPS_ISO:
                print(f"Rotated Z is not orthogonal, norm(eye(n)-Z^T*Z)={np.linalg.norm(np.eye(n) - Z.T @ Z)}.")

    if np.linalg.norm(np.eye(n) - Z.T @ Z) > WARN_EPS_ISO:
        not_ortho = True
    else:
        not_ortho = False

    S = np.zeros((N, n, n))
    U = np.zeros((np.sum(m), n))
    for i in range(N):
        Qi = get_mat_from_stacked(Q, m, i)
        if not_ortho:
            Bi = Qi @ np.linalg.inv(Z.T)
        else:
            Bi = Qi @ Z
        Si = np.linalg.norm(Bi, ord=2, axis=0)
        ind_pos = Si > ZEROTOL
        U[np.sum(m[:i]):np.sum(m[:i + 1]), ind_pos] = Bi[:, ind_pos] * (1.0 / Si[ind_pos])

        nzero = np.sum(~ind_pos)
        if nzero > 0:
            UQi, SQi, _ = np.linalg.svd(Qi, full_matrices=False)

            ind_zero_i = SQi <= ZEROTOL
            ni2 = np.sum(ind_zero_i)
            if ni2 == 0:
                Qitmp = Qi[:, ~ind_pos]
                Qitmp_norm = np.linalg.norm(Qitmp, axis=0)
                Qitmp_norm[Qitmp_norm <= ZEROTOL] = 1
                U[np.sum(m[:i]):np.sum(m[:i + 1]), ~ind_pos] = Qitmp * (1.0 / Qitmp_norm)
            else:
                Ui2 = UQi[:, ind_zero_i]
                if ni2 < nzero:
                    Ui2 = np.tile(Ui2, (1, np.ceil(nzero / ni2).astype(int)))
                U[np.sum(m[:i]):np.sum(m[:i + 1]), ~ind_pos] = Ui2[:, :nzero]
        S[i, :, :] = np.diag(Si)
    S = S.reshape(N * n, n)

    return U, S, Z, Tau, taumin, taumax, iso_classes


def get_mat_from_stacked(Q, m, i):
    start_row = sum(m[:i])
    end_row = start_row + m[i]
    return Q[start_row:end_row, :]
