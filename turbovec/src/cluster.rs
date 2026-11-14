//! Deterministic Euclidean k-means: a classic Lloyd baseline and
//! **e-means**, the mini-batch exponential-moving-average solver of
//! Bessa et al. ("e-means: outperforming k-means in one-tenth the
//! epochs", Meta FAIR, arXiv 2611.09501, 2026).
//!
//! turbovec needs a vector clustering primitive for inverted-file
//! (IVF) coarse-centroid training; until now the only clustering in
//! the crate was the one-dimensional Lloyd-Max scalar quantizer in
//! `codebook` (a different problem). This module adds the general
//! `k`-centroid Euclidean solver.
//!
//! # Determinism
//!
//! Both solvers are **bit-for-bit deterministic**: the same
//! `(sample, k, dim, seed)` yields byte-identical centroids on any
//! host, at any thread count. That is a hard requirement for callers
//! that persist centroids to an on-disk format and gate it with
//! equality tests (pg_turbovec's IVF relfile). Determinism is bought
//! the same way as the rest of the crate:
//!
//! * every random choice (initialisation, per-epoch shuffle, revival
//!   donor) is drawn from a caller-supplied `ChaCha8Rng(seed)`, never
//!   from thread-local or time-seeded state;
//! * every `f64` reduction runs over a **fixed partition in a fixed
//!   order** derived from `(n, k, dim)` and never from the thread
//!   count, because `f64` addition is not associative (the same
//!   discipline `rotation`/`pack` use).
//!
//! e-means output is *not* expected to equal the Lloyd output; it is a
//! different (and, per the paper, usually lower-loss) codebook. The
//! guarantee is reproducibility, not agreement with Lloyd.
//!
//! # e-means in one paragraph
//!
//! Each cluster `j` keeps a count `N_j` and a vector sum `S_j`; the
//! centroid is their ratio `c_j = S_j / N_j`. A run of `T` epochs
//! processes the data in freshly-shuffled mini-batches of size `B`
//! (both `B` and the peak step `alpha0` come from fitted scaling
//! laws, so there is nothing to tune). Each batch updates the
//! statistics by an exponential moving average
//! `S_j <- (1-alpha) S_j + alpha s_j`, with `alpha` annealed on a
//! cosine schedule. Every five batches, dead clusters (count far
//! below the mean) are revived by splitting a large donor. A final
//! last-epoch-mean pass sets each centroid to the exact mean of the
//! points assigned to it in the last epoch.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The result of a clustering run.
#[derive(Clone, Debug)]
pub struct ClusterResult {
    /// `k * dim` centroids, row-major (`centroids[j*dim .. (j+1)*dim]`
    /// is centroid `j`).
    pub centroids: Vec<f32>,
    /// Lloyd iterations run (`cluster_lloyd`) or e-means batches
    /// processed (`cluster_emeans`). Diagnostic; a deterministic
    /// function of the input.
    pub steps: usize,
}

/// Squared Euclidean distance. Scalar and order-fixed so it is
/// identical on every target (the clustering argmin must not depend on
/// a SIMD reduction order).
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Index of the nearest centroid to `x`, ties broken toward the lower
/// index (so the assignment is a pure function of the inputs).
#[inline]
fn nearest(x: &[f32], centroids: &[f32], k: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for j in 0..k {
        let d = sq_dist(x, &centroids[j * dim..(j + 1) * dim]);
        if d < best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

/// Pick `k` distinct row indices from `0..n` uniformly at random, in a
/// deterministic order, via a partial Fisher-Yates over `rng`. When
/// `k >= n` every row is returned (padded by repeating the last row's
/// index is avoided by the caller, which never asks for `k > n` here).
fn choose_distinct(n: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    debug_assert!(k <= n);
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.gen_range(0..(n - i) as u64) as usize);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

// ---------------------------------------------------------------------
// Lloyd baseline
// ---------------------------------------------------------------------

/// Classic Lloyd's k-means with k-means++ (D2) seeding.
///
/// A deterministic reference implementation: k-means++ seeds from
/// `rng`, then up to `max_iter` assign/update passes with a fixed-order
/// `f64` mean accumulation and an empty-cluster rule that reseeds a dead
/// centroid to the point farthest from its current centroid. Returns
/// once no centroid moves (byte-level equality of consecutive centroid
/// arrays) or `max_iter` passes have run.
///
/// `sample` is `n * dim` row-major. `k >= 1`, `dim >= 1`. For `n == 0`
/// the centroids are all-zero. For `k >= n` every point becomes its own
/// centroid (remaining slots reseed via the empty-cluster rule).
pub fn cluster_lloyd(
    sample: &[f32],
    n: usize,
    k: usize,
    dim: usize,
    max_iter: usize,
    seed: u64,
) -> ClusterResult {
    assert!(k >= 1, "cluster_lloyd: k must be >= 1");
    assert!(dim >= 1, "cluster_lloyd: dim must be >= 1");
    debug_assert_eq!(sample.len(), n * dim);
    let mut centroids = vec![0.0f32; k * dim];
    if n == 0 {
        return ClusterResult {
            centroids,
            steps: 0,
        };
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let row = |i: usize| &sample[i * dim..(i + 1) * dim];

    // k-means++ seeding.
    let first = rng.gen_range(0..n as u64) as usize;
    centroids[0..dim].copy_from_slice(row(first));
    let mut d2 = vec![f32::INFINITY; n];
    for (i, di) in d2.iter_mut().enumerate() {
        *di = sq_dist(row(i), &centroids[0..dim]);
    }
    for c in 1..k {
        let total: f64 = d2.iter().map(|&x| x as f64).sum();
        let chosen = if total <= 0.0 {
            rng.gen_range(0..n as u64) as usize
        } else {
            let target = rng.gen_range(0.0..total);
            let mut acc = 0.0f64;
            let mut idx = n - 1;
            for (i, &x) in d2.iter().enumerate() {
                acc += x as f64;
                if acc >= target {
                    idx = i;
                    break;
                }
            }
            idx
        };
        centroids[c * dim..(c + 1) * dim].copy_from_slice(row(chosen));
        for (i, di) in d2.iter_mut().enumerate() {
            let d = sq_dist(row(i), &centroids[c * dim..(c + 1) * dim]);
            if d < *di {
                *di = d;
            }
        }
    }

    // Lloyd iterations.
    let mut assign = vec![0usize; n];
    let mut prev = vec![0.0f32; k * dim];
    let mut iters = 0usize;
    for _ in 0..max_iter {
        iters += 1;
        prev.copy_from_slice(&centroids);
        for (i, a) in assign.iter_mut().enumerate() {
            *a = nearest(row(i), &centroids, k, dim);
        }
        // Fixed-order f64 accumulation (serial, ascending row order).
        let mut sums = vec![0.0f64; k * dim];
        let mut counts = vec![0u64; k];
        for (i, &j) in assign.iter().enumerate() {
            counts[j] += 1;
            let s = &mut sums[j * dim..(j + 1) * dim];
            let r = row(i);
            for d in 0..dim {
                s[d] += r[d] as f64;
            }
        }
        for j in 0..k {
            if counts[j] > 0 {
                let inv = 1.0 / counts[j] as f64;
                for d in 0..dim {
                    centroids[j * dim + d] = (sums[j * dim + d] * inv) as f32;
                }
            } else {
                // Empty cluster: reseed to the point farthest from its
                // assigned centroid (deterministic, data-driven).
                let mut worst = 0usize;
                let mut worst_d = -1.0f32;
                for i in 0..n {
                    let d = sq_dist(row(i), &centroids[assign[i] * dim..(assign[i] + 1) * dim]);
                    if d > worst_d {
                        worst_d = d;
                        worst = i;
                    }
                }
                centroids[j * dim..(j + 1) * dim].copy_from_slice(row(worst));
            }
        }
        if centroids == prev {
            break;
        }
    }
    ClusterResult {
        centroids,
        steps: iters,
    }
}

// ---------------------------------------------------------------------
// e-means scaling laws (site Table 2)
// ---------------------------------------------------------------------

/// e-means batch size for a problem of `n` points, `k` clusters, `T`
/// epochs: `B = c_B * k^0.13 * n^0.88 * T^0.81`, rounded to the nearest
/// power of two and clamped to `1..=n`. At `T > 2` the default law is
/// divided by 8 (the paper's large-scale throughput adjustment).
/// `c_B = 0.035` (the default law).
pub fn emeans_batch_size(n: usize, k: usize, t: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let (nf, kf, tf) = (n as f64, k.max(1) as f64, t.max(1) as f64);
    let mut b = 0.035 * kf.powf(0.13) * nf.powf(0.88) * tf.powf(0.81);
    if t > 2 {
        b /= 8.0;
    }
    let b = round_pow2(b.max(1.0));
    b.clamp(1, n)
}

/// e-means peak step for `(n, k, T, B)`:
/// `alpha0 = min(0.824 * k^-0.46 * B^0.90 * n^-0.45 * T^-0.11, 1)`.
/// The step law depends on `B`, so compute `B` first.
pub fn emeans_alpha0(n: usize, k: usize, t: usize, b: usize) -> f32 {
    let (nf, kf, tf, bf) = (
        n.max(1) as f64,
        k.max(1) as f64,
        t.max(1) as f64,
        b.max(1) as f64,
    );
    let a = 0.824 * kf.powf(-0.46) * bf.powf(0.90) * nf.powf(-0.45) * tf.powf(-0.11);
    a.min(1.0) as f32
}

/// Round a positive value to the nearest power of two (ties to the
/// higher power).
fn round_pow2(x: f64) -> usize {
    if x <= 1.0 {
        return 1;
    }
    let lo = 1usize << x.log2().floor() as u32;
    let hi = lo << 1;
    if (x - lo as f64) <= (hi as f64 - x) {
        lo
    } else {
        hi
    }
}

// ---------------------------------------------------------------------
// e-means
// ---------------------------------------------------------------------

/// Revive dead clusters by reseeding them to poorly-covered data
/// points.
///
/// A cluster is dead when its EMA count is below `0.1 * mean(count)`.
/// The paper revives by splitting a large donor with a tiny `[1±ε]`
/// perturbation and relying on later batches to pull the two children
/// apart. At the scales and epoch budgets used for IVF coarse
/// training (large `k` relative to the batch, few epochs) that drift
/// does not complete: a near-copy child competes for the donor's own
/// points and the originally-uncovered region stays empty, so dead
/// clusters accumulate. We instead reseed each dead cluster to an
/// actual batch point that is FAR from every current centroid (the
/// D2-style choice Lloyd's empty-cluster rule uses), which claims an
/// uncovered region immediately. This is the "resample from batch"
/// strategy the paper ablates (Table 14), sharpened to pick the
/// worst-covered point rather than a uniform one, and it keeps every
/// cluster live.
///
/// The revived cluster is seeded with a small weight so later batches
/// dominate it, matching the EMA initialisation. `batch` is the slice
/// of point indices assigned in the current batch (the pool of real
/// points to reseed from); `sample` is the full `n*dim` corpus. Each
/// batch point's nearest-centroid distance is computed ONCE
/// (`O(batch * k)`), then the dead clusters take the farthest
/// well-separated points. Deterministic: the sort breaks ties by point
/// index, so the reseed targets are a fixed function of the input.
fn revive_dead_clusters(
    counts: &mut [f64],
    sums: &mut [f64],
    centroids: &[f32],
    sample: &[f32],
    batch: &[usize],
    k: usize,
    dim: usize,
) {
    let mean = counts.iter().sum::<f64>() / k as f64;
    let threshold = 0.1 * mean;
    let dead: Vec<usize> = (0..k).filter(|&j| counts[j] < threshold).collect();
    if dead.is_empty() || batch.is_empty() {
        return;
    }
    // Seed weight for a revived cluster: small, so subsequent EMA
    // updates dominate, but non-zero so `c = S/N` is well-defined.
    const SEED_N: f64 = 1e-3;

    // Each batch point's distance to its nearest centroid, computed
    // ONCE (O(batch * k), not per dead cluster -- the per-dead rescan
    // was O(dead * batch * k) and dominated the whole run at large k).
    let mut cov: Vec<(f32, usize)> = batch
        .iter()
        .map(|&i| {
            let x = &sample[i * dim..(i + 1) * dim];
            let mut nd = f32::INFINITY;
            for j in 0..k {
                let d = sq_dist(x, &centroids[j * dim..(j + 1) * dim]);
                if d < nd {
                    nd = d;
                }
            }
            (nd, i)
        })
        .collect();
    // Farthest-first: sort by coverage distance descending (ties to the
    // lower point index, so the order is a fixed function of the input).
    cov.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    // Reseed each dead cluster to the next farthest point, skipping
    // points within `min_sep2` of one already used this pass so two
    // dead clusters don't land on the same uncovered spot.
    let min_sep2 = {
        // A tenth of the farthest coverage distance, squared scale; 0
        // when everything is already well covered (nothing reseeds).
        let far = cov.first().map(|&(d, _)| d).unwrap_or(0.0);
        far * 0.01
    };
    let mut used: Vec<usize> = Vec::with_capacity(dead.len());
    let mut it = cov.into_iter();
    for &dead_j in &dead {
        let pick = loop {
            let Some((_, i)) = it.next() else {
                return; // batch exhausted
            };
            let x = &sample[i * dim..(i + 1) * dim];
            let too_close = used
                .iter()
                .any(|&u| sq_dist(x, &sample[u * dim..(u + 1) * dim]) < min_sep2);
            if !too_close {
                break i;
            }
        };
        used.push(pick);
        let src = &sample[pick * dim..(pick + 1) * dim];
        counts[dead_j] = SEED_N;
        let dst = &mut sums[dead_j * dim..(dead_j + 1) * dim];
        for d in 0..dim {
            dst[d] = SEED_N * src[d] as f64;
        }
    }
}

/// e-means: mini-batch EMA k-means with automatic batch/step from
/// scaling laws, cosine step annealing, dead-cluster revival, and a
/// last-epoch-mean finalisation.
///
/// `sample` is `n * dim` row-major; `epochs` is `T` (one pass over all
/// `n` points per epoch). `k >= 1`, `dim >= 1`. For `n == 0` the
/// centroids are all-zero; for `k >= n` the result degenerates to one
/// centroid per point (revival keeps the rest live). Deterministic in
/// `seed`.
pub fn cluster_emeans(
    sample: &[f32],
    n: usize,
    k: usize,
    dim: usize,
    epochs: usize,
    seed: u64,
) -> ClusterResult {
    assert!(k >= 1, "cluster_emeans: k must be >= 1");
    assert!(dim >= 1, "cluster_emeans: dim must be >= 1");
    debug_assert_eq!(sample.len(), n * dim);
    let centroids = vec![0.0f32; k * dim];
    if n == 0 {
        return ClusterResult {
            centroids,
            steps: 0,
        };
    }
    let t = epochs.max(1);
    let b = emeans_batch_size(n, k, t).min(n);
    let alpha0 = emeans_alpha0(n, k, t, b) as f64;
    cluster_emeans_tuned(sample, n, k, dim, t, b, alpha0, seed)
}

/// e-means with the batch size `B` and peak step `alpha0` supplied
/// directly, bypassing the scaling laws. Used by tests and by callers
/// that want to override the automatic tuning. All other behaviour
/// matches [`cluster_emeans`].
#[allow(clippy::too_many_arguments)]
pub fn cluster_emeans_tuned(
    sample: &[f32],
    n: usize,
    k: usize,
    dim: usize,
    epochs: usize,
    b: usize,
    alpha0: f64,
    seed: u64,
) -> ClusterResult {
    assert!(k >= 1 && dim >= 1);
    debug_assert_eq!(sample.len(), n * dim);
    if n == 0 {
        return ClusterResult {
            centroids: vec![0.0f32; k * dim],
            steps: 0,
        };
    }
    let b = b.clamp(1, n);
    let t = epochs.max(1);
    let row = |i: usize| &sample[i * dim..(i + 1) * dim];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Statistics in f64 for a stable EMA (the same precision the mean
    // accumulation uses). Centroid = S / N, materialised on demand.
    let mut counts = vec![0.0f64; k];
    let mut sums = vec![0.0f64; k * dim];

    // Initialisation: k distinct random rows, with near-zero weight so
    // the first updates dominate. N_j = 1e-10, S_j = N_j * c_j.
    let init = choose_distinct(n, k.min(n), &mut rng);
    for (j, &pick) in init.iter().enumerate() {
        counts[j] = 1e-10;
        let r = row(pick);
        for d in 0..dim {
            sums[j * dim + d] = 1e-10 * r[d] as f64;
        }
    }
    // When k > n, the surplus clusters start empty; revival populates
    // them. Seed them at a far point so they are not all identical.
    for j in init.len()..k {
        counts[j] = 1e-10;
        let r = row(j % n);
        for d in 0..dim {
            sums[j * dim + d] = 1e-10 * r[d] as f64;
        }
    }

    let total_batches = t * n.div_ceil(b);
    let mut centroids = vec![0.0f32; k * dim];
    let materialize = |counts: &[f64], sums: &[f64], centroids: &mut [f32]| {
        for j in 0..k {
            let nj = counts[j];
            if nj > 0.0 {
                let inv = 1.0 / nj;
                for d in 0..dim {
                    centroids[j * dim + d] = (sums[j * dim + d] * inv) as f32;
                }
            }
        }
    };

    let mut batch_no = 0usize;
    let mut order: Vec<usize> = (0..n).collect();
    // Record last-epoch assignments for the finalisation.
    let mut last_assign = vec![u32::MAX; n];
    for epoch in 0..t {
        // Fresh deterministic shuffle of the point order each epoch.
        fisher_yates(&mut order, &mut rng);
        let is_last_epoch = epoch == t - 1;
        let mut off = 0usize;
        while off < n {
            let bs = b.min(n - off);
            // Current centroids drive this batch's assignment.
            materialize(&counts, &sums, &mut centroids);
            // Cosine-annealed step. f = 0.1 floor.
            let frac = if total_batches > 1 {
                (batch_no as f64) / ((total_batches - 1) as f64)
            } else {
                0.0
            };
            let cos = 0.5 * (1.0 + (std::f64::consts::PI * frac).cos());
            let alpha = alpha0 * (0.1 + 0.9 * cos);

            // Batch statistics: s_j, n_j over the batch's assigned points.
            let mut bs_counts = vec![0.0f64; k];
            let mut bs_sums = vec![0.0f64; k * dim];
            for &i in &order[off..off + bs] {
                let j = nearest(row(i), &centroids, k, dim);
                bs_counts[j] += 1.0;
                let s = &mut bs_sums[j * dim..(j + 1) * dim];
                let r = row(i);
                for d in 0..dim {
                    s[d] += r[d] as f64;
                }
                if is_last_epoch {
                    last_assign[i] = j as u32;
                }
            }
            // EMA update: S <- (1-a)S + a s ; N <- (1-a)N + a n.
            for j in 0..k {
                counts[j] = (1.0 - alpha) * counts[j] + alpha * bs_counts[j];
                let s = &mut sums[j * dim..(j + 1) * dim];
                let bsj = &bs_sums[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    s[d] = (1.0 - alpha) * s[d] + alpha * bsj[d];
                }
            }
            batch_no += 1;
            if batch_no.is_multiple_of(5) {
                // Reseed from the batch using the CURRENT centroids (the
                // ones this update just produced), so a dead cluster is
                // moved to a genuinely uncovered point.
                materialize(&counts, &sums, &mut centroids);
                revive_dead_clusters(
                    &mut counts,
                    &mut sums,
                    &centroids,
                    sample,
                    &order[off..off + bs],
                    k,
                    dim,
                );
            }
            off += bs;
        }
    }

    // Last-epoch-mean finalisation: each centroid = exact mean of the
    // points assigned to it in the last epoch; empty clusters keep the
    // trained centroid. Fixed-order f64 accumulation.
    materialize(&counts, &sums, &mut centroids);
    let mut fin_sums = vec![0.0f64; k * dim];
    let mut fin_counts = vec![0u64; k];
    for (i, &a) in last_assign.iter().enumerate() {
        if a == u32::MAX {
            continue;
        }
        let j = a as usize;
        fin_counts[j] += 1;
        let s = &mut fin_sums[j * dim..(j + 1) * dim];
        let r = row(i);
        for d in 0..dim {
            s[d] += r[d] as f64;
        }
    }
    for j in 0..k {
        if fin_counts[j] > 0 {
            let inv = 1.0 / fin_counts[j] as f64;
            for d in 0..dim {
                centroids[j * dim + d] = (fin_sums[j * dim + d] * inv) as f32;
            }
        }
    }

    ClusterResult {
        centroids,
        steps: total_batches,
    }
}

/// In-place deterministic Fisher-Yates shuffle over `rng`.
fn fisher_yates(order: &mut [usize], rng: &mut ChaCha8Rng) {
    let n = order.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=(i as u64)) as usize;
        order.swap(i, j);
    }
}

/// Mean squared distance of each point to its nearest centroid: the
/// k-means objective. Provided for callers and tests that compare two
/// codebooks on the same data.
pub fn mse(sample: &[f32], n: usize, centroids: &[f32], k: usize, dim: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for i in 0..n {
        let x = &sample[i * dim..(i + 1) * dim];
        let j = nearest(x, centroids, k, dim);
        acc += sq_dist(x, &centroids[j * dim..(j + 1) * dim]) as f64;
    }
    acc / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic synthetic mixture: `blobs` clusters of `per` points
    // each, blob centres on a wide grid, small jitter. Returns
    // (sample, n). Uses a fixed RNG so the fixture itself is stable.
    fn make_blobs(blobs: usize, per: usize, dim: usize, spread: f32) -> (Vec<f32>, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(0xB10B);
        let n = blobs * per;
        let mut s = vec![0.0f32; n * dim];
        for bl in 0..blobs {
            let mut centre = vec![0.0f32; dim];
            for (d, c) in centre.iter_mut().enumerate() {
                *c = ((bl * 97 + d * 13) % 100) as f32 * spread;
            }
            for p in 0..per {
                let i = bl * per + p;
                for d in 0..dim {
                    let jitter = (rng.gen_range(0..1000u64) as f32 / 1000.0 - 0.5) * 2.0;
                    s[i * dim + d] = centre[d] + jitter;
                }
            }
        }
        (s, n)
    }

    #[test]
    fn emeans_is_deterministic() {
        let (s, n) = make_blobs(5, 400, 8, 50.0);
        let a = cluster_emeans(&s, n, 5, 8, 4, 42);
        let b = cluster_emeans(&s, n, 5, 8, 4, 42);
        assert_eq!(a.centroids, b.centroids, "same seed => byte-identical");
    }

    #[test]
    fn lloyd_is_deterministic() {
        let (s, n) = make_blobs(5, 400, 8, 50.0);
        let a = cluster_lloyd(&s, n, 5, 8, 25, 42);
        let b = cluster_lloyd(&s, n, 5, 8, 25, 42);
        assert_eq!(a.centroids, b.centroids);
    }

    #[test]
    fn emeans_determinism_independent_of_rayon_pool() {
        // The solver is single-threaded by construction, but assert the
        // output does not depend on the ambient rayon pool size either.
        let (s, n) = make_blobs(4, 300, 8, 40.0);
        let run = || cluster_emeans(&s, n, 4, 8, 4, 7).centroids;
        let p1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let p8 = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let a = p1.install(run);
        let b = p8.install(run);
        assert_eq!(a, b);
    }

    #[test]
    fn batch_size_is_power_of_two_and_bounded() {
        for &(n, k, t) in &[
            (1_000usize, 16usize, 4usize),
            (1_000_000, 1024, 4),
            (10_000, 256, 2),
        ] {
            let b = emeans_batch_size(n, k, t);
            assert!(b >= 1 && b <= n, "B={b} out of 1..={n}");
            assert!(b.is_power_of_two(), "B={b} not a power of two");
        }
    }

    #[test]
    fn alpha0_in_unit_interval() {
        for &(n, k, t) in &[(1_000usize, 16usize, 4usize), (1_000_000, 1024, 4)] {
            let b = emeans_batch_size(n, k, t);
            let a = emeans_alpha0(n, k, t, b);
            assert!(a > 0.0 && a <= 1.0, "alpha0={a} out of (0,1]");
        }
    }

    #[test]
    fn emeans_beats_lloyd_at_ivf_scale() {
        // The scaling laws are fit for high-dimensional, large-n
        // problems (the IVF coarse-training regime): n >= ~10^5,
        // k >= ~256. At that scale e-means reaches a lower loss than
        // Lloyd-at-25-iterations in a handful of epochs, which is the
        // whole point. A tiny k=5/n=2500 toy is OUT of the laws'
        // domain and can land two centroids on one blob (the paper's
        // per-seed duplicate trap); we therefore test in-domain.
        let (k, dim, per) = (256usize, 32usize, 200usize);
        let n = k * per;
        let mut rng = ChaCha8Rng::seed_from_u64(0x5CA1E);
        let mut s = vec![0.0f32; n * dim];
        for bl in 0..k {
            let mut centre = vec![0.0f32; dim];
            for (d, c) in centre.iter_mut().enumerate() {
                *c = ((bl * 131 + d * 17) % 1000) as f32;
            }
            for p in 0..per {
                let i = bl * per + p;
                for d in 0..dim {
                    let j = (rng.gen_range(0..1000u64) as f32 / 1000.0 - 0.5) * 4.0;
                    s[i * dim + d] = centre[d] + j;
                }
            }
        }
        let em = cluster_emeans(&s, n, k, dim, 4, 123);
        let ll = cluster_lloyd(&s, n, k, dim, 25, 123);
        let mse_e = mse(&s, n, &em.centroids, k, dim);
        let mse_l = mse(&s, n, &ll.centroids, k, dim);
        assert!(
            mse_e <= mse_l,
            "e-means (4 epochs) MSE {mse_e} should be <= Lloyd-25 MSE {mse_l} at IVF scale"
        );
    }

    #[test]
    fn revival_leaves_no_dead_clusters_on_imbalanced_data() {
        // Two blobs of very different sizes, k=8: without revival most
        // centroids pile on the big blob and several stay dead. Assert
        // every returned centroid owns at least one point.
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD);
        let dim = 8;
        let big = 2000usize;
        let small = 100usize;
        let n = big + small;
        let mut s = vec![0.0f32; n * dim];
        for i in 0..big {
            for d in 0..dim {
                s[i * dim + d] = (rng.gen_range(0..1000u64) as f32 / 1000.0) * 2.0;
            }
        }
        for i in big..n {
            for d in 0..dim {
                s[i * dim + d] = 100.0 + (rng.gen_range(0..1000u64) as f32 / 1000.0) * 2.0;
            }
        }
        let k = 8;
        let em = cluster_emeans(&s, n, k, dim, 6, 99);
        let mut owned = vec![0usize; k];
        for i in 0..n {
            owned[nearest(&s[i * dim..(i + 1) * dim], &em.centroids, k, dim)] += 1;
        }
        let dead = owned.iter().filter(|&&c| c == 0).count();
        assert_eq!(dead, 0, "revival must leave no dead clusters, got {dead}");
    }

    #[test]
    fn edge_cases() {
        // n == 0
        let r = cluster_emeans(&[], 0, 4, 8, 4, 1);
        assert_eq!(r.centroids, vec![0.0f32; 32]);
        // k == 1
        let (s, n) = make_blobs(1, 100, 8, 10.0);
        let r = cluster_emeans(&s, n, 1, 8, 4, 1);
        assert_eq!(r.centroids.len(), 8);
        // k >= n: no panic, centroids sized k*dim
        let r = cluster_emeans(&s[0..5 * 8], 5, 8, 8, 3, 1);
        assert_eq!(r.centroids.len(), 64);
    }
}
