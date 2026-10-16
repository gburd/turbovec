//! pg_turbovec fork carry #7 (Phase S2a): the prepared-image + borrowed-view
//! searcher.
//!
//! Proves the S2a contract from `.agent/notes/S2_SPIKE_FINDINGS.md` §6:
//! `serialize_prepared` emits a position-independent, pointer-free blob of the
//! decoded search structures, and `search_from_prepared` searches it through
//! borrowed slices with a candidate set + scores BIT-IDENTICAL to the owned
//! `TurboQuantIndex::search`.

use turbovec::prepared::{search_from_prepared, PreparedView};
use turbovec::TurboQuantIndex;

const DIM: usize = 128;
const BITS: usize = 4;
const K: usize = 10;

/// Cheap deterministic pseudo-random vectors (xorshift), matching the spike
/// prototype's generator so the data distribution is the same shape.
fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0;
        out.push(v);
    }
    out
}

fn l2_normalize_rows(v: &mut [f32], dim: usize) {
    for row in v.chunks_mut(dim) {
        let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in row.iter_mut() {
                *x /= n;
            }
        }
    }
}

fn build_index(n: usize, seed: u64) -> TurboQuantIndex {
    let mut vecs = make_vectors(n, DIM, seed);
    l2_normalize_rows(&mut vecs, DIM);
    let mut idx = TurboQuantIndex::new(DIM, BITS).expect("new");
    idx.add(&vecs);
    idx.prepare();
    idx
}

/// Copy a byte image into a FRESH allocation at a (very likely) different base
/// address, to prove the view is position-independent — a `PreparedView` built
/// over the copy must resolve and search identically to one over the original.
fn relocated_copy(bytes: &[u8]) -> Vec<u8> {
    // A large decoy allocation first, so the clone lands at a different VA
    // than the serializer's buffer. Position independence must not depend on
    // the two living at the same address.
    let _decoy: Vec<u8> = vec![0u8; 64 * 1024 * 1024];
    let mut copy = Vec::with_capacity(bytes.len());
    copy.extend_from_slice(bytes);
    assert_ne!(
        copy.as_ptr() as usize,
        bytes.as_ptr() as usize,
        "copy must be at a different base address"
    );
    copy
}

/// The core S2a contract: the borrowed-view search over a serialized,
/// RELOCATED image returns the SAME ids and SAME scores, bit for bit, as the
/// owned `search`. Run for both the classic blocked layout (small index) and
/// the planes layout (large index, exercises the `low` region).
fn assert_bit_identical(n: usize, seed: u64) {
    let idx = build_index(n, seed);

    let mut queries = make_vectors(2, DIM, seed ^ 0xDEAD_BEEF);
    l2_normalize_rows(&mut queries, DIM);

    // Owned baseline.
    let owned = idx.search(&queries, K);

    // Serialize the decoded structures to a pointer-free blob.
    let image = idx.serialize_prepared();
    let orig_bytes = image.as_bytes().to_vec();

    // Build a view over a COPY at a different allocation — position
    // independence. The original image buffer is dropped before searching the
    // copy, so a lingering pointer into it would fault or go stale.
    let relocated = relocated_copy(&orig_bytes);
    drop(image);

    let view = PreparedView::new(&relocated).expect("view over relocated image");
    assert_eq!(view.dim(), DIM);
    assert_eq!(view.bit_width(), BITS);
    assert_eq!(view.len(), n);

    let (scores, indices) = search_from_prepared(&view, &queries, K);

    // Scores are bit-identical (f32 bit patterns), and the candidate set
    // (slot indices) is identical and in the same order.
    assert_eq!(
        indices, owned.indices,
        "n={n}: borrowed-view candidate set differs from owned search"
    );
    assert_eq!(
        scores.len(),
        owned.scores.len(),
        "n={n}: score count differs"
    );
    for (i, (a, b)) in scores.iter().zip(owned.scores.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "n={n}: score {i} differs: view={a} owned={b}"
        );
    }
}

#[test]
fn search_from_prepared_matches_owned_search_bit_for_bit() {
    // Classic (non-planes) blocked layout.
    assert_bit_identical(2_000, 42);
}

#[test]
fn search_from_prepared_is_byte_identical_at_a_different_base() {
    // Same assertion, emphasised: the view is built over a relocated copy, so
    // passing proves position independence. A larger index than the classic
    // test, still below the planes threshold on a non-AVX-512 host.
    assert_bit_identical(8_000, 7);
}

#[test]
fn search_from_prepared_matches_owned_search_in_planes_layout() {
    // >= PLANES_MIN_VECTORS (32_768): on an AVX-512 host this takes the planes
    // layout, so the serialized blob carries a non-empty `low` region and the
    // view rebuilds the planes sub-caches. On a non-AVX-512 host the same
    // index stays in the classic layout and the test still asserts
    // bit-identity -- either way the contract holds. Kept just over the planes
    // threshold (not far past it) so this CPU-heavy case does not saturate all
    // cores alongside cargo's other parallel test binaries -- the process
    // global codebook MEMO's timing-sensitive fork-safety tests re-solve under
    // sustained external load (fork carry #6, MEMO_PATIENCE).
    assert_bit_identical(33_000, 99);
}

#[test]
fn prepared_view_rejects_a_truncated_image() {
    let idx = build_index(1_000, 1);
    let image = idx.serialize_prepared();
    let bytes = image.as_bytes();
    // A header-only prefix must be rejected, not read out of bounds.
    let truncated = &bytes[..bytes.len() - 1];
    assert!(PreparedView::new(truncated).is_err());
    // Bad magic.
    let mut bad = bytes.to_vec();
    bad[0] ^= 0xFF;
    assert!(PreparedView::new(&bad).is_err());
}

#[test]
fn empty_and_lazy_indexes_serialize_and_search_to_empty() {
    // Empty (dim known, no rows).
    let empty = TurboQuantIndex::new(DIM, BITS).expect("new");
    let image = empty.serialize_prepared();
    let view = PreparedView::new(image.as_bytes()).expect("view");
    assert!(view.is_empty());
    let q = vec![0.0f32; DIM];
    let (scores, indices) = search_from_prepared(&view, &q, K);
    assert!(scores.is_empty() && indices.is_empty());
}

/// Section bounds alone are not enough: `search::search` reads the blocked
/// codes with unchecked SIMD loads sized from `n_vectors`/`n_blocks`, so an
/// image whose sections are in-bounds but whose geometry is inconsistent
/// (short `data`, wrong `n_blocks`, planes flag without a `low` region) must be
/// rejected by `PreparedView::new`, never searched. A torn or buggy shared-
/// segment build is exactly how such an image would arrive.
#[test]
fn prepared_view_rejects_inconsistent_geometry() {
    // Header layout (prepared.rs): magic 0..8, version 8..12, bits 12..16,
    // dim 16..20, flags 20..24, n_vectors 24..32, n_blocks 32..40, then the
    // section table at 40: (offset u64, len u64) per section, DATA first.
    let idx = build_index(1_000, 7);
    let good = idx.serialize_prepared().into_bytes();
    assert!(
        PreparedView::new(&good).is_ok(),
        "control: a well-formed image is accepted"
    );

    // 1. n_blocks claims more blocks than `data` holds.
    let mut more_blocks = good.clone();
    let nb = u64::from_le_bytes(more_blocks[32..40].try_into().unwrap());
    more_blocks[32..40].copy_from_slice(&(nb + 1).to_le_bytes());
    assert!(
        PreparedView::new(&more_blocks).is_err(),
        "n_blocks > data must be rejected"
    );

    // 2. `data` section shortened by one block (still in-bounds).
    let mut short_data = good.clone();
    let dl = u64::from_le_bytes(short_data[48..56].try_into().unwrap());
    short_data[48..56].copy_from_slice(&(dl - 32).to_le_bytes());
    assert!(
        PreparedView::new(&short_data).is_err(),
        "short data section must be rejected"
    );

    // 3. planes flag set on a classic image (no `low` region).
    let mut fake_planes = good.clone();
    let flags = u32::from_le_bytes(fake_planes[20..24].try_into().unwrap());
    fake_planes[20..24].copy_from_slice(&(flags | 1).to_le_bytes());
    assert!(
        PreparedView::new(&fake_planes).is_err(),
        "planes flag without low must be rejected"
    );
}
