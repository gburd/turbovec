//! pg_turbovec fork carry #7 (Phase S2a): a position-independent prepared
//! image + a borrowed-view searcher.
//!
//! # Why this exists
//!
//! A searchable [`TurboQuantIndex`](crate::TurboQuantIndex) is a graph of
//! owned `Vec`s. A `Vec` owns its allocation and frees it on `Drop`, so it
//! cannot be constructed over foreign / `mmap`'d memory, and there is no
//! public constructor that returns a *searchable view over borrowed bytes*.
//! That is the exact gap the S2 feasibility spike documented
//! (`.agent/notes/S2_SPIKE_FINDINGS.md` §6): to serve ONE decoded index
//! read-only from Postgres shared memory to MANY backends, turbovec must grow
//!
//!   1. a way to emit the *decoded* search structures as a single
//!      contiguous, pointer-free blob of `(offset, len)` sections relative to
//!      a base, so a segment builder can `memcpy` it into a DSA/DSM region
//!      once ([`TurboQuantIndex::serialize_prepared`] → [`PiImage`]); and
//!   2. a search entry that reads those structures through *borrowed slices*
//!      — never owning or freeing them — so N processes can each build a view
//!      over their own mapping of the same bytes at their own base
//!      ([`PreparedView`] + [`search_from_prepared`]).
//!
//! # What is and is not in the blob
//!
//! The blob holds exactly the row-dependent search state the scoring kernel
//! reads: the SIMD-blocked codes (`BlockedCache::data`, and `low` under the
//! planes layout), the per-vector `scales`, the TQ+ calibration
//! (`tqplus_shift` / `tqplus_scale`), and the scalar metadata
//! (`bit_width`, `dim`, `n_vectors`, `n_blocks`, `is_planes`).
//!
//! The rotation matrix and the Lloyd-Max codebook are NOT in the blob: they
//! are pure deterministic functions of `(bit_width, dim)`, so each process
//! rebuilds them on [`PreparedView::new`] for microseconds — the same thing
//! [`TurboQuantIndex::prepare`] does. The planes derived sub-caches
//! (`stats` / `sample` / `centre`) are likewise rederived on the view from
//! the blob's own bytes.
//!
//! # Position independence
//!
//! Every section is stored as `(offset, len)` relative to the blob start, so
//! a [`PreparedView`] built over a copy of the bytes at a *different* base
//! resolves and searches identically — proven by
//! `search_from_prepared_is_byte_identical_at_a_different_base`. The float
//! sections are stored at 4-byte-aligned offsets; `PreparedView::new` checks
//! the base pointer is 4-aligned (a DSA/DSM/`mmap` segment is page-aligned,
//! so this always holds) and resolves `&[f32]` views over them with no copy.
//!
//! # Bit-identity contract
//!
//! [`search_from_prepared`] feeds the identical operands to the identical
//! `search::search` the owned [`TurboQuantIndex::search_with_mask`] path uses.
//! The candidate set and the scores are therefore bit-identical to the owned
//! path for the same index + query + k — guarded by
//! `search_from_prepared_matches_owned_search_bit_for_bit`.

use crate::codebook;
use crate::pack;
use crate::rotation::Rotation;
use crate::search::{self, PlanesRef};
use crate::{TurboQuantIndex, BLOCK};

/// Magic bytes at the head of a [`PiImage`]: "TVPREP" + format version.
const MAGIC: [u8; 8] = *b"TVPREP01";
/// Blob format version. Bumped if the section layout changes.
const PI_VERSION: u32 = 1;

const FLAG_PLANES: u32 = 1 << 0;

/// Byte offset of each `(offset, len)` section pair in the header, in the
/// order they are written. Keep in sync with [`write_sections`].
const SEC_DATA: usize = 0;
const SEC_LOW: usize = 1;
const SEC_SCALES: usize = 2;
const SEC_TQSHIFT: usize = 3;
const SEC_TQSCALE: usize = 4;
const N_SECTIONS: usize = 5;

/// Fixed header size: magic(8) + version(4) + bit_width(4) + dim(4) +
/// flags(4) + n_vectors(8) + n_blocks(8) + N_SECTIONS * (off:8 + len:8).
const HEADER_LEN: usize = 8 + 4 + 4 + 4 + 4 + 8 + 8 + N_SECTIONS * 16;

/// A position-independent, pointer-free image of a prepared index's decoded
/// search structures.
///
/// The bytes are self-describing: a fixed header of scalars + a table of
/// `(offset, len)` sections, followed by the section payloads. Every offset
/// is relative to byte 0 of the image, so the whole thing can be `memcpy`'d
/// into a shared-memory segment at any base and read back by a
/// [`PreparedView`] built over that mapping.
///
/// Build one with [`TurboQuantIndex::serialize_prepared`]; consume
/// `into_bytes()` / `as_bytes()` to copy it into your segment.
#[derive(Clone)]
pub struct PiImage {
    bytes: Vec<u8>,
}

impl std::fmt::Debug for PiImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PiImage")
            .field("len", &self.bytes.len())
            .finish()
    }
}

impl PiImage {
    /// The image as a contiguous byte slice. `memcpy` this into your
    /// shared-memory segment.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the image, returning the owned bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Total size of the image in bytes — what a segment builder must
    /// allocate.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the image carries no searchable rows (an empty or lazy
    /// index). A view over such an image searches to an empty result.
    pub fn is_empty(&self) -> bool {
        // n_vectors == 0 marks the empty/lazy case; see serialize_prepared.
        read_u64(&self.bytes, 24) == 0
    }
}

/// A borrowed, read-only view over a [`PiImage`]'s bytes.
///
/// Holds no owned copy of the big buffers: `data`, `low` and `scales` are
/// slices resolved from `base + offset` of the backing byte slice. The only
/// things it owns are the two tiny deterministic rebuilds — the rotation
/// matrix and the codebook — plus, under the planes layout, the small derived
/// sub-caches (`stats` / `sample` / `centre`), exactly as a cold owned search
/// would build them on first call.
///
/// Construct one with [`PreparedView::new`] over your own mapping of the
/// image bytes; many processes can each hold their own `PreparedView` over
/// the same shared segment at their own base, all searching read-only.
pub struct PreparedView<'a> {
    bits: usize,
    dim: usize,
    n_vectors: usize,
    n_blocks: usize,
    is_planes: bool,

    // Borrowed, zero-copy sections resolved from the backing bytes.
    data: &'a [u8],
    low: &'a [u8],
    scales: &'a [f32],
    tqplus_shift: &'a [f32],
    tqplus_scale: &'a [f32],

    // Deterministic rebuilds, owned by the view (cheap, pure fn of
    // (bits, dim)) — identical to what prepare() derives.
    rotation: Rotation,
    centroids: Vec<f32>,
}

/// Errors from constructing a [`PreparedView`] over untrusted bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PreparedViewError {
    /// Buffer too short to hold the fixed header.
    TooShort { got: usize, need: usize },
    /// Magic bytes did not match — not a `PiImage`.
    BadMagic,
    /// Image format version this build does not understand.
    UnsupportedVersion(u32),
    /// `bit_width` outside the supported `2..=4`.
    BitWidthOutOfRange(usize),
    /// `dim` not a positive multiple of 8.
    DimNotPositiveMultipleOf8(usize),
    /// A section `(offset, len)` runs past the end of the buffer, or a float
    /// section's length is not a whole number of `f32`s.
    SectionOutOfBounds { section: usize },
    /// A float section's byte offset is not 4-aligned within the image, or
    /// the base pointer of the buffer is not 4-aligned, so a `&[f32]` view
    /// cannot be taken without a copy.
    Misaligned { section: usize },
    /// A section's declared length does not match the geometry implied by the
    /// scalar header (e.g. `scales.len() != n_vectors`).
    LengthMismatch { section: usize },
}

impl std::fmt::Display for PreparedViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { got, need } => {
                write!(f, "prepared image too short: {got} bytes, need >= {need}")
            }
            Self::BadMagic => write!(f, "prepared image bad magic (not a PiImage)"),
            Self::UnsupportedVersion(v) => write!(f, "prepared image version {v} unsupported"),
            Self::BitWidthOutOfRange(b) => write!(f, "prepared image bit_width {b} out of range"),
            Self::DimNotPositiveMultipleOf8(d) => {
                write!(f, "prepared image dim {d} not a positive multiple of 8")
            }
            Self::SectionOutOfBounds { section } => {
                write!(f, "prepared image section {section} out of bounds")
            }
            Self::Misaligned { section } => {
                write!(
                    f,
                    "prepared image section {section} misaligned for f32 view"
                )
            }
            Self::LengthMismatch { section } => {
                write!(f, "prepared image section {section} length mismatch")
            }
        }
    }
}

impl std::error::Error for PreparedViewError {}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn read_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Resolve a borrowed `&[u8]` section from `(offset, len)` against the image.
fn resolve_u8(
    image: &[u8],
    off: usize,
    len: usize,
    section: usize,
) -> Result<&[u8], PreparedViewError> {
    let end = off
        .checked_add(len)
        .ok_or(PreparedViewError::SectionOutOfBounds { section })?;
    if end > image.len() {
        return Err(PreparedViewError::SectionOutOfBounds { section });
    }
    Ok(&image[off..end])
}

/// Resolve a borrowed `&[f32]` section — zero-copy — from `(offset, byte_len)`
/// against the image. Requires the image base and the section offset both
/// 4-aligned, and `byte_len` a whole number of `f32`s. A DSA/DSM/`mmap`
/// segment is page-aligned, so the base check always passes in production;
/// `serialize_prepared` 4-aligns every float section offset.
fn resolve_f32(
    image: &[u8],
    off: usize,
    byte_len: usize,
    section: usize,
) -> Result<&[f32], PreparedViewError> {
    let bytes = resolve_u8(image, off, byte_len, section)?;
    if byte_len % 4 != 0 {
        return Err(PreparedViewError::LengthMismatch { section });
    }
    // The base of the whole image must be 4-aligned, and the section offset
    // within it must be 4-aligned, for the resolved pointer to be aligned.
    if (image.as_ptr() as usize) % 4 != 0 || off % 4 != 0 {
        return Err(PreparedViewError::Misaligned { section });
    }
    let n = byte_len / 4;
    // SAFETY: `bytes` is in-bounds (resolve_u8), covers `n * 4` bytes, is
    // 4-aligned (checked above), and f32 has no invalid bit patterns, no
    // padding, and the same size/no-drop as [u8;4]. The lifetime is tied to
    // `image`, so the view never outlives the backing bytes. Read-only.
    let ptr = bytes.as_ptr() as *const f32;
    Ok(unsafe { std::slice::from_raw_parts(ptr, n) })
}

impl<'a> PreparedView<'a> {
    /// Build a borrowed view over a [`PiImage`]'s bytes.
    ///
    /// `image` is the raw bytes of a `PiImage` — your own mapping of the
    /// shared segment. The view borrows the big buffers directly out of it
    /// (no copy) and rebuilds the small deterministic structures. The bytes
    /// must outlive the view.
    ///
    /// Validates the header and every section against the scalar geometry, so
    /// a corrupt or truncated image is rejected with a typed error rather
    /// than read out of bounds.
    pub fn new(image: &'a [u8]) -> Result<Self, PreparedViewError> {
        if image.len() < HEADER_LEN {
            return Err(PreparedViewError::TooShort {
                got: image.len(),
                need: HEADER_LEN,
            });
        }
        if image[0..8] != MAGIC {
            return Err(PreparedViewError::BadMagic);
        }
        let version = read_u32(image, 8);
        if version != PI_VERSION {
            return Err(PreparedViewError::UnsupportedVersion(version));
        }
        let bits = read_u32(image, 12) as usize;
        let dim = read_u32(image, 16) as usize;
        let flags = read_u32(image, 20);
        let n_vectors = read_u64(image, 24) as usize;
        let n_blocks = read_u64(image, 32) as usize;
        let is_planes = flags & FLAG_PLANES != 0;

        if !(2..=4).contains(&bits) {
            return Err(PreparedViewError::BitWidthOutOfRange(bits));
        }
        if dim == 0 || dim % 8 != 0 {
            return Err(PreparedViewError::DimNotPositiveMultipleOf8(dim));
        }

        // Section table starts right after the scalar fields.
        let table = 40;
        let sec = |i: usize| -> (usize, usize) {
            let base = table + i * 16;
            (
                read_u64(image, base) as usize,
                read_u64(image, base + 8) as usize,
            )
        };

        let (do_, dl) = sec(SEC_DATA);
        let (lo, ll) = sec(SEC_LOW);
        let (so, sl) = sec(SEC_SCALES);
        let (sho, shl) = sec(SEC_TQSHIFT);
        let (sco, scl) = sec(SEC_TQSCALE);

        let data = resolve_u8(image, do_, dl, SEC_DATA)?;
        let low = resolve_u8(image, lo, ll, SEC_LOW)?;
        let scales = resolve_f32(image, so, sl, SEC_SCALES)?;
        let tqplus_shift = resolve_f32(image, sho, shl, SEC_TQSHIFT)?;
        let tqplus_scale = resolve_f32(image, sco, scl, SEC_TQSCALE)?;

        // Geometry checks mirroring from_parts: scales has one entry per
        // vector; the calibration arrays are either empty or exactly `dim`
        // and equal length.
        if scales.len() != n_vectors {
            return Err(PreparedViewError::LengthMismatch {
                section: SEC_SCALES,
            });
        }
        if tqplus_shift.len() != tqplus_scale.len() {
            return Err(PreparedViewError::LengthMismatch {
                section: SEC_TQSCALE,
            });
        }
        if !tqplus_shift.is_empty() && tqplus_shift.len() != dim {
            return Err(PreparedViewError::LengthMismatch {
                section: SEC_TQSHIFT,
            });
        }

        // Geometry: `search::search` reads `data`/`low` with unchecked SIMD
        // loads sized from `n_vectors`/`n_blocks`, so in-bounds sections are
        // not enough — the section LENGTHS must be exactly what the owned
        // build (`BlockedCache::build`, `pack::repack`/`planes_repack`)
        // produces for this geometry. A torn or buggy segment build must be
        // rejected here, never searched.
        if n_vectors > 0 {
            let (want_blocks, nbg, classic_bytes) = pack::blocked_geometry(n_vectors, bits, dim);
            if n_blocks != want_blocks {
                return Err(PreparedViewError::LengthMismatch { section: SEC_DATA });
            }
            let (want_data, want_low) = if is_planes {
                if !pack::planes_geom_ok(bits, nbg) {
                    return Err(PreparedViewError::LengthMismatch { section: SEC_LOW });
                }
                let (nsg, low_row) = pack::planes_geom(bits, nbg);
                (n_blocks * nsg * BLOCK, n_vectors * low_row)
            } else {
                (classic_bytes, 0)
            };
            if data.len() != want_data {
                return Err(PreparedViewError::LengthMismatch { section: SEC_DATA });
            }
            if low.len() != want_low {
                return Err(PreparedViewError::LengthMismatch { section: SEC_LOW });
            }
        } else if n_blocks != 0 || !data.is_empty() || !low.is_empty() || is_planes {
            return Err(PreparedViewError::LengthMismatch { section: SEC_DATA });
        }

        // Rebuild the deterministic structures — pure functions of
        // (bit_width, dim), identical to what prepare()/search() derive.
        let rotation = Rotation::new(dim);
        let (_, centroids) = codebook::codebook(bits, dim);

        Ok(Self {
            bits,
            dim,
            n_vectors,
            n_blocks,
            is_planes,
            data,
            low,
            scales,
            tqplus_shift,
            tqplus_scale,
            rotation,
            centroids,
        })
    }

    /// Vector dimensionality of the viewed index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bit width of the viewed index.
    pub fn bit_width(&self) -> usize {
        self.bits
    }

    /// Number of vectors in the viewed index.
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Whether the viewed index has no rows.
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }
}

/// Search a prepared index through a borrowed [`PreparedView`], returning
/// `(scores, slot_indices)` — the SAME `(Vec<f32>, Vec<i64>)` shape and the
/// SAME values that [`TurboQuantIndex::search`] returns for the same index +
/// `queries` + `k`.
///
/// The scoring reads the blocked codes and scales through the view's borrowed
/// slices and never owns or frees them, so N processes can each call this over
/// their own mapping of one shared image. The result is bit-identical to the
/// owned path — the exact-distance recheck on the pg_turbovec side is a
/// release contract, and this entry must not perturb the candidate set.
///
/// `queries` is `nq * dim` coordinates row-major. Panics on a ragged or
/// non-finite query, exactly like [`TurboQuantIndex::search`]; use
/// [`try_search_from_prepared`] for the non-panicking form.
pub fn search_from_prepared(
    view: &PreparedView<'_>,
    queries: &[f32],
    k: usize,
) -> (Vec<f32>, Vec<i64>) {
    try_search_from_prepared(view, queries, k).unwrap_or_else(|e| panic!("{e}"))
}

/// [`search_from_prepared`] as a `Result`: the non-panicking form. Mirrors
/// [`TurboQuantIndex::try_search`]'s validation (ragged / non-finite query).
pub fn try_search_from_prepared(
    view: &PreparedView<'_>,
    queries: &[f32],
    k: usize,
) -> Result<(Vec<f32>, Vec<i64>), crate::SearchError> {
    let dim = view.dim;
    let nq = queries.len() / dim;
    if queries.len() != nq * dim {
        return Err(crate::SearchError::QueryBufferNotMultipleOfDim {
            queries_len: queries.len(),
            dim,
        });
    }
    if let Some((vi, ci, v)) = crate::first_invalid_coord(queries, dim) {
        return Err(crate::SearchError::InvalidQueryValue {
            query_index: vi,
            coord_index: ci,
            value: v,
        });
    }
    if view.n_vectors == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Rebuild the planes derived sub-caches from the view's own borrowed
    // bytes — the exact computation the cold owned path performs in
    // search_with_mask (lib.rs). Owned here because they are tiny derived
    // state, not the big buffers; the big buffers (data/low/scales) stay
    // borrowed and uncopied.
    let (planes_low, planes_sample, planes_centre);
    let planes = if view.is_planes {
        debug_assert!(view.bits == 2 || view.bits == 4);
        let nbg = dim / (8 / view.bits);
        let stats = pack::planes_stats(
            view.data,
            view.low,
            view.n_vectors,
            view.bits,
            nbg,
            &view.centroids,
        );
        planes_low = view.low;
        planes_sample = pack::planes_sample(view.data, view.scales, view.n_vectors, dim / 8);
        let centre = if view.tqplus_shift.is_empty() {
            None
        } else {
            let w: Vec<f32> = (0..dim)
                .map(|d| stats.m * view.tqplus_shift[d] / view.tqplus_scale[d])
                .collect();
            planes_centre = pack::planes_centre(view.data, view.n_vectors, dim / 8, &w);
            Some(planes_centre.as_slice())
        };
        Some(PlanesRef {
            low: planes_low,
            stats,
            sample: planes_sample
                .as_ref()
                .map(|(c, s)| (c.as_slice(), s.as_slice())),
            centre,
        })
    } else {
        None
    };

    let (scores, indices) = search::search(
        queries,
        nq,
        &view.rotation,
        view.data,
        &view.centroids,
        view.scales,
        view.tqplus_shift,
        view.tqplus_scale,
        view.bits,
        dim,
        view.n_vectors,
        view.n_blocks,
        k,
        None,
        planes,
    );
    Ok((scores, indices))
}

fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// Append `src` to `blob`, 4-aligning the start first (padding with zero),
/// and return the `(offset, byte_len)` of the appended region.
fn append_aligned_f32(blob: &mut Vec<u8>, src: &[f32]) -> (u64, u64) {
    while blob.len() % 4 != 0 {
        blob.push(0);
    }
    let off = blob.len() as u64;
    for &x in src {
        blob.extend_from_slice(&x.to_le_bytes());
    }
    (off, (src.len() * 4) as u64)
}

/// Append raw bytes to `blob` and return `(offset, len)`.
fn append_u8(blob: &mut Vec<u8>, src: &[u8]) -> (u64, u64) {
    let off = blob.len() as u64;
    blob.extend_from_slice(src);
    (off, src.len() as u64)
}

impl TurboQuantIndex {
    /// Emit the DECODED search structures of a prepared index as a single
    /// contiguous, pointer-free [`PiImage`] — a `(offset, len)`-sectioned
    /// blob that a segment builder can `memcpy` into a DSA/DSM region once,
    /// to be read read-only by many backends through [`PreparedView`].
    ///
    /// This is additive: it does not change `new` / `add` / `from_parts` /
    /// `from_bytes` / `prepare` / `search`, and does not mutate the index. It
    /// warms the search caches first (like [`Self::prepare`]) so the blocked
    /// layout exists to serialize; an empty or lazy index yields a valid
    /// image whose [`PreparedView`] searches to the empty result.
    ///
    /// The blob holds the SIMD-blocked codes, the per-vector scales, the TQ+
    /// calibration and the scalar geometry. It deliberately omits the
    /// rotation and codebook: those are pure functions of `(bit_width, dim)`
    /// and are rebuilt for free on the view side (the same way `prepare`
    /// derives them), so they never need to cross into shared memory.
    pub fn serialize_prepared(&self) -> PiImage {
        // Warm the caches so `blocked` exists — the same work `prepare`
        // does, and a no-op if `prepare`/`search` already ran. Takes `&self`
        // via OnceLock, so this stays an immutable, additive accessor.
        self.prepare();

        let bits = self.bit_width();
        // dim/n may be unset (lazy) or zero (empty): emit a well-formed empty
        // image. A PreparedView over it short-circuits to the empty result.
        // Child modules may read the crate-root struct's private fields, so
        // the blocked cache and the calibration arrays are read directly.
        // The empty-image arm uses 'static empty slices, which coerce to the
        // borrow lifetime of the populated arm.
        let empty_f32: &[f32] = &[];
        let empty_u8: &[u8] = &[];
        let (dim, n_vectors, n_blocks, is_planes, data, low, scales, tqshift, tqscale) =
            match (self.dim, self.n_vectors, self.blocked.get()) {
                (Some(dim), n_vectors, Some(blocked)) if n_vectors > 0 => (
                    dim,
                    n_vectors,
                    blocked.n_blocks,
                    blocked.is_planes(),
                    blocked.data.as_slice(),
                    blocked.low.as_slice(),
                    self.scales.as_slice(),
                    self.tqplus_shift.as_slice(),
                    self.tqplus_scale.as_slice(),
                ),
                _ => (
                    self.dim.unwrap_or(0),
                    0usize,
                    0usize,
                    false,
                    empty_u8,
                    empty_u8,
                    empty_f32,
                    empty_f32,
                    empty_f32,
                ),
            };

        let mut flags = 0u32;
        if is_planes {
            flags |= FLAG_PLANES;
        }

        // Header: fixed scalars, then the section table (filled after we know
        // the offsets). Reserve the table space first so payloads start after
        // the header and offsets are stable.
        let mut blob: Vec<u8> = Vec::with_capacity(
            HEADER_LEN
                + data.len()
                + low.len()
                + scales.len() * 4
                + (tqshift.len() + tqscale.len()) * 4
                + 32,
        );
        blob.extend_from_slice(&MAGIC);
        push_u32(&mut blob, PI_VERSION);
        push_u32(&mut blob, bits as u32);
        push_u32(&mut blob, dim as u32);
        push_u32(&mut blob, flags);
        push_u64(&mut blob, n_vectors as u64);
        push_u64(&mut blob, n_blocks as u64);
        debug_assert_eq!(blob.len(), 40);
        // Reserve the section table (N_SECTIONS * 16 bytes), fill later.
        let table_at = blob.len();
        blob.resize(table_at + N_SECTIONS * 16, 0);
        debug_assert_eq!(blob.len(), HEADER_LEN);

        // Payloads. Byte sections (data, low) can land at any offset; float
        // sections are 4-aligned so the view can borrow &[f32] zero-copy.
        let sec_data = append_u8(&mut blob, data);
        let sec_low = append_u8(&mut blob, low);
        let sec_scales = append_aligned_f32(&mut blob, scales);
        let sec_tqshift = append_aligned_f32(&mut blob, tqshift);
        let sec_tqscale = append_aligned_f32(&mut blob, tqscale);

        write_sections(
            &mut blob,
            table_at,
            &[sec_data, sec_low, sec_scales, sec_tqshift, sec_tqscale],
        );

        PiImage { bytes: blob }
    }
}

/// Write the `(offset, len)` section table into the reserved header region.
fn write_sections(blob: &mut [u8], table_at: usize, sections: &[(u64, u64)]) {
    debug_assert_eq!(sections.len(), N_SECTIONS);
    for (i, (off, len)) in sections.iter().enumerate() {
        let at = table_at + i * 16;
        blob[at..at + 8].copy_from_slice(&off.to_le_bytes());
        blob[at + 8..at + 16].copy_from_slice(&len.to_le_bytes());
    }
}
