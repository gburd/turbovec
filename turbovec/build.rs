//! pg_turbovec fork carry #5: choose how `search.rs` emits `vpdpbusd`.
//!
//! stdarch's `_mm512_dpbusd_epi32` binds `llvm.x86.avx512.vpdpbusd.512` with
//! the `<64 x i8>` operand types that LLVM 22 introduced. A rustc whose stdarch
//! carries that binding but which links an older LLVM -- nixpkgs' rustc
//! 1.97/1.98 on LLVM 21, for one -- rejects the call ("intrinsic signature
//! mismatch") and the crate does not compile. On such a compiler the same
//! instruction is emitted through inline asm instead (`dpbusd!` in
//! `search.rs`). On LLVM >= 22 nothing changes: the intrinsic is used as before.
//!
//! `TURBOVEC_DPBUSD_ASM=1` forces the asm path, so it can be tested on an
//! LLVM 22 compiler.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TURBOVEC_DPBUSD_ASM");
    println!("cargo:rustc-check-cfg=cfg(turbovec_dpbusd_asm)");
    let forced = std::env::var("TURBOVEC_DPBUSD_ASM").is_ok_and(|v| v == "1");
    if forced || llvm_major().is_some_and(|major| major < 22) {
        println!("cargo:rustc-cfg=turbovec_dpbusd_asm");
    }
}

/// LLVM major version of the compiler cargo is building with, from
/// `$RUSTC -vV` ("LLVM version: 21.1.8"). `None` if it can't be read, which
/// keeps the intrinsic (the pre-carry behaviour).
fn llvm_major() -> Option<u32> {
    let rustc = std::env::var_os("RUSTC")?;
    let out = Command::new(rustc).arg("-vV").output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let version = text
        .lines()
        .find_map(|l| l.strip_prefix("LLVM version: "))?;
    version.split('.').next()?.trim().parse().ok()
}
