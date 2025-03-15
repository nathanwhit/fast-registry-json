use std::hint::black_box;

use deno_npm::registry::NpmPackageInfo;
use divan::Bencher;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

#[divan::bench]
fn bench_pluck_versions(b: Bencher) {
    let input = std::fs::read_to_string(
        "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/next/registry.json",
    )
    .unwrap();
    b.bench(|| black_box(fast_registry_json::pluck_versions(&input)));
}

#[divan::bench]
fn bench_parse_versions(b: Bencher) {
    let input: String = std::fs::read_to_string(
        "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/next/registry.json",
    )
    .unwrap();
    b.bench(|| {
        black_box({
            let info: NpmPackageInfo = serde_json::from_str(&input).unwrap();
            // info.versions.keys().cloned().collect::<Vec<_>>()
            info
        })
    });
}
