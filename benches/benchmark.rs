use std::hint::black_box;

use deno_npm::registry::{NpmPackageInfo, NpmPackageVersionInfo};
use divan::Bencher;
use rustc_hash::FxHashMap;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

const NEXT_PATH: &str =
    "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/next/registry.json";

#[divan::bench]
fn bench_pluck_versions(b: Bencher) {
    let input = std::fs::read_to_string(NEXT_PATH).unwrap();
    b.bench(|| {
        black_box({
            let versions = fast_registry_json::pluck_versions(&input).unwrap();
            let mut map =
                FxHashMap::with_capacity_and_hasher(versions.versions.len(), Default::default());
            for (version, range) in versions.versions.into_iter().zip(versions.version_ranges) {
                map.insert(version, range);
            }
            let &(start, end) = map.get("15.0.0-canary.202").unwrap();
            let info: NpmPackageVersionInfo =
                serde_json::from_str(&input[start as usize..end as usize]).unwrap();
            assert_eq!(info.version.to_string(), "15.0.0-canary.202");
            info
        })
    });
}

#[divan::bench]
fn bench_parse_versions(b: Bencher) {
    let input = std::fs::read_to_string(NEXT_PATH).unwrap();
    let version = deno_semver::Version::parse_from_npm("15.0.0-canary.202").unwrap();
    b.bench(|| {
        black_box({
            let info: NpmPackageInfo = serde_json::from_str(&input).unwrap();
            let version_info = info.versions.get(&version).cloned().unwrap();
            assert_eq!(version_info.version.to_string(), "15.0.0-canary.202");
            version_info
        })
    });
}

const PRISMA_PATH: &str =
    "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/@prisma/client/registry.json";

#[divan::bench]
fn bench_pluck_versions_prisma(b: Bencher) {
    let input = std::fs::read_to_string(PRISMA_PATH).unwrap();
    b.bench(|| {
        black_box({
            let versions = fast_registry_json::pluck_versions(&input).unwrap();
            let mut map =
                FxHashMap::with_capacity_and_hasher(versions.versions.len(), Default::default());
            for (version, range) in versions.versions.into_iter().zip(versions.version_ranges) {
                map.insert(version, range);
            }
            let &(start, end) = map.get("5.1.0-dev.64").unwrap();
            let info: NpmPackageVersionInfo =
                serde_json::from_str(&input[start as usize..end as usize]).unwrap();
            assert_eq!(info.version.to_string(), "5.1.0-dev.64");
            info
        })
    });
}

#[divan::bench]
fn bench_parse_versions_prisma(b: Bencher) {
    let input = std::fs::read_to_string(PRISMA_PATH).unwrap();
    let version = deno_semver::Version::parse_from_npm("5.1.0-dev.64").unwrap();
    b.bench(|| {
        black_box({
            let info: NpmPackageInfo = serde_json::from_str(&input).unwrap();
            let version_info = info.versions.get(&version).cloned().unwrap();
            assert_eq!(version_info.version.to_string(), "5.1.0-dev.64");
            version_info
        })
    });
}
