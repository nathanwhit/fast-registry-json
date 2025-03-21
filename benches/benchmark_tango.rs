use std::hint::black_box;

use deno_npm::registry::{NpmPackageInfo, NpmPackageVersionInfo};
use deno_semver::Version;
use rustc_hash::FxHashMap;
use tango_bench::{benchmark_fn, tango_main};

const NEXT_PATH: &str =
    "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/next/registry.json";

fn bench_pluck_versions(input: &str) -> NpmPackageVersionInfo {
    let versions = fast_registry_json::pluck_versions(input).unwrap();
    let mut map = FxHashMap::with_capacity_and_hasher(versions.versions.len(), Default::default());
    for (version, range) in versions.versions.into_iter().zip(versions.version_ranges) {
        map.insert(version, range);
    }
    let &(start, end) = map.get("15.0.0-canary.202").unwrap();
    let info: NpmPackageVersionInfo =
        serde_json::from_str(&input[start as usize..end as usize]).unwrap();
    assert_eq!(info.version.to_string(), "15.0.0-canary.202");
    info
}

fn bench_parse_versions(input: &str, version: &Version) -> NpmPackageVersionInfo {
    let info: NpmPackageInfo = serde_json::from_str(&input).unwrap();
    let version_info = info.versions.get(version).cloned().unwrap();
    assert_eq!(version_info.version.to_string(), version.to_string());
    version_info
}

const PRISMA_PATH: &str =
    "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/@prisma/client/registry.json";

fn bench_pluck_versions_prisma(input: &str) -> NpmPackageVersionInfo {
    let versions = fast_registry_json::pluck_versions(&input).unwrap();
    let mut map = FxHashMap::with_capacity_and_hasher(versions.versions.len(), Default::default());
    for (version, range) in versions.versions.into_iter().zip(versions.version_ranges) {
        map.insert(version, range);
    }
    let &(start, end) = map.get("5.1.0-dev.64").unwrap();
    let info: NpmPackageVersionInfo =
        serde_json::from_str(&input[start as usize..end as usize]).unwrap();
    assert_eq!(info.version.to_string(), "5.1.0-dev.64");
    info
}

fn bench_parse_versions_prisma(input: &str, version: &Version) -> NpmPackageVersionInfo {
    let info: NpmPackageInfo = serde_json::from_str(&input).unwrap();
    let version_info = info.versions.get(&version).cloned().unwrap();
    assert_eq!(version_info.version.to_string(), version.to_string());
    version_info
}

fn next_benchmarks() -> impl tango_bench::IntoBenchmarks {
    [
        benchmark_fn("next_pluck_versions", move |b| {
            let input = std::fs::read_to_string(NEXT_PATH).unwrap();
            b.iter(move || {
                // let input = input.clone();
                let _ = black_box(bench_pluck_versions(&input));
            })
        }),
        benchmark_fn("next_parse_versions", |b| {
            let input = std::fs::read_to_string(NEXT_PATH).unwrap();
            let version = deno_semver::Version::parse_from_npm("15.0.0-canary.202").unwrap();
            b.iter(move || {
                // let input = input.clone();
                // let version = version.clone();
                let _ = black_box(bench_parse_versions(&input, &version));
            })
        }),
    ]
}

fn prisma_benchmarks() -> impl tango_bench::IntoBenchmarks {
    [
        benchmark_fn("prisma_pluck_versions", move |b| {
            let input = std::fs::read_to_string(PRISMA_PATH).unwrap();
            b.iter(move || {
                let _ = black_box(bench_pluck_versions_prisma(&input));
            })
        }),
        benchmark_fn("prisma_parse_versions", |b| {
            let input = std::fs::read_to_string(PRISMA_PATH).unwrap();
            let version = deno_semver::Version::parse_from_npm("5.1.0-dev.64").unwrap();
            b.iter(move || {
                let _ = black_box(bench_parse_versions_prisma(&input, &version));
            })
        }),
    ]
}

tango_bench::tango_benchmarks!(next_benchmarks(), prisma_benchmarks());
tango_main!();
