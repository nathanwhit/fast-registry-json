use std::hint::black_box;

use deno_npm::registry::{NpmPackageInfo, NpmPackageVersionInfo};
use divan::Bencher;
use rustc_hash::FxHashMap;

fn main() {
    // Run registered benchmarks.
    divan::main();
}

macro_rules! bench_parse_pluck {
    ($name: ident, $path: expr, $version: expr) => {
        paste::paste! {
            #[divan::bench]
            fn [<bench_pluck_versions_ $name>](b: Bencher) {
                let input = std::fs::read_to_string($path).unwrap();
                b.bench(|| {
                    black_box({
                        let versions = fast_registry_json::pluck_versions(&input).unwrap();
                        let mut map = FxHashMap::with_capacity_and_hasher(
                            versions.versions.len(),
                            Default::default(),
                        );
                        for (version, range) in
                            versions.versions.into_iter().zip(versions.version_ranges)
                        {
                            map.insert(version, range);
                        }
                        let &(start, end) = map.get($version).unwrap();
                        let info: NpmPackageVersionInfo =
                            serde_json::from_str(&input[start as usize..end as usize]).unwrap();
                        assert_eq!(info.version.to_string(), $version);
                        info
                    })
                });
            }

            #[divan::bench]
            fn [<bench_parse_versions_ $name>](b: Bencher) {
                let input = std::fs::read_to_string($path).unwrap();
                let version = deno_semver::Version::parse_from_npm($version).unwrap();
                b.bench(|| {
                    black_box({
                        let info: NpmPackageInfo = serde_json::from_str(&input).unwrap();
                        let version_info = info.versions.get(&version).cloned().unwrap();
                        assert_eq!(version_info.version.to_string(), $version);
                        version_info
                    })
                });
            }

            #[divan::bench]
            fn [<bench_packument_index_ $name>](b: Bencher) {
                let input = std::fs::read_to_string($path).unwrap();
                b.bench(|| {
                    black_box(fast_registry_json::pluck_packument_index(&input).unwrap())
                });
            }

            #[divan::bench]
            fn [<bench_packument_index_deser_ $name>](b: Bencher) {
                let input = std::fs::read_to_string($path).unwrap();
                b.bench(|| {
                    black_box({
                        let index = fast_registry_json::pluck_packument_index(&input).unwrap();
                        let range = index
                            .versions
                            .iter()
                            .zip(index.version_ranges)
                            .find_map(|(version, range)| (*version == $version).then_some(range))
                            .unwrap();
                        let info: NpmPackageVersionInfo =
                            serde_json::from_str(&input[range.0 as usize..range.1 as usize])
                                .unwrap();
                        assert_eq!(info.version.to_string(), $version);
                        info
                    })
                });
            }
        }
    };
}

macro_rules! registry_path {
    ($name: literal) => {
        concat!(
            "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/",
            $name,
            "/registry.json"
        )
    };
}

const NEXT_PATH: &str = registry_path!("next");
const PRISMA_PATH: &str = registry_path!("@prisma/client");
const NODE_PTY_PATH: &str = registry_path!("node-pty");
const CLIENT_ONLY_PATH: &str = registry_path!("client-only");
const DRIZZLE_ORM_PATH: &str = registry_path!("drizzle-orm");
const DRIZZLE_KIT_PATH: &str = registry_path!("drizzle-kit");

bench_parse_pluck!(next, NEXT_PATH, "15.0.0-canary.202");
bench_parse_pluck!(prisma, PRISMA_PATH, "5.1.0-dev.64");
bench_parse_pluck!(node_pty, NODE_PTY_PATH, "1.0.0");
bench_parse_pluck!(client_only, CLIENT_ONLY_PATH, "0.0.1");
bench_parse_pluck!(drizzle_orm, DRIZZLE_ORM_PATH, "0.29.0");
bench_parse_pluck!(drizzle_kit, DRIZZLE_KIT_PATH, "0.29.0");
