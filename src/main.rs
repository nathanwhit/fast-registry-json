fn main() {
    let input = std::fs::read_to_string(
        "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/next/registry.json",
    )
    .unwrap();
    for _ in 0..200 {
        let _versions = std::hint::black_box(fast_registry_json::pluck_versions(&input));
        // println!("versions: {:?}", versions);
    }
}
