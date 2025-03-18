fn main() {
    let input = std::fs::read_to_string(
        "/Users/nathanwhit/Library/Caches/deno/npm/registry.npmjs.org/next/registry.json",
    )
    .unwrap();
    // let input = r#"{ "versions": { "aaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {}, "bcdefghijkzzzzzzzzzzzzzzzzzzzzzzzzzzz": {} } }"#;
    // let input = r#"{"versions":{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaa":{},"bcdefghijkabcd":"asdf"}}"#;
    // let input = r#"{"versions":{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaa":{},"bcdefghijkab":"asdf"}}"#;

    // println!("{}", &input[122688 - 32..122688 + 64]);
    for _ in 0..10 {
        let _versions = std::hint::black_box(fast_registry_json::pluck_versions(&input));
        // println!("versions: {:?}", _versions);
    }
}

/*2 Token { start: 6332343, end: 6332352, kind: String } Token { start: 6332353, end: 6332416, kind: String }, Token { start: 6332417, end: 6332452, kind: String } */
