
use fp::run;

use criterion::{criterion_group, criterion_main, Criterion};

fn g1projective_generator_is_on_curve() {
    let c = pollster::block_on(run(&vec![1], "G1Projective_test_generator_is_on_curve"));
    // should be converted to 0 
    assert!(c[0] == 0);
}

fn g1projective_identity_is_on_curve() {
    let c = pollster::block_on(run(&vec![0], "G1Projective_test_identity_is_on_curve"));
    // should be converted to 1 
    assert!(c[0] == 1);
}

fn g1projective_test_equality() {
    let c = pollster::block_on(run(&vec![1], "G1Projective_test_equality"));
    // should be converted to 0 
    assert!(c[0] == 0);
}

fn g1projective_test_conditionally_select_affine_should_select_first () {
    let c = pollster::block_on(run(&vec![0], "G1Projective_test_conditionally_select_affine_should_select_first"));
    // should be converted to 1 
    assert!(c[0] == 1);
}


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("G1projective Generator is on curve", |b| b.iter(|| g1projective_generator_is_on_curve()));
    c.bench_function("G1Projective identity is on curve ", |b| b.iter(|| g1projective_identity_is_on_curve()));
    c.bench_function("G1Projective test equality", |b| b.iter(|| g1projective_test_equality()));
    c.bench_function("G1projective test conditionally select affine ", |b| b.iter(|| g1projective_test_conditionally_select_affine_should_select_first()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

