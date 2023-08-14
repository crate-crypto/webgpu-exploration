use fp::run;

#[test]
fn multiply_test() {
    let c = pollster::block_on(run(&vec![9, 4], "multiply_test"));
    // 9 * 4
    assert!(c[0] == 36);
    // no carry
    assert!(c[1] == 0);
}
#[test]
fn sum_test() {
    let d1: Vec<u32> = vec![0x5360bb59];

    let d2: Vec<u32> = vec![0x9fd28773];

    let c = pollster::block_on(run(&vec![d1[0], d2[0]], "sum_test"));

    assert!(c[0] == 4080222924);
    assert!(c[1] == 0);
}

#[test]
fn fp_subtract_test() {
    let d1: Vec<u32> = vec![
        0x78678032, 0x5360bb59, 0x799e128e, 0x7dd275ae, 0xce4f4dcf, 0x5c5b5071, 0x78dbb3e,
        0xcdb21f93, 0xe73f474a, 0xc32365c5, 0x89babe5b, 0x115a2a54,
    ];

    let d2: Vec<u32> = vec![
        0x3d23dda0, 0x9fd28773, 0x738b3554, 0xb16bf2af, 0xd3cc6d1d, 0x3e57a75b, 0x627fd6d6,
        0x900bc0bd, 0xefb245fe, 0xd319a080, 0xe4bb2091, 0x15fdcaa4,
    ];

    let expected_output: Vec<u32> = vec![
        0x3b434d3d, 0x6d8d33e6, 0xb766dd39, 0xeb1282fd, 0xf133d6d5, 0x85347bb6, 0x9892f727,
        0xa21daa5a, 0x3ad8ae23, 0x3b256cfb, 0xde7f8464, 0x155d7199,
    ];

    let c = pollster::block_on(run(
        &d1.into_iter().chain(d2.into_iter()).collect::<Vec<u32>>(),
        "fp_subtract_test",
    ));

    assert_eq!(c[0..=11], expected_output);
}

#[test]
fn fp_negative_test() {
    let d1: Vec<u32> = vec![
        0x78678032, 0x5360bb59, 0x799e128e, 0x7dd275ae, 0xce4f4dcf, 0x5c5b5071, 0x78dbb3e,
        0xcdb21f93, 0xe73f474a, 0xc32365c5, 0x89babe5b, 0x115a2a54,
    ];

    let expected_output: Vec<u32> = vec![
        0x87982a79, 0x669e44a6, 0x37b5ed71, 0xa0d98a50, 0x2861a854, 0xad5822f, 0xebf75781,
        0x96c52bf1, 0x5c0c658c, 0x87f841f0, 0xafc5283e, 0x8a6e795,
    ];

    let c = pollster::block_on(run(&d1, "fp_neg_test"));

    assert_eq!(c[0..=11], expected_output);
}

#[test]
fn adc_test() {
    let d1: Vec<u32> = vec![
        0x78678032, 0x5360bb59, 0x799e128e, 0x7dd275ae, 0xce4f4dcf, 0x5c5b5071, 0x78dbb3e,
        0xcdb21f93, 0xe73f474a, 0xc32365c5, 0x89babe5b, 0x115a2a54,
    ];

    let d2: Vec<u32> = vec![
        0x3d23dda0, 0x9fd28773, 0x738b3554, 0xb16bf2af, 0xd3cc6d1d, 0x3e57a75b, 0x627fd6d6,
        0x900bc0bd, 0xefb245fe, 0xd319a080, 0xe4bb2091, 0x15fdcaa4,
    ];
    // a + b + carry

    for (a, b) in d1.iter().zip(d2.iter()) {
        let c = pollster::block_on(run(&vec![*a, *b, 0], "adc_test"));
        println!("{}, {}", c[0], c[1]);
        // assert!(c[0] == 3045809618);
        // assert!(c[1] == 0);
    }
}

#[test]
fn mac_test() {
    // a + (b*c) + carry
    let c = pollster::block_on(run(&vec![0, 0x20170cd4, 0xb1196af7, 0], "mac_test"));
    // 1 + (2 * 4) + 5;
    assert_eq!(c[0], 1447110796);
    // no carry
    assert_eq!(c[1], 372449159);
}

// https://gist.github.com/rust-play/3ae2c2e7f7d1483b9fe6b0c1b0410684
#[test]
fn fp_add_test() {
    let a_and_b: Vec<u32> = vec![
        // a portion
        0x78678032, 0x5360bb59, 0x799e128e, 0x7dd275ae, 0xce4f4dcf, 0x5c5b5071, 0x78dbb3e,
        0xcdb21f93, 0xe73f474a, 0xc32365c5, 0x89babe5b, 0x115a2a54, // b portion
        0x3d23dda0, 0x9fd28773, 0x738b3554, 0xb16bf2af, 0xd3cc6d1d, 0x3e57a75b, 0x627fd6d6,
        0x900bc0bd, 0xefb245fe, 0xd319a080, 0xe4bb2091, 0x15fdcaa4,
    ];

    let add_value = pollster::block_on(run(&a_and_b, "add_test"));

    let c: Vec<u32> = vec![
        0xb58bb327, 0x393442cc, 0x3bd547e3, 0x1092685f, 0xab6ac4c9, 0x3382252c, 0x76887f55,
        0xf94694cb, 0x93a5e071, 0x4b215e90, 0x34f5f853, 0xd56e30f,
    ];

    assert_eq!(add_value[0..=11], c);
}

#[test]
fn subtract_p_test() {
    let a: Vec<u32> = vec![
        // a portion
        0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf,
        0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea,
    ];

    let add_value = pollster::block_on(run(&a, "subtract_p_test"));

    let c: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    assert_eq!(add_value, c);
}

//https://gist.github.com/rust-play/731b1ddf924b7e00d65e0a20951a25ff
#[test]
fn another_subtract_p_test() {
    let a: Vec<u32> = vec![
        3045809618, 4080222924, 3978905570, 792619101, 2719726316, 2595420108, 1779274260,
        1572724816, 3606154568, 2520581701, 1853218540, 660075768,
    ];

    let add_value = pollster::block_on(run(&a, "subtract_p_test"));

    let c: Vec<u32> = vec![
        0xb58bb327, 0x393442cc, 0x3bd547e3, 0x1092685f, 0xab6ac4c8, 0x3382252b, 0x76887f55,
        0xf94694cb, 0x93a5e070, 0x4b215e8f, 0x34f5f852, 0xd56e30e,
    ];

    assert_eq!(add_value, c);
}

#[test]
fn sbb_test() {
    let a: Vec<u32> = vec![
        3045809618, 4080222924, 3978905570, 792619101, 2719726316, 2595420108, 1779274260,
        1572724816, 3606154568, 2520581701, 1853218540, 660075768,
    ];

    let md: Vec<u32> = vec![
        // a portion
        0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf,
        0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea,
    ];
    // a - (b+borrow)
    let mut borrow = 0;
    for i in 0..12 {
        let c = pollster::block_on(run(&vec![a[i], md[i], borrow], "sbb_test"));
        borrow = c[1];
        println!("{} {}", c[0], c[1]);
    }
    // 1 - (2 + 1);
    // wrapping subtraction
    // assert_eq!(c[0], 994288274);
    // assert!(c[1] == 0);
}

#[test]
fn fp_multiply_test() {
    let d1: Vec<u32> = vec![
        0x20170cd4, 0x397a383, 0x9e761d30, 0x734c1b2c, 0x9a48beb5, 0x5ed255ad, 0x22a7fcfc,
        0x95a3c6b, 0xd4e26a27, 0x2294ce75, 0x70011ebb, 0x13338bd8,
    ];

    let d2: Vec<u32> = vec![
        0xb1196af7, 0xb9c3c7c5, 0x6ce335c1, 0x2580e208, 0x8a57ef42, 0xf49aed3d, 0x9846e878,
        0x41f281e4, 0xc38452ce, 0xe0762346, 0x26e57dc0, 0x652e893,
    ];

    let expected_output: Vec<u32> = vec![
        0x11ab5355, 0xf96ef3d7, 0xf148dd, 0xe8d459ea, 0x5f00fa78, 0x53f7354a, 0x125c5f83,
        0x9e34a4f3, 0xca74c19e, 0x3fbe0c47, 0xbd4adfe4, 0x1b06a8b,
    ];

    let c = pollster::block_on(run(
        &d1.into_iter().chain(d2.into_iter()).collect::<Vec<u32>>(),
        "fp_multiply_test",
    ));

    assert_eq!(c[0..=11], expected_output);
}
