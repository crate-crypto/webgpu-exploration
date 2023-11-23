use fp::run;

use criterion::{criterion_group, criterion_main, Criterion};

fn fp6_add() {
    let d0 : Vec<u32> = vec![
                0xb1b8_2d58,
            0x47f9_cb98,
                0xa3aa_1d9d,
            0x5fe9_11eb,
                0x4dd8_1db3,
            0x96bf_1b5f,
                0xc925_9f5b,
            0x8100_d27c,
                0x7464_0eab,
            0xafa2_0b96,
                0xd8d9_497d,
            0x09bb_cea7,
    ];

    let d1:Vec<u32> =  vec![
                0xb166_2daa,
            0x0303_cb98,
                0x0a62_1d5a,
            0xd931_10aa,
                0x5be4_a468,
            0xbfa9_820c,
                0xcb05_a348,
            0x0ba3_643e,
                0x1f1c_25a6,
            0xdc35_34bb,
                0x19c0_e1c1,
            0x06c3_05bb,
        ];
    let d2: Vec<u32> = vec![
                0xb162_d858,
            0x46f9_cb98,
                0xf7aa_1d57,
            0x0be9_109c,
                0xfece_41d2,
            0xc791_bc55,
                0x4e38_5ec2,
            0xf84c_5770,
                0xc010_e60f,
            0xcb49_c1d9,
                0x58bf_e3c8,
            0x0acd_b8e1,
        ];
    let d3: Vec<u32> = vec![
                0xb15f_8306,
            0x8aef_cb98,
                0xe4f2_1d54,
            0x3ea1_108f,
                0xa1b7_df3b,
            0xcf79_f69f,
                0xd16b_1a3c,
            0xe4f5_4aa1,
                0x6105_a679,
            0xba5e_4ef8,
                0x97be_e5cf,
            0x0ed8_6c07,
            ];
    let d4: Vec<u32> = vec![
                0xb15c_2db4,
            0xcee5_cb98,
                0xd23a_1d51,
            0x7159_1082,
                0x44a1_7ca4,
            0xd762_30e9,
                0x549d_d5b6,
            0xd19e_3dd3,
                0x01fa_66e3,
            0xa972_dc17,
                0xd6bd_e7d6,
            0x12e3_1f2d,
        ];

    let d5: Vec<u32> = vec![
                0xb173_2d9d,
            0xad2a_cb98,
                0x0696_1d64,
            0x2cfd_10dd,
                0xc6ef_24e8,
            0x0739_6b86,
                0xb1bf_c820,
            0xbd76_e2fd,
                0xde94_d0d5,
            0x6afe_a7f6,
                0x5744_c040,
            0x1099_4b0c,
            ];

    let d6 : Vec<u32> = vec![
                0xb16f_d84b,
            0xf120_cb98,
                0xf3de_1d61,
            0x5fb5_10cf,
                0x69d8_c251,
            0x0f21_a5d0,
                0x34f2_839a,
            0xaa1f_d62f,
                0x7f89_913f,
            0x5a13_3515,
                0x9643_c247,
            0x14a3_fe32,
    ];

    let d7:Vec<u32> =  vec![
                0xb16c_82f9,
            0x3516_cb98,
                0xe126_1d5f,
            0x926d_10c2,
                0x0cc2_5fba,
            0x1709_e01a,
                0xb825_3f14,
            0x96c8_c960,
                0x207e_51a9,
            0x4927_c234,
                0xd542_c44e,
            0x18ae_b158,
        ];
    let d8: Vec<u32> = vec![
                0xb169_82fc,
            0xbf0d_cb98,
                0x1d1a_1d5c,
            0xa679_10b7,
                0xb8fb_06ff,
            0xb7c1_47c2,
                0x47d2_e7ce,
            0x1efa_710d,
                0x7e27_653c,
            0xed20_a79c,
                0xdac1_dfba,
            0x02b8_5294,
        ];
    let d9: Vec<u32> = vec![
                0xb180_82e5,
            0x9d52_cb98,
                0x5176_1d6f,
            0x621d_1111,
                0x3b48_af43,
            0xe798_8260,
            0xa4f4_da37,
            0x0ad3_1637,
                0x5ac1_cf2e,
            0xaeac_737c,
                0x5b48_b824,
            0x006e_7e73,
            ];
    let d10: Vec<u32> = vec![
                0xb17d_2d93,
            0xe148_cb98,
                0x3ebe_1d6c,
            0x94d5_1104,
                0xde32_4cac,
            0xef80_bca9,
                0x2827_95b1,
            0xf77c_0969,
                0xfbb6_8f97,
            0x9dc1_009a,
                0x9a47_ba2b,
            0x0479_3199,

        ];
    let d11: Vec<u32> = vec![
                0xb179_d841,
            0x253e_cb98,
                0x2c06_1d6a,
            0xc78d_10f7,
                0x811b_ea15,
            0xf768_f6f3,
                0xab5a_512b,
            0xe424_fc9a,
                0x9cab_5001,
            0x8cd5_8db9,
                0xd946_bc32,
            0x0883_e4bf,
            ];


    let c = pollster::block_on(run(
        &d0.into_iter()
            .chain(d1.into_iter())
            .chain(d2.into_iter())
            .chain(d3.into_iter())
            .chain(d4.into_iter())
            .chain(d5.into_iter())
            .chain(d6.into_iter())
            .chain(d7.into_iter())
            .chain(d8.into_iter())
            .chain(d9.into_iter())
            .chain(d10.into_iter())
            .chain(d11.into_iter())
            .collect::<Vec<u32>>(),
        "Fp6_add_test",
    ));

    //TODO
    assert_eq!(c,c);

}


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Fp6 add", |b| b.iter(|| fp6_add()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

