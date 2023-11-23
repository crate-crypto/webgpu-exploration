use fp::run;

#[test]
fn fp2_add_test() {
    let d0 : Vec<u32> = vec![
            0x63ee_70d4,
            0xc9a2_1831,
            0x196b_5c91,
            0xbc37_70a7,
            0x304c_5f44,
            0xa247_f8c1,
            0x726c_80b5,
            0xb01f_c2a3,
            0xbbd9_19c9,
            0xe1d2_93e5,
            0x020e_f2ca,
            0x04b7_8e80];

    let d1:Vec<u32> =  vec![
            0x0462_618f,
            0x952e_a446,
            0xf025_c62f,
            0x238d_5edd,
            0x2ea9_2e72,
            0xf6c9_4b01,
            0xc1c9_3808,
            0x03ce_24ea,
            0x45da_483c,
            0x0559_50f9,
            0x0df4_eabc,
            0x010a_768d,
        ];

    let d2: Vec<u32> = vec![
            0xa4d2_c1fe,
            0xa1e0_9175,
            0x204e_ff12,
            0x8b33_acfc,
            0x1b45_6e42,
            0xe244_15a1,
            0xb6ee_1936,
            0x61d9_96b1,
            0x667c_853c,
            0x1164_dbe8,
            0xcc7d_9c79,
            0x0788_557a,
        ];
    let d3: Vec<u32> = vec![
            0x6f48_fa36,
            0xda6a_87cc,
            0x277c_1903,
            0x0fc7_b488,
            0xdc44_8187,
            0x9445_ac4a,
            0xc909_9209,
            0x0261_6d5b,
            0x2db5_8d48,
            0xdbed_4677,
            0x76c7_b7b1,
            0x11b9_4d50,
            ];


    let c = pollster::block_on(run(
        &d0.into_iter()
            .chain(d1.into_iter())
            .chain(d2.into_iter())
            .chain(d3.into_iter())
            .collect::<Vec<u32>>(),
        "Fp2_add_test",
    ));

    let expected_output : Vec<u32> = vec![ 0x08c1_32d2, 0x6b82_a9a7, 0x39ba_5ba4, 0x476b_1da3, 0x4b91_cd87, 0x848c_0e62, 0x295a_99ec, 0x11f9_5955, 0x2255_9f06, 0xf337_6fce, 0xce8c_8f43, 0x0c3f_e3fa, 0x73ab_5bc5, 0x6f99_2c12, 0x17a1_df33, 0x3355_1366, 0x0aed_aff9, 0x8b0e_f74c, 0x8ad2_ca12, 0x062f_9246, 0x738f_d584, 0xe146_9770, 0x84bc_a26d, 0x12c3_c3dd];


 assert_eq!(c[0..24], expected_output);

}

#[test]
fn fp2_sub_test() {
    let d0 : Vec<u32> = vec![
            0x63ee_70d4,
            0xc9a2_1831,
            0x196b_5c91,
            0xbc37_70a7,
            0x304c_5f44,
            0xa247_f8c1,
            0x726c_80b5,
            0xb01f_c2a3,
            0xbbd9_19c9,
            0xe1d2_93e5,
            0x020e_f2ca,
            0x04b7_8e80,
];

    let d1:Vec<u32> =  vec![
            0x0462_618f,
            0x952e_a446,
            0xf025_c62f,
            0x238d_5edd,
            0x2ea9_2e72,
            0xf6c9_4b01,
            0xc1c9_3808,
            0x03ce_24ea,
            0x45da_483c,
            0x0559_50f9,
            0x0df4_eabc,
            0x010a_768d,
        ];

    let d2: Vec<u32> = vec![
           0xa4d2_c1fe,
            0xa1e0_9175,
            0x204e_ff12,
            0x8b33_acfc,
            0x1b45_6e42,
            0xe244_15a1,
            0xb6ee_1936,
            0x61d9_96b1,
            0x667c_853c,
            0x1164_dbe8,
            0xcc7d_9c79,
            0x0788_557a,
        ];
    let d3: Vec<u32> = vec![
            0x6f48_fa36,
            0xda6a_87cc,
            0x277c_1903,
            0x0fc7_b488,
            0xdc44_8187,
            0x9445_ac4a,
            0xc909_9209,
            0x0261_6d5b,
            0x2db5_8d48,
            0xdbed_4677,
            0x76c7_b7b1,
            0x11b9_4d50,
            ];


    let c = pollster::block_on(run(
        &d0.into_iter()
            .chain(d1.into_iter())
            .chain(d2.into_iter())
            .chain(d3.into_iter())
            .collect::<Vec<u32>>(),
        "Fp2_sub_test",
    ));

    let expected_output : Vec<u32> = vec![
            0xbf1b_5981,
            0xe1c0_86bb,
            0xaa70_5d7e,
            0x4faf_c3a9,
            0x0bb7_e726,
            0x2734_b5c1,
            0xaf03_7a3e,
            0xb2bd_7776,
            0x98a8_4164,
            0x1b89_5fb3,
            0x6f11_3cec,
            0x1730_4aef,
            0x9519_1204,
            0x74c3_1c79,
            0x79fd_ad2b,
            0x3271_aa54,
            0x4915_a30f,
            0xc9b4_7157,
            0xec44_b8be,
            0x65e4_0313,
            0x5b70_67cb,
            0x7487_b238,
            0xd0ad_19a4,
            0x0952_3b26,
    ];


 assert_eq!(c[0..24], expected_output);
}


#[test]
fn fp2_test_negation() {
    let d0 : Vec<u32> = vec![
         0x63ee_70d4,
            0xc9a2_1831,
            0x196b_5c91,
            0xbc37_70a7,
            0x304c_5f44,
            0xa247_f8c1,
            0x726c_80b5,
            0xb01f_c2a3,
            0xbbd9_19c9,
            0xe1d2_93e5,
            0x020e_f2ca,
            0x04b7_8e80,
];

    let d1:Vec<u32> =  vec![
            0x0462_618f,
            0x952e_a446,
            0xf025_c62f,
            0x238d_5edd,
            0x2ea9_2e72,
            0xf6c9_4b01,
            0xc1c9_3808,
            0x03ce_24ea,
            0x45da_483c,
            0x0559_50f9,
            0x0df4_eabc,
            0x010a_768d,
        ];


    let c = pollster::block_on(run(
        &d0.into_iter()
            .chain(d1.into_iter())
            .collect::<Vec<u32>>(),
        "Fp6_add_test",
    ));

    let expected_output : Vec<u32> = vec![
            0x9c11_39d7,
            0xf05c_e7ce,
            0x97e8_a36d,
            0x6274_8f57,
            0xc664_96df,
            0xc4e8_d9df,
            0x8118_9209,
            0xb457_88e1,
            0x8772_930d,
            0x6949_13d0,
            0x3770_f3cf,
            0x1549_836a,
            0xfb9d_491c,
            0x24d0_5bb9,
            0xc12e_39d0,
            0xfb1e_a120,
            0xc807_c7b1,
            0x7067_879f,
            0x31bb_dab6,
            0x60a9_269a,
            0xfd71_649b,
            0x45c2_56bc,
            0x2b8a_fbde,
            0x18f6_9b5d,
    ];


 // assert_eq!(c[0..24], expected_output);
}


