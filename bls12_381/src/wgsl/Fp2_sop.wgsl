

fn Fp2_sum_of_products(a:array<Fp,2> , b: array<Fp,2>) -> Fp {
    var u: array<u32, 12> = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u,0u, 0u, 0u, 0u, 0u, 0u);
    var t: array<u32, 13> = array<u32, 13>(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],u[8],u[9],u[10],u[11],0u);



    var t0 = mac(t[0], a[0].value[0], b[0].value[0], 0u);
    t[0] = t0[0];
    var t1 = mac(t[1], a[0].value[0], b[0].value[1], t0[1]);
    t[1] = t1[0];
    var t2 = mac(t[2], a[0].value[0], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    var t3 = mac(t[3], a[0].value[0], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    var t4 = mac(t[4], a[0].value[0], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    var t5 = mac(t[5], a[0].value[0], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    var t6 = mac(t[6], a[0].value[0], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    var t7 = mac(t[7], a[0].value[0], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    var t8 = mac(t[8], a[0].value[0], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    var t9 = mac(t[9], a[0].value[0], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    var t10 = mac(t[10], a[0].value[0], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    var t11 = mac(t[11], a[0].value[0], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    var t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  


    t0 = mac(t[0], a[1].value[0], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[0], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[0], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[0], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[0], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[0], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[0], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[0], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[0], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[0], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[0], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[0], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[1], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[1], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[1], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[1], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[1], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[1], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[1], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[1], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[1], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[1], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[1], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[1], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[1], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[1], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[1], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[1], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[1], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[1], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[1], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[1], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[1], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[1], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[1], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[1], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[2], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[2], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[2], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[2], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[2], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[2], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[2], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[2], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[2], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[2], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[2], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[2], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[2], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[2], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[2], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[2], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[2], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[2], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[2], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[2], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[2], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[2], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[2], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[2], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[3], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[3], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[3], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[3], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[3], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[3], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[3], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[3], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[3], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[3], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[3], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[3], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[3], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[3], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[3], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[3], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[3], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[3], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[3], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[3], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[3], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[3], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[3], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[3], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[4], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[4], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[4], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[4], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[4], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[4], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[4], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[4], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[4], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[4], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[4], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[4], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[4], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[4], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[4], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[4], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[4], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[4], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[4], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[4], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[4], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[4], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[4], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[4], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[5], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[5], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[5], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[5], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[5], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[5], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[5], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[5], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[5], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[5], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[5], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[5], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[5], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[5], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[5], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[5], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[5], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[5], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[5], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[5], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[5], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[5], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[5], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[5], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[6], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[6], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[6], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[6], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[6], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[6], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[6], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[6], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[6], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[6], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[6], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[6], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[6], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[6], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[6], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[6], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[6], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[6], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[6], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[6], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[6], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[6], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[6], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[6], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[7], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[7], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[7], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[7], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[7], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[7], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[7], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[7], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[7], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[7], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[7], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[7], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[7], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[7], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[7], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[7], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[7], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[7], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[7], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[7], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[7], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[7], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[7], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[7], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[8], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[8], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[8], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[8], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[8], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[8], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[8], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[8], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[8], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[8], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[8], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[8], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[8], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[8], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[8], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[8], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[8], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[8], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[8], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[8], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[8], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[8], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[8], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[8], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[9], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[9], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[9], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[9], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[9], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[9], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[9], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[9], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[9], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[9], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[9], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[9], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[9], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[9], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[9], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[9], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[9], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[9], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[9], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[9], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[9], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[9], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[9], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[9], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[10], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[10], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[10], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[10], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[10], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[10], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[10], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[10], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[10], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[10], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[10], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[10], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[10], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[10], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[10], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[10], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[10], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[10], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[10], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[10], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[10], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[10], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[10], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[10], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[0].value[11], b[0].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[0].value[11], b[0].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[0].value[11], b[0].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[0].value[11], b[0].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[0].value[11], b[0].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[0].value[11], b[0].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[0].value[11], b[0].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[0].value[11], b[0].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[0].value[11], b[0].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[0].value[11], b[0].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[0].value[11], b[0].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[0].value[11], b[0].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    t0 = mac(t[0], a[1].value[11], b[1].value[0], 0u);
    t[0] = t0[0];
    t1 = mac(t[1], a[1].value[11], b[1].value[1], t0[1]);
    t[1] = t1[0];
    t2 = mac(t[2], a[1].value[11], b[1].value[2], t1[1]);
    t[2] = t2[0];  
    t3 = mac(t[3], a[1].value[11], b[1].value[3], t2[1]);
    t[3] = t3[0];  
    t4 = mac(t[4], a[1].value[11], b[1].value[4], t3[1]);
    t[4] = t4[0];  
    t5 = mac(t[5], a[1].value[11], b[1].value[5], t4[1]);
    t[5] = t5[0];  
    t6 = mac(t[6], a[1].value[11], b[1].value[6], t5[1]);
    t[6] = t6[0];  
    t7 = mac(t[7], a[1].value[11], b[1].value[7], t6[1]);
    t[7] = t7[0];  
    t8 = mac(t[8], a[1].value[11], b[1].value[8], t7[1]);
    t[8] = t8[0];  
    t9 = mac(t[9], a[1].value[11], b[1].value[9], t8[1]);
    t[9] = t9[0];  
    t10 = mac(t[10], a[1].value[11], b[1].value[10], t9[1]);
    t[10] = t10[0];  
    t11 = mac(t[11], a[1].value[11], b[1].value[11], t10[1]);
    t[11] = t11[0];  
    t12 = adc(t[12], 0u, t11[1]);
    t[12] = t12[0];  
                  

    var k = t[0] * INV;
    var f = mac(t[0], k, MODULUS[0], 0u);
    var r1 = mac(t[1], k, MODULUS[1], f[1]);
    var r2 = mac(t[2], k, MODULUS[2], r1[1]);
    var r3 = mac(t[3], k, MODULUS[3], r2[1]);
    var r4 = mac(t[4], k, MODULUS[4], r3[1]);
    var r5 = mac(t[5], k, MODULUS[5], r4[1]);
    var r6 = mac(t[6], k, MODULUS[6], r5[1]);
    var r7 = mac(t[7], k, MODULUS[7], r6[1]);
    var r8 = mac(t[8], k, MODULUS[8], r7[1]);
    var r9 = mac(t[9], k, MODULUS[9], r8[1]);
    var r10 = mac(t[10], k, MODULUS[10], r9[1]);
    var r11 = mac(t[11], k, MODULUS[11], r10[1]);

    var r12 = adc(t[12], 0u, r11[1]);

    
    let final_fp = Fp(array<u32,12>(
        r1[0],
        r2[0],
        r3[0],
        r4[0],
        r5[0],
        r6[0],
        r7[0],
        r8[0],
        r9[0],
        r10[0],
        r11[0],
        r12[0]
    ));

    return subtract_p(final_fp);
}

