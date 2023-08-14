import functools
p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

a = functools.reduce(lambda a, b: (a << 64)+b,
                     [
    3051072770903393768,
    11629791499313985915,
    8689276483349490602,
    693720623975851170,
    6415838286187876640,
    3795771640209342192,
    15133964633486399704,
    8125486586422361559,
    8992179073402572659,
    16734090500167908874,
    11657655863148526223,
    2124775885383657,
][::-1], 0)


k = functools.reduce(lambda a, b: (a << 32)+b,
                     [1447110796,
                      2016594960,
                      2451220526,
                      822115760,
                      288833397,
                      2209798107,
                      251694668,
                      3760951996,
                      1212064369,
                      66686927,
                      805426398,
                      3000234609,
                      1689231883,
                      2210589724,
                      2561629332,
                      1663594504,
                      2122780711,
                      13412768,
                      3782576117,
                      3452149145,
                      2654222121,
                      ][::-1], 0)
print(a)
print(k)


d = functools.reduce(lambda a, b: (a << 64)+b,
                     [
    0x0397_a383_2017_0cd4,
    0x734c_1b2c_9e76_1d30,
    0x5ed2_55ad_9a48_beb5,
    0x095a_3c6b_22a7_fcfc,
    0x2294_ce75_d4e2_6a27,
    0x1333_8bd8_7001_1ebb,
][::-1], 0)

b = functools.reduce(lambda a, b: (a << 64)+b,
                     [
    0xb9c3_c7c5_b119_6af7,
    0x2580_e208_6ce3_35c1,
    0xf49a_ed3d_8a57_ef42,
    0x41f2_81e4_9846_e878,
    0xe076_2346_c384_52ce,
    0x0652_e893_26e5_7dc0
][::-1], 0)

print(d * b)
c = functools.reduce(lambda a, b: (a << 64)+b,
                     [
    0xf96e_f3d7_11ab_5355,
    0xe8d4_59ea_00f1_48dd,
    0x53f7_354a_5f00_fa78,
    0x9e34_a4f3_125c_5f83,
    0x3fbe_0c47_ca74_c19e,
    0x01b0_6a8b_bd4a_dfe4,][::-1], 0)

# print(c)


# a = functools.reduce(lambda a, b: (a << 64)+b,
#                      [0x3934_42cc_b58b_b327,
#                       0x1092_685f_3bd5_47e3,
#                       0x3382_252c_ab6a_c4c9,
#                       0xf946_94cb_7688_7f55,
#                       0x4b21_5e90_93a5_e071,
#                       0x0d56_e30f_34f5_f853,
#                       ][::-1], 0)

# a = (p+1)//4

# a = p ** (2)

alen = a.bit_length()
modulus = []
# print(a)
# print(hex(a))

base = 32
for i in range(alen, 0, -base):
    # print(hex(int(bin(a)[2:][i-base if (i-base) > 0 else 0: i], base=2))+",")
    print(str(int(bin(a)[2:][i-base if (i-base) > 0 else 0: i], base=2))+",")


print(p % 4)
