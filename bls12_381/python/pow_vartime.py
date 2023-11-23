print("""
fn Fp2_pow_vartime(fp2: Fp2 , by: array<u32,12>) -> Fp2 {
    var res = Fp2_one;
""")

for e in range(11, -1 , -1):
    for i in range(63,-1,-1):
        print(f'''
    res = Fp2_square(res);
    if (by[{e}] >> {i}u & 1u) == 1u {{
        res = Fp2_mul(res, fp2);
    }}
              ''')

print("return res;")
print("}")

