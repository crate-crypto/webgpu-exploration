import functools

# check if the element is larger than (p-1) // 2 
lexicographic_value =  functools.reduce(lambda a, b: (a << 64)+b
                     ,[
 0xdcff_7fff_ffff_d556,
 0x0f55_ffff_58a9_ffff,
 0xb398_6950_7b58_7b12,
 0xb23b_a5c2_79c2_895f,
 0x258d_d3db_21a5_d66b,
 0x0d00_88f5_1cbf_f34d
 ][::-1], 0)

print(lexicographic_value)


alen = lexicographic_value.bit_length()

base = 32
for i in range(alen, 0, -base):
    print(hex(int(bin(lexicographic_value)[2:][i-base if (i-base) > 0 else 0: i], base=2))+",")
