

for counter in range(12):
     print(f"""
        let fp{counter} = Fp(array<u32,12>(v_indices[{0+counter*12}], v_indices[{1 + counter*12}], v_indices[{2 + counter*12}], v_indices[{3 + counter*12}], v_indices[{4 + counter*12}], v_indices[{5 + counter*12}], v_indices[{6 + counter*12}], v_indices[{7 + counter*12}], v_indices[{8 + counter*12}], v_indices[{9 + counter*12}], v_indices[{10 + counter*12}], v_indices[{11 + counter*12}]));

    """)
