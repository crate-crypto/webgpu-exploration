p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab


def is_quadratic_residue(n, p):
    if n % p == 0:
        return True
    return pow(n, (p-1)//2, p) == 1


def tonelli_shanks(n, p):
    if n % p == 0:
        return 0

    if not is_quadratic_residue(n, p):
        print("value is not quadratic residue")
    else:
        print("value is quadratic residue")

        # if p = 3(mod 4)
        if p % 4 == 3:
            return pow(n, (p+1)//4, p)

    # haven't implemented for other cases as for bls12-381 this satisifies the case
    return 0


print(tonelli_shanks(9, 43))
