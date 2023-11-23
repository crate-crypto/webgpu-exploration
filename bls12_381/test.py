a = 3
b = 2333322


def multiply(a: int, b: int) -> list[int]:
    # Split a and b into lower and upper 16 bits
    a_low = a & 0xFFFF
    a_high = a >> 16
    b_low = b & 0xFFFF
    b_high = b >> 16
    # Perform the multiplication
    product_low = a_low * b_low
    product_mid = (a_low * b_high) + (a_high * b_low)
    product_high = a_high * b_high
    # Carry propagation
    carry = (product_low >> 16) + (product_mid & 0xFFFF)
    carry_high = (product_high >> 16) + (product_mid >> 16) + (carry >> 16)
    # Assemble the result
    result_low = (carry << 16) | (product_low & 0xFFFF)
    result_high = carry_high + product_high
    if (result_high > 0):
        result_high -= 1
    return [result_high, result_low]



# 67063
print(multiply(a, b))
print(a*b)
print((a*b) & 0xffff_ffff)
print((a*b) >> 32)
 
