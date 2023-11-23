M = 72639
n = 5
b = 10
R = b ** n
T = 7118368
b = 10
# M =  4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787
# R = 2 ** 32


def gcd(a,b):
    if (b == 0):
        return a 
    else:
        return gcd(b, a % b)

def extended_gcd(a,b):
    if (a == 0):
        return b, 0 ,1

    gcd , s1 , t1 = extended_gcd(b%a , a) 
    s , t = (t1 - (b // a) * s1, s1)
    return gcd, s, t 
        


assert(gcd(M,R) == 1)

gc, M_inv , R_inv = (extended_gcd(M, R))

# R_inv calculation for positive number since 0 to 2**32 is the required calculation

print(R_inv)
print(M_inv)

M_inv = -M_inv

while M_inv <= 0:
    M_inv += b 

print(M_inv )

def montgomery_reduce(T):
    A = [] 
    A_num = T
    for i in range(0,n):
        u = int(A[i]) * (M_inv % b)
        A_num = A_num + u*M*(b**i)
        A = str(A_num)[::-1]
        print(u)
        print(A_num)
    

montgomery_reduce(T)








