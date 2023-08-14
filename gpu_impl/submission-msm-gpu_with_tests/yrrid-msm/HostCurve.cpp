/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#include <stdio.h>
#include <stdint.h>

/* Quick and dirty low performance host side XYZZ implementation */

namespace Host {

namespace BLS12377 {

uint32_t NP0=0xFFFFFFFFu;

uint32_t N[12]={
  0x00000001, 0x8508C000, 0x30000000, 0x170B5D44, 0xBA094800, 0x1EF3622F,
  0x00F5138F, 0x1A22D9F3, 0x6CA1493B, 0xC63B05C0, 0x17C510EA, 0x01AE3A46,
};

uint32_t R[12]={
  0xFFFFFF68, 0x02CDFFFF, 0x7FFFFFB1, 0x51409F83, 0x8A7D3FF2, 0x9F7DB3A9,     // 2^384 mod N
  0x6E7C6305, 0x7B4E97B7, 0x803C84E8, 0x4CF495BF, 0xE2FDF49A, 0x008D6661,
};

uint32_t RCubed[12]={
  0x8815DE20, 0x581F532F, 0xBE329585, 0xE50F4148, 0x0449F513, 0x2BE8B118,     // 2^(384*3) mod N
  0xC804A20E, 0x6A2A9516, 0x13590CB9, 0x3F725407, 0xC0E7DDA5, 0x01065AB4,
};

class G1Montgomery {
  public:
  typedef uint32_t Value[12];

  static void setZero(Value& r) {
    for(int i=0;i<12;i++)
      r[i]=0;
  }

  static void setOne(Value& r) {
    for(int i=0;i<12;i++) 
      r[i]=(i==0) ? 1 : 0;
  }

  static void setR(Value& r) {
    for(int i=0;i<12;i++)
      r[i]=R[i];
  }

  static void set(Value& r, const Value& field) {
    for(int i=0;i<12;i++)
      r[i]=field[i];
  }

  static void load(Value& field, uint32_t* ptr) {
    for(int i=0;i<12;i++)
      field[i]=ptr[i];
  }

  static void store(uint32_t* ptr, Value& field) {
    for(int i=0;i<12;i++)
      ptr[i]=field[i];
  }

  static void exportField(uint64_t* ptr, const Value& field) { // +
    for(int i=0;i<6;i++)
      ptr[i]=(((uint64_t)field[i*2+1])<<32) | ((uint64_t)field[i*2]);
  }

  static bool isZero(const Value& field) {
    for(int i=0;i<12;i++)
      if(field[i]!=0)
        return false;
    return true;
  }

  static bool isGE(const Value& a, const Value& b) { // +
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)a[i]) - ((uint64_t)b[i]);
      acc=acc>>32;
    }
    return acc>=0;
  }

  static void addN(Value& r, const Value& field) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)field[i]) + ((uint64_t)N[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
  }

  static bool subN(Value& r, const Value& field) { // +
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)field[i]) - ((uint64_t)N[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
    return acc>=0;
  }

  static void add(Value& r, const Value& a, const Value& b) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)a[i]) + ((uint64_t)b[i]) - ((uint64_t)N[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
    if(acc>=0) 
      return;
    addN(r, r);
  }

  static void sub(Value& r, const Value& a, const Value& b) {
    int64_t acc=0;

    for(int i=0;i<12;i++) {
      acc+=((uint64_t)a[i]) - ((uint64_t)b[i]);
      r[i]=(uint32_t)acc;
      acc=acc>>32;
    }
    if(acc>=0)
      return;
    addN(r, r);
  }

  static void mul(Value& r, const Value& a, const Value& b) { // +
    uint64_t acc, high=0, q;
    uint32_t res[12];

    for(int i=0;i<12;i++)
      res[i]=0;
    for(int j=0;j<12;j++) {
      acc=0;
      for(int i=0;i<12;i++) {
        acc+=((uint64_t)a[j])*((uint64_t)b[i]) + ((uint64_t)res[i]);
        res[i]=(uint32_t)acc;
        acc=acc>>32;
      }
      high+=acc;
      q=(uint64_t)(res[0]*NP0);
      acc=q*((uint64_t)N[0]) + ((uint64_t)res[0]);
      acc=acc>>32;
      for(int i=1;i<12;i++) {
        acc+=q*((uint64_t)N[i]) + ((uint64_t)res[i]);
        res[i-1]=(uint32_t)acc;
        acc=acc>>32;
      }
      high+=acc;
      res[11]=(uint32_t)high;
      high=high>>32;
    }
    if(high!=0 || isGE(res, N))
      subN(r, res);
    else
      set(r, res);
  }

  static void shiftRight(Value& r, const Value& field, uint32_t bits) {
    uint32_t words=bits>>5;
    uint32_t left;

    if(words>0) {
      for(int i=0;i<12;i++) {
        if(i+words<12) 
          r[i]=field[i+words];
        else
          r[i]=0;
      }
      bits=bits-words*32;
    }
    else {
      for(int i=0;i<12;i++)
        r[i]=field[i];
    }

    if(bits==0)
      return;
    left=32-bits;
    for(int i=0;i<11;i++) 
      r[i]=(r[i]>>bits) | (r[i+1]<<left);
    r[11]=r[11]>>bits;
  }

  static void swap(Value& a, Value& b) {
    uint32_t swap;

    for(int i=0;i<12;i++) {
      swap=a[i];
      a[i]=b[i];
      b[i]=swap;
    }
  }

  static void reduce(Value& r, const Value& field) {
    set(r, field);
    while(isGE(r, N)) 
      subN(r, r);
  }

  static void print(const Value& field) {
    for(int i=11;i>=0;i--) 
      printf("%08X", field[i]);
    printf("\n");
  }

  static void inverse(Value& r, const Value& field) {
    Value A, B, X, Y;

    // slow, but very easy to code
    set(A, field);
    set(B, N);
    setOne(X);
    setZero(Y); 
    while(!isZero(A)) {
      if((A[0] & 0x01)!=0) {
        if(!isGE(A, B)) {
          swap(A, B);
          swap(X, Y);
        }
        sub(A, A, B);
        sub(X, X, Y);
      }
      shiftRight(A, A, 1);
      if((X[0] & 0x01)!=0)
        addN(X, X);
      shiftRight(X, X, 1);
    }

    mul(r, Y, RCubed); 
  }

  static void dump(const Value& field) { // +
    Value local;

    setOne(local);
    mul(local, local, field);
    for(int i=11;i>=0;i--)
      printf("%08X", local[i]);
    printf("\n");
  }
};

} /* namespace BLS12377 */

template<class Field>
class PointXYZZ {
  typedef typename Field::Value FieldValue;
  
  public:
  FieldValue x;
  FieldValue y;
  FieldValue zz;
  FieldValue zzz;
  
  PointXYZZ() {
  }

  void reduce() {
    Field::reduce(x, x);
    Field::reduce(y, y);
    Field::reduce(zz, zz);
    Field::reduce(zzz, zzz);
  }

  PointXYZZ(const FieldValue& xValue, const FieldValue& yValue, const FieldValue& zzValue, const FieldValue& zzzValue) {
    Field::set(x, xValue);
    Field::set(y, yValue);
    Field::set(zz, zzValue);
    Field::set(zzz, zzzValue);
  }
  
  void load(uint32_t* ptr) {
    Field::load(x, ptr);
    Field::load(y, ptr + 12);
    Field::load(zz, ptr + 24);
    Field::load(zzz, ptr + 36);
    reduce();
  }

  void store(uint32_t* ptr) {
    Field::store(ptr, x);
    Field::store(ptr + 12, y);
    Field::store(ptr + 24, zz);
    Field::store(ptr + 36, zzz);
  }

  void normalize() { // +
    FieldValue I;

    if(Field::isZero(zz)) {
      Field::setZero(x);
      Field::setZero(y);
      Field::setZero(zzz);
      return;
    }
    Field::inverse(I, zzz);
    Field::mul(y, y, I);
    Field::mul(I, I, zz);
    Field::mul(I, I, I);
    Field::mul(x, x, I);
    Field::setR(zz);
    Field::setR(zzz);
  }  

  void dump() {
    printf("   x = ");
    Field::dump(x);
    printf("   y = ");
    Field::dump(y);
    printf("  zz = ");
    Field::dump(zz);
    printf(" zzz = ");
    Field::dump(zzz);
  }
};

template<class Field>
class AccumulatorXYZZ {
  typedef typename Field::Value FieldValue;

  public:
  PointXYZZ<Field> xyzz;

  AccumulatorXYZZ() {
    Field::setZero(xyzz.zz);
  }
  
  void set(const FieldValue& x, const FieldValue& y, const FieldValue& zz, const FieldValue& zzz) {
    Field::set(xyzz.x, x);
    Field::set(xyzz.y, y);
    Field::set(xyzz.zz, zz);
    Field::set(xyzz.zzz, zzz);
  }

  void setZero() {
    Field::setZero(xyzz.zz);
  }

  void dbl(const PointXYZZ<Field>& point) {
    FieldValue U, V, W, S, M, T, X, Y, ZZ, ZZZ;

    if(Field::isZero(point.zz)) {
      Field::setZero(xyzz.zz);
      return;
    }

    Field::add(U, point.y, point.y);
    Field::mul(V, U, U);
    Field::mul(W, U, V);
    Field::mul(ZZ, V, point.zz);
    Field::mul(ZZZ, W, point.zzz);
    Field::mul(S, point.x, V);
    Field::mul(M, point.x, point.x);
    Field::add(T, M, M);
    Field::add(M, T, M);
    Field::mul(X, M, M);
    Field::sub(X, X, S);
    Field::sub(X, X, S);
    Field::sub(T, S, X);
    Field::mul(Y, M, T);
    Field::mul(T, W, point.y);
    Field::sub(Y, Y, T);
    set(X, Y, ZZ, ZZZ);
  }

  void add(const PointXYZZ<Field>& point) { // +
    FieldValue U1, U2, S1, S2, P, R, PP, PPP, Q, T, X, Y, ZZ, ZZZ;

    if(Field::isZero(point.zz))
      return;

    if(Field::isZero(xyzz.zz)) {
      set(point.x, point.y, point.zz, point.zzz);
      return;
    }

    Field::mul(U1, xyzz.x, point.zz);
    Field::mul(U2, point.x, xyzz.zz);
    Field::mul(S1, xyzz.y, point.zzz);
    Field::mul(S2, point.y, xyzz.zzz);
    Field::sub(P, U2, U1);
    Field::sub(R, S2, S1);
    if(Field::isZero(P) && Field::isZero(R)) {
      dbl(point);
      return;
    }
    Field::mul(PP, P, P);
    Field::mul(PPP, PP, P);
    Field::mul(Q, U1, PP);
    Field::mul(X, R, R);
    Field::sub(X, X, PPP);
    Field::sub(X, X, Q);
    Field::sub(X, X, Q);
    Field::sub(T, Q, X);
    Field::mul(Y, R, T);
    Field::mul(T, S1, PPP);
    Field::sub(Y, Y, T);
    Field::mul(ZZ, xyzz.zz, point.zz);
    Field::mul(ZZ, ZZ, PP);
    Field::mul(ZZZ, xyzz.zzz, point.zzz);
    Field::mul(ZZZ, ZZZ, PPP);
    set(X, Y, ZZ, ZZZ);
  }
};

} /* namespace Host */

void setZero_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  FILE *file = fopen("tests/setZero_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.setZero(r);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void setOne_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  FILE *file = fopen("tests/setOne_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.setOne(r);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void setR_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  FILE *file = fopen("tests/setR_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.setR(r);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void set_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  uint32_t field[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/set_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.set(r, field);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void load_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  uint32_t field[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/load_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.load(r, field);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void store_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  uint32_t field[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/store_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.store(r, field);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void exportField_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint64_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  uint32_t field[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/exportField_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.exportField(r, field);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void isZero_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};

  FILE *file = fopen("tests/isZero_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  bool k = g.isZero(r);

  length+=sprintf(json_data+length, "\"output\":"
  "%u}", k);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void isGE_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/isGE_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  bool k = g.isGE(r, b);

  length+=sprintf(json_data+length, "\"output\":"
  "%u}", k);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void addN_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/addN_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.addN(r, b);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void subN_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};

  FILE *file = fopen("tests/subN_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.subN(r, b);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void add_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t c[12] = {120,110,100,90,80,70,60,50,40,30,20,10};

  FILE *file = fopen("tests/add_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.add(r, b, c);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void sub_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t c[12] = {120,110,100,90,80,70,60,50,40,30,20,10};

  FILE *file = fopen("tests/sub_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.sub(r, b, c);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void mul_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t c[12] = {120,110,100,90,80,70,60,50,40,30,20,10};

  FILE *file = fopen("tests/mul_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.mul(r, b, c);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void shiftRight_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  FILE *file = fopen("tests/shiftRight_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.shiftRight(r, b, 1);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void swap_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  FILE *file = fopen("tests/swap_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.swap(r, b);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void reduce_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  FILE *file = fopen("tests/reduce_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.reduce(r, b);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void inverse_test() {
  printf("\n");
  Host::BLS12377::G1Montgomery g;

  uint32_t r[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t b[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  FILE *file = fopen("tests/inverse_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  g.inverse(r, b);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", r[0], r[1], r[2], r[3], r[4], r[5], 
          r[6], r[7], r[8], r[9], r[10], r[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void reduce_PointXYZZ_test() {
  printf("\n");

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::PointXYZZ<Field> p(x,y,zz,zzz);

  FILE *file = fopen("tests/reduce_PointXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]},", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  p.reduce();

  length+=sprintf(json_data+length, "\"output\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void load_PointXYZZ_test() {
  printf("\n");

  uint32_t ptr[48];

  for(int i = 0; i<48; i++) {
    ptr[i] = i;
  }

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::PointXYZZ<Field> p(x,y,zz,zzz);

  FILE *file = fopen("tests/load_PointXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]},", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  p.load(ptr);

  length+=sprintf(json_data+length, "\"output\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void store_PointXYZZ_test() {
  printf("\n");

  uint32_t ptr[48];

  for(int i = 0; i<48; i++) {
    ptr[i] = i;
  }

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::PointXYZZ<Field> p(x,y,zz,zzz);

  FILE *file = fopen("tests/store_PointXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]},", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  p.store(ptr);

  length+=sprintf(json_data+length, "\"output\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}", ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], 
          ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11],
          ptr[12], ptr[13], ptr[14], ptr[15], ptr[16], ptr[17], 
          ptr[18], ptr[19], ptr[20], ptr[21], ptr[22], ptr[23],
          ptr[24], ptr[25], ptr[26], ptr[27], ptr[28], ptr[29], 
          ptr[30], ptr[31], ptr[32], ptr[33], ptr[34], ptr[35],
          ptr[36], ptr[37], ptr[38], ptr[39], ptr[40], ptr[41], 
          ptr[42], ptr[43], ptr[44], ptr[45], ptr[46], ptr[47]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void normalize_PointXYZZ_test() {
  printf("\n");

  uint32_t ptr[48];

  for(int i = 0; i<48; i++) {
    ptr[i] = i;
  }

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::PointXYZZ<Field> p(x,y,zz,zzz);

  FILE *file = fopen("tests/normalize_PointXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]},", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  p.normalize();

  length+=sprintf(json_data+length, "\"output\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}", p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], 
          p.x[6], p.x[7], p.x[8], p.x[9], p.x[10], p.x[11],
          p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], 
          p.y[6], p.y[7], p.y[8], p.y[9], p.y[10], p.y[11],
          p.zz[0], p.zz[1], p.zz[2], p.zz[3], p.zz[4], p.zz[5], 
          p.zz[6], p.zz[7], p.zz[8], p.zz[9], p.zz[10], p.zz[11],
          p.zzz[0], p.zzz[1], p.zzz[2], p.zzz[3], p.zzz[4], p.zzz[5], 
          p.zzz[6], p.zzz[7], p.zzz[8], p.zzz[9], p.zzz[10], p.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void set_AccumulatorXYZZ_test() {
  printf("\n");

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::AccumulatorXYZZ<Field> acc;

  FILE *file = fopen("tests/set_AccumulatorXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}},", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  acc.set(x, y, zz, zzz);

  length+=sprintf(json_data+length, "\"output\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}}", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void setZero_AccumulatorXYZZ_test() {
  printf("\n");

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::AccumulatorXYZZ<Field> acc;

  FILE *file = fopen("tests/setZero_AccumulatorXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}},", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  acc.setZero();

  length+=sprintf(json_data+length, "\"output\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}}", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void dbl_AccumulatorXYZZ_test() {
  printf("\n");

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::PointXYZZ<Field> p(x,y,zz,zzz);

  Host::AccumulatorXYZZ<Field> acc;

  FILE *file = fopen("tests/dbl_AccumulatorXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}},", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  acc.dbl(p);

  length+=sprintf(json_data+length, "\"output\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}}", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}

void add_AccumulatorXYZZ_test() {
  printf("\n");

  uint32_t x[12] = {1,2,3,4,5,6,7,87,8,9,11,12};
  uint32_t y[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  uint32_t zzz[12] = {12,11,10,9,8,7,6,5,4,3,2,1};
  
  typedef Host::BLS12377::G1Montgomery Field;

  Host::PointXYZZ<Field> p(x,y,zz,zzz);

  Host::AccumulatorXYZZ<Field> acc;

  FILE *file = fopen("tests/add_AccumulatorXYZZ_test.json", "w");
  if (file == NULL) {
    printf("Failed to open file\n");
    return;
  }

  char json_data[1024]= ""; 
  int length = 0;

  length+=sprintf(json_data+length, "{\"input\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}},", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  acc.add(p);

  length+=sprintf(json_data+length, "\"output\":{\"xyzz\":{\"x\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"y\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u],"
  "\"zzz\":["
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u,"
  "%u]}}}", acc.xyzz.x[0], acc.xyzz.x[1], acc.xyzz.x[2], acc.xyzz.x[3], acc.xyzz.x[4], acc.xyzz.x[5], 
          acc.xyzz.x[6], acc.xyzz.x[7], acc.xyzz.x[8], acc.xyzz.x[9], acc.xyzz.x[10], acc.xyzz.x[11],
          acc.xyzz.y[0], acc.xyzz.y[1], acc.xyzz.y[2], acc.xyzz.y[3], acc.xyzz.y[4], acc.xyzz.y[5], 
          acc.xyzz.y[6], acc.xyzz.y[7], acc.xyzz.y[8], acc.xyzz.y[9], acc.xyzz.y[10], acc.xyzz.y[11],
          acc.xyzz.zz[0], acc.xyzz.zz[1], acc.xyzz.zz[2], acc.xyzz.zz[3], acc.xyzz.zz[4], acc.xyzz.zz[5], 
          acc.xyzz.zz[6], acc.xyzz.zz[7], acc.xyzz.zz[8], acc.xyzz.zz[9], acc.xyzz.zz[10], acc.xyzz.zz[11],
          acc.xyzz.zzz[0], acc.xyzz.zzz[1], acc.xyzz.zzz[2], acc.xyzz.zzz[3], acc.xyzz.zzz[4], acc.xyzz.zzz[5], 
          acc.xyzz.zzz[6], acc.xyzz.zzz[7], acc.xyzz.zzz[8], acc.xyzz.zzz[9], acc.xyzz.zzz[10], acc.xyzz.zzz[11]);

  printf(json_data);

  fprintf(file, "%s\n", json_data);

  fclose(file);
}