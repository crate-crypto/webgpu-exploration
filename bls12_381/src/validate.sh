#! /bin/bash

files=('Fp.wgsl' 'Fp2.wgsl' 'Fp2_sop.wgsl' 'Fp2_pow_vartime.wgsl')
main="all.wgsl"


rm $main
touch $main

for f in "${files[@]}"
do
  cat $f >> $main
done

naga $main


