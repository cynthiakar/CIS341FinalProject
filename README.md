# CIS341FinalProject
CIS 341 Final Project

## Mac Compile
`gcc -march=haswell -mno-avx -O3 dgemm.c`
OR
`gcc -march=haswell -mno-avx2 -O3 dgemm.c`
OR 
`gcc -mavx -O3 dgemm.c`