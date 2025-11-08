* Signature: 0x7c40923e75465e42
NAME lp_scratch_small
ROWS
 N  OBJ
 E  eq1     
 L  le1     
 G  ge1     
 E  rng1    
COLUMNS
    x0        OBJ       1
    x0        eq1       2
    x0        le1       1
    x0        rng1      1
    x1        OBJ       -2
    x1        eq1       -1
    x1        le1       2
    x1        ge1       -1
    x1        rng1      1
    x2        OBJ       0.5
    x2        le1       1
    x3        eq1       1
    x4        OBJ       3
    x4        ge1       1
    x4        rng1      1
    Rgrng1    rng1      1
RHS
    RHS1      eq1       3
    RHS1      le1       7
    RHS1      ge1       -1
    RHS1      rng1      3
BOUNDS
 MI BND1      x1      
 UP BND1      x1        10
 FX BND1      x2        2
 FR BND1      x3      
 LO BND1      x4        1
 UP BND1      x4        5
 UP BND1      Rgrng1    2
ENDATA
