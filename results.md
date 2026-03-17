## CD and HD

1000 epochs, twd/regularization after 200, pruning after 700, densify at beginning (top 30%) and at 200

### Armadillo (128^3)

#### Large 

chamfer: 0.0018 | hausdorff: 0.0131

#### Small 

chamfer: 0.0018 | hausdorff: 0.0139 (40%)
chamfer: 0.0021 | hausdorff: 0.0142 (60%)
chamfer: 0.0081 | hausdorff: 0.8128 (80%)

#### Densified 

chamfer: 0.0015 | hausdorff: 0.0103

#### AIRe

chamfer: 0.0021 | hausdorff: 0.0158 (40%)
chamfer: 0.0024 | hausdorff: 0.0156 (60% of deep layers)
chamfer: 0.0023 | hausdorff: 0.0139 (80%)


#### DepGraph

chamfer: 0.0019 | hausdorff: 0.0216 (40%)
chamfer: 0.0020 | hausdorff: 0.0125 (60% of deep layers)
chamfer: 0.0024 | hausdorff: 0.8467 (80%)

#### AIRe Densified

chamfer: 0.0020 | hausdorff: 0.6410 (40%)
chamfer: 0.0018 | hausdorff: 0.0111 (60% of deep layers)
chamfer: 0.0021 | hausdorff: 0.0236 (80%)

#### DepGraph Densified

chamfer: 0.0016 | hausdorff: 0.0098 (40%)
chamfer: 0.0018 | hausdorff: 0.9952 (60% of deep layers)
chamfer: 0.0023 | hausdorff: 0.9533 (80%)

### Lucy (128^3)

#### Large 

chamfer: 0.0020 | hausdorff: 0.0436

#### Densified 

chamfer: 0.0018 | hausdorff: 0.0239

#### AIRe

chamfer: 0.0023 | hausdorff: 0.0319 (40%)
chamfer: 0.0029 | hausdorff: 0.8875 (60% of deep layers)
chamfer: 0.0090 | hausdorff: 1.0092 (80%)

#### DepGraph

chamfer: 0.0023 | hausdorff: 0.0375 (40%)
chamfer: 0.0024 | hausdorff: 0.0729 (60% of deep layers)
chamfer: 0.0065 | hausdorff: 0.8489 (80%)

#### AIRe Densified

chamfer: 0.0020 | hausdorff: 0.0187 (40%)
chamfer: 0.0021 | hausdorff: 0.0284 (60% of deep layers)
chamfer: 0.0033 | hausdorff: 0.9737 (80%)

#### DepGraph Densified

chamfer: 0.0019 | hausdorff: 0.0482 (40%)
chamfer: 0.0019 | hausdorff: 0.1097 (60% of deep layers)
chamfer: 0.0028 | hausdorff: 1.1381 (80%)

### Dragon (128^3)

#### Large 

chamfer: 0.0024 | hausdorff: 0.0452

#### Densified 

chamfer: 0.0021 | hausdorff: 0.0494

#### AIRe

chamfer: 0.0024 | hausdorff: 0.0349 (40%)
chamfer: 0.0026 | hausdorff: 0.0386 (60% of deep layers)
chamfer: 0.0070 | hausdorff: 0.9718 (80%)

#### DepGraph

chamfer: 0.0025 | hausdorff: 0.0325 (40%)
chamfer: 0.0026 | hausdorff: 0.6328 (60% of deep layers)
chamfer: 0.0041 | hausdorff: 1.0778 (80%)

#### AIRe Densified

chamfer: 0.0025 | hausdorff: 1.0349 (40%)
chamfer: 0.0023 | hausdorff: 0.0390 (60% of deep layers)
chamfer: 0.0025 | hausdorff: 1.0349 (80%)

#### DepGraph Densified

chamfer: 0.0022 | hausdorff: 0.0270 (40%)
chamfer: 0.0022 | hausdorff: 0.0392 (60% of deep layers)
chamfer: 0.0059 | hausdorff: 0.9291 (80%)

### Bunny (128^3)

#### Large 

chamfer: 0.0014 | hausdorff: 0.0245

#### Densified 

chamfer: 0.0012 | hausdorff: 0.0282

#### AIRe

chamfer: 0.0017 | hausdorff: 0.0274 (40%)
chamfer: 0.0017 | hausdorff: 0.0277 (60% of deep layers)
chamfer: 0.0020 | hausdorff: 0.0240 (80%)

#### DepGraph

chamfer: 0.0015 | hausdorff: 0.0253 (40%)
chamfer: 0.0015 | hausdorff: 0.0247 (60% of deep layers)
chamfer: 0.0018 | hausdorff: 0.0290 (80%)

#### AIRe Densified

chamfer: 0.0015 | hausdorff: 0.0258 (40%)
chamfer: 0.0016 | hausdorff: 0.0266 (60% of deep layers)
chamfer: 0.0017 | hausdorff: 0.0271 (80%)

#### DepGraph Densified

chamfer: 0.0013 | hausdorff: 0.0272 (40%)
chamfer: 0.0014 | hausdorff: 0.0277 (60% of deep layers)
chamfer: 0.0016 | hausdorff: 0.2948 (80%)

IoU's for armadillo:
    baseline 0.9791
    densified 0.9820
Pruning ratio 0.4
    AIRe:
        normal 0.9749
        densified 0.9769
    DepGraph:
        normal 0.9776
        densified 0.9808
Pruning ratio 0.6
    AIRe:
        normal 0.9714
        densified 0.9783
    DepGraph:
        normal 0.9762
        densified 0.9793
Pruning ratio 0.8
    AIRe:
        normal 0.9736
        densified 0.9759
    DepGraph:
        normal 0.9719
        densified 0.9747
IoU's for lucy:
    baseline 0.9498
    densified 0.9556
Pruning ratio 0.4
    AIRe:
        normal 0.9434
        densified 0.9503
    DepGraph:
        normal 0.9457
        densified 0.9546
Pruning ratio 0.6
    AIRe:
        normal 0.9482
        densified 0.9490
    DepGraph:
        normal 0.9408
        densified 0.9535
Pruning ratio 0.8
    AIRe:
        normal 0.9232
        densified 0.9457
    DepGraph:
        normal 0.9241
        densified 0.9464
IoU's for dragon:
mesh loaded!
    baseline 0.9416
    densified 0.9473
    0.4 pruning ratio:
        AIRe:
            normal 0.9405
            densified 0.9487
        DepGraph:
            normal 0.9392
            densified 0.9448
    0.6 pruning ratio:
        AIRe:
            normal 0.9367
            densified 0.9420
        DepGraph:
            normal 0.9377
            densified 0.9468
    0.8 pruning ratio:
        AIRe:
            normal 0.9266
            densified 0.9449
        DepGraph:
            normal 0.9222
            densified 0.9253
IoU's for bunny:
mesh loaded!
    baseline 0.9914
    densified 0.9928
    0.4 pruning ratio:
        AIRe:
            normal 0.9898
            densified 0.9913
        DepGraph:
            normal 0.9911
            densified 0.9925
    0.6 pruning ratio:
        AIRe:
            normal 0.9900
            densified 0.9902
        DepGraph:
            normal 0.9909
            densified 0.9915
    0.8 pruning ratio:
        AIRe:
            normal 0.9881
            densified 0.9900
        DepGraph:
            normal 0.9890
            densified 0.9903

### Frequency Ratio (first layer) after densifying

### All: 59% with omega = 30 

#### Bunny

[2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]
41% with omega = 60

#### Armadillo

3% with omega = 120
38% with omega = 60

[4., 4., 4., 4., 4., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]

#### Lucy

8.6% with omega = 120
32.4% with omega = 60

[4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
        4., 4., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]

#### Dragon

9.4% with omega = 120
31.6% with omega = 60

[4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
        4., 4., 4., 4., 4., 4., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1.]

[ 8.9617e-03,  1.1448e-01, -6.2116e-01],
        [-9.0149e-03, -2.5347e-01, -6.2993e-01],
        [-9.0780e-02, -5.2849e-02, -5.3492e-01],
        [ 2.5326e-01, -8.2212e-03,  7.1155e-01],
        [ 2.8119e-01,  1.4875e-01,  4.8059e-02],
        [-2.0592e-01, -1.7850e-02, -1.7945e-01],
        [-1.3500e-01,  3.0591e-02,  6.9969e-02],
        [-1.4578e-01, -5.1708e-02,  5.9002e-01],
        [-2.1288e-01,  1.3016e-01,  5.2122e-01],
        [ 2.5623e-01,  1.3705e-02, -3.1204e-01],
        [-1.4893e-01, -2.2693e-01, -9.4444e-01],
        [-2.3055e-01,  1.9520e-01,  5.2155e-01],
        [-5.2332e-02, -1.4744e-02,  3.7184e-01],
        [ 1.8943e-01,  1.2629e-01, -9.7732e-01],