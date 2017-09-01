# BLP
This code is for BLP-random coefficients estimation. Essentially, it tries to replicate the results in 'A Research Assistant's Guide to Random Coefficient Discrete Choice Models of Demand' by Aviv Nevo. While Nevo's GMM objective function's value is 14.9, I extend the number of iterations to reach objective function value 4.56. The results are more precise given Nevo's fake data. 

# Replication for Nevo (2000) Table I:

```
Mean Estimates:
               Mean        SD     Income  Income^2       Age      Child
Constant  -1.956236  0.367702   3.188302  0.000000  1.091190   0.000000
Price    -32.164280  1.862521  15.462670 -0.629078  0.000000  10.954305
Sugar      0.139150 -0.002463  -0.193821  0.000000  0.031423   0.000000
Mushy      0.648903  0.077208   1.433925  0.000000 -1.432826   0.000000
Standard Errors:
              Mean        SD      Income  Income^2       Age     Child
Constant  0.254098  0.126959    1.153352  0.000000  0.918825  0.000000
Price     7.716360  1.046685  172.175621  8.946287  0.000000  4.901362
Sugar     0.012845  0.012000    0.044427  0.000000  0.034553  0.000000
Mushy     0.191134  0.196911    0.671257  0.000000  1.021373  0.000000
--- 214.407812833786 seconds ---
```
There are some slight differences, which are probably due to the value of objective function and minimization method 

# 2017/8/22 update:

I found the bug which was cuasing wrong standard errors. Now the estimates and standard errors are correct. The minimum of the objective function is reached at around 45 iteration. It takes around 7 mins to compute the results. 

```
Warning: Maximum number of iterations has been exceeded.
         Current function value: 4.561597
         Iterations: 50
         Function evaluations: 885
         Gradient evaluations: 59
Mean Estimates:
               Mean        SD      Income   Income^2       Age      Child
Constant  -2.011406  0.558994    2.293785   0.000000  1.284469   0.000000
Price    -62.856428  3.320055  590.672038 -30.314168  0.000000  11.047769
Sugar      0.116232 -0.005842   -0.385869   0.000000  0.052343   0.000000
Mushy      0.500210  0.094235    0.745010   0.000000 -1.352803   0.000000
Standard Errors:
               Mean        SD      Income   Income^2       Age    Child
Constant   0.327458  0.162904    1.211498   0.000000  0.630778  0.00000
Price     14.851506  1.345163  271.358937  14.149859  0.000000  4.12216
Sugar      0.016058  0.013522    0.121895   0.000000  0.026040  0.00000
Mushy      0.198818  0.185449    0.804220   0.000000  0.666749  0.00000
--- 424.098836183548 seconds ---
```

# Comparison: 
You can also refer the Matlab codes provided by Prof. Rasmusen. He re-wrote Matlab codes from Nevo and Hall. It's working on latest Matlab 2017. The computation is similar to the Python codes here, and the standard errors are more likely to be correct. However, in the main .m file, instead of "semcoef = [semd(1); se(1); semd]", it should be "semcoef = [semd(1); se(1); semd(2:3)]".  

Here's the link:
http://www.rasmusen.org/zg604/lectures/blp/frontpage.htm

Another reference is from Prof. Ro. He used Cython and gradients to speed up the convergence. I can't make the gradient to work because the Jacobian is likely to be incorrect. 

Here's his link: 
https://github.com/joonro/BLP-Python
