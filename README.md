# BLP
This code is used for BLP-random coefficients estimation. Essentially, it tries to replicate the results in 'A Research Assistant's Guide to Random Coefficient Discrete Choice Models of Demand' by Aviv Nevo. While Nevo's GMM objective function's value is 14.9, I extend the number of iterations to reach objective function value 4.56. The results are more precise given Nevo's fake data. However, one problem is that my standard errors are likely to be incorrect. I can't find the bugs yet, likely in the Jacobian function. 

```
Warning: Maximum number of iterations has been exceeded.
         Current function value: 4.562276
         Iterations: 40
         Function evaluations: 675
         Gradient evaluations: 45
Mean Estimates:
               Mean        SD      Income   Income^2       Age      Child
Constant  -2.012682  0.556899    2.297560   0.000000  1.287231   0.000000
Price    -62.531714  3.305179  585.264225 -30.036636  0.000000  11.095503
Sugar      0.116260 -0.005727   -0.384089   0.000000  0.052301   0.000000
Mushy      0.499805  0.092130    0.757584   0.000000 -1.359024   0.000000
Standard Errors:
               Mean        SD     Income    Income^2       Age     Child
Constant   0.326163  0.162155   1.334189    0.000000  0.013491  0.000000
Price     14.740366  0.185479   1.207367  269.330067  0.000000  0.120993
Sugar      0.016024  0.800234  14.043097    0.000000  0.633089  0.000000
Mushy      0.198622  0.025923   0.668999    0.000000  4.124887  0.000000
--- 359.14806294441223 seconds ---
```

# Comparison: 
You can also refer the Matlab codes provided by Prof. Rasmusen. He re-wrote Matlab codes from Nevo and Hall. It's working on latest Matlab 2017. The computation is similar to the Python codes here, and the standard errors are more likely to be correct. However, in the main .m file, instead of "semcoef = [semd(1); se(1); semd]", it should be "semcoef = [semd(1); se(1); semd(2:3)]".  

Here's the link:
http://www.rasmusen.org/zg604/lectures/blp/frontpage.htm

Another reference is from Prof. Ro. He used Cython and gradients to speed up the convergence. I can't make the gradient to work because the Jacobian is likely to be incorrect. 

Here's his link: 
https://github.com/joonro/BLP-Python
