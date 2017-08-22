# BLP
This code is for BLP-random coefficients estimation. Essentially, it tries to replicate the results in 'A Research Assistant's Guide to Random Coefficient Discrete Choice Models of Demand' by Aviv Nevo. While Nevo's GMM objective function's value is 14.9, I extend the number of iterations to reach objective function value 4.56. The results are more precise given Nevo's fake data. 

# Replication for Nevo (2000) Table I:

```
Warning: Maximum number of iterations has been exceeded.
         Current function value: 14.895122
         Iterations: 25
         Function evaluations: 450
         Gradient evaluations: 30
Mean Estimates:
               Mean        SD     Income  Income^2       Age      Child
Constant  -1.952633  0.368435   3.191387  0.000000  1.078232   0.000000
Price    -32.228194  1.849553  17.493097 -0.736473  0.000000  11.015146
Sugar      0.138623 -0.002771  -0.194065  0.000000  0.031475   0.000000
Mushy      0.616595  0.073966   1.411502  0.000000 -1.407856   0.000000
Standard Errors:
              Mean        SD      Income  Income^2       Age     Child
Constant  0.255334  0.125979    1.137800  0.000000  0.898642  0.000000
Price     7.785776  1.040751  173.058851  8.991769  0.000000  4.867536
Sugar     0.012876  0.011941    0.044673  0.000000  0.034061  0.000000
Mushy     0.187898  0.195609    0.662680  0.000000  1.001100  0.000000
--- 321.60219740867615 seconds ---
```
There are some slight differences, which are probably due to the value of objective function and minimization method 

# 2017/8/22 update:

I found the bug which was cuasing wrong standard errors. Now the estimates and standard errors are correct. The minimum of the objective function is reached at around 45 iteration. It takes less than 10 mins to compute the results. 

```
Warning: Maximum number of iterations has been exceeded.
         Current function value: 4.561534
         Iterations: 50
         Function evaluations: 1170
         Gradient evaluations: 78
Mean Estimates:
               Mean        SD      Income   Income^2       Age      Child
Constant  -2.010873  0.558518    2.293647   0.000000  1.284360   0.000000
Price    -62.791246  3.316215  589.490516 -30.252891  0.000000  11.050478
Sugar      0.116244 -0.005813   -0.385428   0.000000  0.052287   0.000000
Mushy      0.499868  0.093713    0.746653   0.000000 -1.353060   0.000000
Standard Errors:
               Mean        SD      Income   Income^2       Age     Child
Constant   0.327211  0.162715    1.210062   0.000000  0.630978  0.000000
Price     14.827514  1.342623  270.905894  14.125879  0.000000  4.122291
Sugar      0.016047  0.013513    0.121682   0.000000  0.026011  0.000000
Mushy      0.198686  0.185443    0.803185   0.000000  0.666919  0.000000
--- 602.1178584098816 seconds ---
```

# Comparison: 
You can also refer the Matlab codes provided by Prof. Rasmusen. He re-wrote Matlab codes from Nevo and Hall. It's working on latest Matlab 2017. The computation is similar to the Python codes here, and the standard errors are more likely to be correct. However, in the main .m file, instead of "semcoef = [semd(1); se(1); semd]", it should be "semcoef = [semd(1); se(1); semd(2:3)]".  

Here's the link:
http://www.rasmusen.org/zg604/lectures/blp/frontpage.htm

Another reference is from Prof. Ro. He used Cython and gradients to speed up the convergence. I can't make the gradient to work because the Jacobian is likely to be incorrect. 

Here's his link: 
https://github.com/joonro/BLP-Python
