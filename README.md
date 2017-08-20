# BLP
This code is used for BLP-random coefficients estimation. Essentially, it tries to replicate the results in 'A Research Assistant's Guide to Random Coefficient Discrete Choice Models of Demand' by Aviv Nevo. While Nevo's GMM objective function's value is 14.9, I extend the number of iterations to reach objective function value 4.56. The results are more precise given Nevo's fake data. However, one problem is that my standard errors are likely to be incorrect. I can't find the bugs yet, likely in the Jacobian function. 



# Comparison: 
You can also refer the Matlab codes provided by Prof. Rasusen. He re-wrote Matlab codes from Nevo and Hall. It's working on latest Matlab. The computation is similar to the Python codes here, and the standard errors are more likely to be correct. However, in the main .m file, instead of "semcoef = [semd(1); se(1); semd]", it should be "semcoef = [semd(1); se(1); semd(2:3)]".  

http://www.rasmusen.org/zg604/lectures/blp/frontpage.htm


