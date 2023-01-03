# filling-series-lr
This code makes use of linear regression (either multiple or simple) to fill gaps in time series.


This code makes use of linear regression (either multiple or simple) to fill gaps in time series.
You can make use of those two additional conditions in order to proceed with your calculations:
1. You can use a filter to correct just some specific points and not all the dataset. However pay attention that your entire dataset will still be used as potential indepent variables (Xs);
2. You can make use of the distance as a limitant of your analysis, i.e., you can define a maximum number of points (n) to be used as variable independents (X) of you variable to be predicted (y) and set only the n-closest points to be selected;
3. Moreover, you can also define that only Xs with |t-statistics| higher than 2 will be used in the MLR computation to garantee that the sigficance is higher than 0¶

You can read read the tutorial of application of the "fillinggapsts" library at the Jupyter notebook called: 1_example.
Alternatively, the jupyter notebook called: 0_fillinggapsts-explained provides a further description of the methodology used in "fillinggapsts". 
