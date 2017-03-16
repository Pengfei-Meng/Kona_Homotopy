# fstopo_kona_latest
This folder is connected with Developer/kona_mengp2 folder on my Lenovo Ubuntu Laptop. 
It contains the work on testing homotopy inequality and approximate-adjoint preconditioner,\
SVD Jacobian preconditioner on Graeme's structural problem, Toy problem. 

Major achievements so far:
1. Homotopy Inequality working for Cuter test problem sets
2. predictor_corrector_cnstr_inequ.py inside Kona is the complete homotopy algorithm
3. predictor_corrector_conditional.py provides the option to turn on Corrector after mu gets lower than certain value
4. approximate adjoint PC working great on Structural Tiny problem, but is too expensive to be applied to bigger problem
5. SVD_Jacobian preconditioner is working great for the constructed Toy problem
6. Optimization results from SNOPT and KONA are being made currently. 
