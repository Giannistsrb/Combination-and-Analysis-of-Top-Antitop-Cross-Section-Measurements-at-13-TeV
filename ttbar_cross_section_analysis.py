import numpy as np
import math 
from numpy.linalg import inv

# We have in this problem 4 calculations for σ[pb] from 4 different experiments: 1. ATLAS, 2. CMS dilepton,
# 3. CMS I + Jets, 4. CMS Hadronic.
# The table with the calculations and the uncertaincies due to statistics, systematic
# uncertaincies and luminocity are below:
# Calculation:     σ(pb)           stat. unc. (pb)    syst. unc. (pb)    lumi. unc. (pb)
# =========================================================================================
# ATLAS dilepton   818             8                  27                 19
# CMS dilepton     815             9                  38                 19
# CMS l+jets       888             2                  27                 20
# CMS hadronic     834             25                 118                23

#Diagonal elements for each of uncertainces:
stat_unc = np.array([8,   9,  2,  25])
syst_unc = np.array([27, 38, 27, 118])
lumi_unc = np.array([19, 19, 20,  23])

#Diagonal elements for each of variances:
stat_var = np.outer(stat_unc, stat_unc)
syst_var = np.outer(syst_unc, syst_unc)
lumi_var = np.outer(lumi_unc, lumi_unc)

#Correlation matrices:
corr_stat = np.diag([1, 1, 1, 1])

corr_syst = np.array([[1,   0.2, 0.2, 0.2], 
                      [0.2, 1,   0.4, 0.4], 
                      [0.2, 0.4, 1,   0.4], 
                      [0.2, 0.4, 0.4,  1]])

corr_lumi = np.array([[1,  0.9, 0.9, 0.9], 
                      [0.9, 1,   1,   1 ], 
                      [0.9, 1,   1,   1 ], 
                      [0.9, 1,   1,   1 ]])


#Covariance matrices for every uncertaince:
cov_mat_stat = stat_var * corr_stat
cov_mat_syst = syst_var * corr_syst
cov_mat_lumi = lumi_var * corr_lumi

#Print the results: 
print("=============================================")
print("The covariance matrix due to statistical uncertainces:")
print(cov_mat_stat)
print("=============================================")
print("The covariance matrix due to systematic uncertainces:")
print(cov_mat_syst)
print("=============================================")
print("The covariance matrix due to luminocity uncertainces:")
print(cov_mat_lumi)

#TOTAL COVARIANT MATRIX OF THE CALCULATIONS:
tot_cov_mat = cov_mat_stat + cov_mat_syst + cov_mat_lumi

#Print the total covariant matrix:
print("=============================================")
print("THE COVARIANT MATRIX OF THE CALCULATIONS IS:")
print(tot_cov_mat)

#σ final estimation using BLUE method:
u               = np.array([[1], [1], [1], [1]])
uT              = np.array([1, 1, 1, 1])

tot_inv_cov_mat = inv(tot_cov_mat)
uTinv_cov_mat   = np.dot(uT, tot_inv_cov_mat)
weights = np.dot(tot_inv_cov_mat, u) / np.dot(uTinv_cov_mat, u)

#Estimation of cross section (and sigma of cross section) according to the BLUE method:
cross_sections            =    np.array([818, 815, 888, 834]) #for each experiment
final_cross_section       =    np.dot(cross_sections, weights)
wTtot_cov_mat             =    np.dot(np.transpose(weights), tot_cov_mat)
final_sigma_cross_section =    math.sqrt(np.dot(wTtot_cov_mat, weights))


print("=============================================")
print("THE INVERSE COVARIANCE MATRIX IS:")
print(tot_inv_cov_mat)
print("=============================================")
print("THE WEIGHTS USING THE BLUE METHOD ARE:")
print(weights)
print("=============================================")
print("THE FINAL ESTIMATION OF CROSS SECTION IS:")

print("σ = ", final_cross_section[0], "±", final_sigma_cross_section)
