import numpy as np
def evaluate_uplift(y_true, uplift, treatment, treatment_indicator=1, control_indicator=0, n_bins=10):
    def r_squared_bam(y, y_hat):
        ss_tot = ((y-y.mean())**2).sum()
        ss_res = ((y-y_hat)**2).sum()
        return 1 - (ss_res/ss_tot),((y-y_hat)**2).mean()
    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]
    bin_res_list, bin_range = [], np.linspace(0, len(uplift), n_bins+1).astype(int)
    trea_count, cont_count = len(y_true[treatment==treatment_indicator]), len(y_true[treatment==control_indicator])
    
    for i in range(bins):
        y_true_bucket = y_true[bin_range[i]:bin_range[i+1]]
        treatment_bucket = treatment[bin_range[i]:bin_range[i+1]]
        uplift_bucket = uplift[bin_range[i]:bin_range[i+1]]
        pred_uplift = np.mean(uplift_bucket)
        true_uplift = np.mean(y_true_bucket[treatment_bucket==treatment_indicator]) - np.mean(y_true_bucket[treatment_bucket==control_indicator])
        y_true_cum = y_true[0:bin_range[i+1]]
        treatment_cum = treatment[0:bin_range[i+1]]
        # true_cum_uplift = (i+1)/bins * (np.mean(y_true_cum[treatment_cum==1]) - np.mean(y_true_cum[treatment_cum==0]))
        true_cum_uplift = np.sum(y_true_cum[treatment_cum==treatment_indicator]) /trea_count - np.sum(y_true_cum[treatment_cum==control_indicator]) /cont_count
        bin_res_list.append(np.array([true_uplift, len(y_true_cum)/len(y_true), true_cum_uplift, pred_uplift]))

    bin_range =np.append([0], np.array(bin_res_list)[:,1])
    qini_range =np.append([0], np.array(bin_res_list)[:,2])
    auuc = auc(bin_range,qini_range/qini_range[-1])
    calibration_r2_score, calibration_mse = r_squared_bam(np.array(bin_res_list)[:,0], np.array(bin_res_list)[:,-1])
    return auuc, calibration_r2_score, calibration_mse, {"bin_range": bin_range, "qini_range": qini_range}, {"bin_range":np.array(bin_res_list)[:,1], "uplift_true_range": np.array(bin_res_list)[:,0], "uplift_pred_range": np.array(bin_res_list)[:,-1]}


import pylift
def evaluate_uplift_with_pylift(uplift, T_test, Y_test, treatment_indicator=1, control_indicator=0, n_bins=10):
    def r_squared_bam(y, y_hat):
        ss_tot = ((y-y.mean())**2).sum()
        ss_res = ((y-y_hat)**2).sum()
        return 1 - (ss_res/ss_tot),((y-y_hat)**2).mean()
    uplift,T_test,Y_test = np.array(uplift).reshape(-1),np.array(T_test).reshape(-1),np.array(Y_test).reshape(-1)

    pylift_eval_norm = pylift.eval.UpliftEval(T_test, Y_test, uplift, n_bins=n_bins)
    bin_range = np.linspace(0, len(uplift), n_bins+1).astype(int)
    true_uplift_bin = np.array(pylift_eval_norm.calc("uplift", n_bins= n_bins)[1])[::-1]
    pre_uplift_bin = np.array([np.mean(uplift[bin_range[i]:bin_range[i+1]]) for i in range(n_bins)])

    lift_culmuate = pylift_eval_norm.calc("qini", n_bins= n_bins)
    qini_test_norm = auc(lift_culmuate[0], lift_culmuate[1]/lift_culmuate[-1])
    return pylift_eval_norm, [lift_culmuate[-1],np.mean(uplift), qini_test_norm], r_squared(true_uplift_bin, pre_uplift_bin), [true_uplift_bin,pre_uplift_bin]