import pickle
import pdb
import numpy as np

f_des = open('DATA/design','rb')
DATA_design = pickle.load(f_des)
f_des.close()

f_sla = open('DATA/slack','rb')
DATA_slack_list = pickle.load(f_sla)
f_sla.close()
DATA_slack = np.concatenate((DATA_slack_list[0], DATA_slack_list[1], DATA_slack_list[2]), axis=0)

f_dul = open('DATA/dual','rb')
DATA_dual_list = pickle.load(f_dul)
f_dul.close()
DATA_dual = np.concatenate((DATA_dual_list[0], DATA_dual_list[1], DATA_dual_list[2]), axis=0)





f_des1 = open('design','rb')
design = pickle.load(f_des1)
f_des1.close()

f_sla1 = open('slack','rb')
slack_l = pickle.load(f_sla1)
f_sla1.close()

slack = np.concatenate((slack_l[0], slack_l[1], slack_l[2]), axis=0)

f_dul1 = open('dual','rb')
dual_l = pickle.load(f_dul1)
f_dul1.close()

dual = np.concatenate((dual_l[0], dual_l[1], dual_l[2]), axis=0)


diff_design = np.linalg.norm(DATA_design - design)
diff_slack = np.linalg.norm(DATA_slack - slack)
diff_dual = np.linalg.norm(DATA_dual - dual)

print diff_design, diff_slack, diff_dual
