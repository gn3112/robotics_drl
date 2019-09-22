from results import results
import matplotlib.pyplot as plt

r=results()
data = r.get_data_multi_exp(['reacher_sac_no_value_fnc_18-06-2019_10-17-44','visual_reacher_new_lr_old_update_21-06-2019_21-34-40'], ['symbolic','visual'])
r.plot_steps(data)
plt.show()
