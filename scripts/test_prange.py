

import numpy as np
test = [[59337.36657826339, 38104.490731338585], (59336.82627594723, 38105.33220231678), [np.nan, np.nan], [np.nan, np.nan], [-59337.36657810833, -38094.49073123902], (-59336.82627594723, -38095.33220231678)]


for j in range(6):
	print(test[j])
	if test[j] == [np.nan, np.nan]:
		print ('gg')