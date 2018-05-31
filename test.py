from __future__ import division
import utils, pickle
import myhmm_scaled as MyHmm
import scipy.cluster.vq as sp
import numpy as np

def test():
	disease = ['Breast Cancer','Colorectal Cancer','Lung Cancer','Prostate Cancer']
	hmm = {}
	hit_rate = 0
	test_count = 0

	for d in disease:
		hmm[d] = pickle.load(open(d+'.pkl','rb'))
	vs = pickle.load(open('size_mapping.pkl','rb'))
	rev_vs = {}
	for length, diseases in vs.items():
		for d in diseases:
			rev_vs[d] = length
	codebooks = pickle.load(open('all_codebooks.pkl','rb'))

	for line in open('testing_data.txt'):
		test_count+=1
		prob = []		

		line = line.replace("null","0").split(',')
		target_disease, observations = line[0],line[1:]
		observations = map(float,observations)

		for d,h in hmm.items():
			testing_data = []
			obs = observations
			if len(obs)%rev_vs[d] !=0:
				obs = obs+[0]*(rev_vs[d]-(len(obs)%rev_vs[d]))

			#split into rev_vs[d] sizes
			for i in range(0,len(obs),rev_vs[d]):
				testing_data.append(obs[i:i+rev_vs[d]])
			
			n = len(testing_data)
			m = rev_vs[d]
			vq_data = map(str,sp.vq(np.reshape(testing_data,(n,m)), codebooks[m])[0])

			pr = h.backward_scaled(vq_data)
			prob.append((pr,d))

		max_pr = max(prob)
		if(max_pr[1]==target_disease): hit_rate+=1
	print(hit_rate)
	print('Accuracy: '+str(hit_rate*100/test_count))

test()