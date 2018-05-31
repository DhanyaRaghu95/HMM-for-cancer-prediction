import utils, pickle
import myhmm_scaled as MyHmm

def train():
	hmm = {}
	disease = ['Breast Cancer','Colorectal Cancer','Lung Cancer','Prostate Cancer']
	for d in disease:
		hmm[d] = MyHmm.MyHmmScaled('initial.json')
	trng, vec_sizes = utils.obtain_training_data()
	vq_data = utils.vector_quantize(trng, vec_sizes, 16)	

	for d,h in hmm.items():
		h.forward_backward_multi_scaled([vq_data[d]])
		pickle.dump(h, open(d+".pkl","wb"))

train()