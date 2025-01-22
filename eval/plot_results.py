import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys


def parse(d):
	dictionary = dict()
	# Removes curly braces and splits the pairs into a list
	pairs = d.split(", '")
	# print(pairs)
	# print('-------')
	for i in pairs:
		pair = i.split("': ")
		# Other symbols from the key-value pair should be stripped.
		dictionary[pair[0].replace("'",'')] = pair[1].replace('\n','').replace('}','').replace(']]',']')
	return dictionary

if __name__ == "__main__":
	resultfile=sys.argv[1]
	# resultfile='20250109_175735.txt'

	resultsread = open(resultfile, 'rt')
	lines = resultsread.read().split('\n[{')
	config_file=lines[0]
	newlines=lines[1].split('{')
	dic=[]
	for l in newlines:
		if l != '' and 'yml' not in l:
			dic.append(parse(l))
	resultsread.close()

	pyresults=[]
	matresults=[]
	for i,el in enumerate(dic):
		pyresults.append(float(el['pyresult'].split(',')[3]))
		matresults.append(float(el['matresults'].split(',')[3]))

	fig, ax = plt.subplots()
	
	bwidth=0.5
	data=[pyresults, matresults]
	
	bins=np.arange(min(min(data)), max(max(data))+bwidth,bwidth)
	ax.hist(data,bins, density=True, histtype="step", stacked=False, label=['Python', 'Matlab'])
	ax.legend()
	ax.set_title('Mean calculated SRT after 40 trials')

	plt.savefig(resultfile.replace('txt', 'png'), bbox_inches='tight')



