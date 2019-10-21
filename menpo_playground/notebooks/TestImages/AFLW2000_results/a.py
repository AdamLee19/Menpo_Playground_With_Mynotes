import os

def saveLMK( fName ):
	name = f[:-4]
	fRead = open(fName, 'r')
	fWrite = open(name+'.pts','w')
	fWrite.write('version: 1\nn_points: 68\n{\n')
	a = fRead.readline()
	while a:
		fWrite.write(a)
		a = fRead.readline()
	fWrite.write('}\n')
	fRead.close()
	fWrite.close()



files = os.listdir('.')



lmks = []
for f in files:
	if f.endswith('.txt'):
		saveLMK(f)




