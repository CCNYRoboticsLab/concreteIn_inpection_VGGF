import os
import re

prepath130='/home/robolab/project/deepProjectBridge/deepLearningBridgeInspection/imageClustersV130B130/'

prepath100='/home/robolab/project/deepProjectBridge/deepLearningBridgeInspection/imageClustersV2100B100/'

ft=open('train_list.txt','w')
fs=open('test_list.txt','w')



trlist130=os.listdir(prepath130+'accepted')
trlist100=os.listdir(prepath100+'accepted')
#print len(trlist)
#print trlist[0]
for i,s in enumerate(trlist130):
	if s[0]!='.':
		if i<len(trlist130)*0.75:
			ft.write('imageClustersV130B130/accepted/'+s+','+str(1)+'\n')
		else:
			fs.write('imageClustersV130B130/accepted/'+s+','+str(1)+'\n')

for i,s in enumerate(trlist100):
	if s[0]!='.':
		if i<len(trlist100)*0.75:
			ft.write('imageClustersV2100B100/accepted/'+s+','+str(1)+'\n')
		else:
			fs.write('imageClustersV2100B100/accepted/'+s+','+str(1)+'\n')

tslist130=os.listdir(prepath130+'nonAccepted')
tslist100=os.listdir(prepath100+'nonAccepted')
#print len(tslist)

for i,s in enumerate(tslist130):
	if s[0]!='.':
		if i<len(tslist130)*0.75:
			ft.write('imageClustersV130B130/nonAccepted/'+s+','+str(0)+'\n')
		else:
			fs.write('imageClustersV130B130/nonAccepted/'+s+','+str(0)+'\n')

for i,s in enumerate(tslist100):
	if s[0]!='.':
		if i<len(tslist100)*0.75:
			ft.write('imageClustersV2100B100/nonAccepted/'+s+','+str(0)+'\n')
		else:
			fs.write('imageClustersV2100B100/nonAccepted/'+s+','+str(0)+'\n')
