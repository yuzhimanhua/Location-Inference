import string
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

y = []
y_pred = []

tot = 0
dist_161 = 0
dist_sum = 0.0

with open('result.txt') as fin:
	for line in fin:
		tmp = line.strip().split()
		y.append(tmp[0])
		y_pred.append(tmp[1])

		dist_sum += float(tmp[2])
		if float(tmp[2]) <= 161:
			dist_161 += 1
		tot += 1

print('Mean Distance Error:', dist_sum/tot)
print('Accuray at 161km:', dist_161/tot)

print('Micro-F1:', f1_score(y, y_pred, average='micro'))
print('Macro-F1:', f1_score(y, y_pred, average='macro'))

print(confusion_matrix(y, y_pred))