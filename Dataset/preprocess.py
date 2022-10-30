import json

word2cnt = dict()
with open('data.json') as fin:
	for line in fin:
		data = json.loads(line)
		text = data['text'].split()
		for word in text:
			if word not in word2cnt:
				word2cnt[word] = 0
			word2cnt[word] += 1

train = 240000
with open('data.json') as fin, open('train.txt', 'w') as fou1, open('test.txt', 'w') as fou2:
	for idx, line in enumerate(fin):
		data = json.loads(line)

		d = data['language'] + ' ' + data['timezone'] + ' ' + data['offset'] + ' ' + \
			data['userlang'] + ' ' + str(data['label']) + ' ' + data['latitude'] + ' ' + data['longitude']

		text = data['text'].split()
		ws = []
		for word in text:
			if word2cnt[word] <= 10000 and word2cnt[word] >= 5:
				ws.append(word)

		d += ' ' + str(len(ws)) + ' ' + ' '.join(ws) + '\n'

		if idx < train:
			fou1.write(d)
		else:
			fou2.write(d)