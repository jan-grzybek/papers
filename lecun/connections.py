x = {i: 0 for i in range(12)}
links = {i: [] for i in range(12)}

for t in range(12):
	for i, k in enumerate(sorted(x.items(), key=lambda x:x[1])):
		if i == 8:
			break
		x[k[0]] += 1
		links[k[0]].append(t)

print(x)
print(links)
