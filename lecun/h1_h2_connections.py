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
print()
for h2_, h1 in links.items():
    for h1_ in h1:
        print(f"lenet->H2_{h2_+1}.forward(&lenet->H2_{h2_+1}, &lenet->H1_{h1_+1}.output);")
    print()
