w = 28
h = 28
tw = 16
th = 16
w, h = 12, 12
pixels = [i/(w*h) for i in range(1, w*h+1)]

w_ratios = [i*(w/tw)+(w/(2*tw)) for i in range(tw)]
h_ratios = [i*(h/th)+(h/(2*th)) for i in range(th)]

print(w_ratios)
w_points = []
for y in h_ratios:
    for x in w_ratios:
        #print(x)
        #w_points.append((int(x-.5), int(x+.4999)))
        x0 = int(x-.5)
        x1 = int(x+0.5)
        x1idx = min(x1, w-1)
        #x1idx = x1
        y0 = int(y-.5)
        y1 = int(y+0.5)
        y1idx = min(y1, h-1)
        #y1idx = y1
        #print(x0)
        #print(x1)
        #print(x)
        huy = (x1+0.5 - x)*pixels[x0+y0*w] + (x-(x0+0.5))*pixels[x1idx+y0*w]
        luy = (x1+0.5 - x)*pixels[x0+y1idx*w] + (x-(x0+0.5))*pixels[x1idx+y1idx*w]
        huu = (y1+0.5-y)*huy + (y-(y0+0.5))*luy
        print(huy)
        print(luy)
        print(huu)
        #print(ww[x0])
        #print(ww[x1idx])
        print("---")

print(w_points)
