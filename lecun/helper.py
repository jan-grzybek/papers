class Registry:
    def __init__(self):
        self.occupied = []
        self.max_ever = -1

    def get(self):
        x = 0
        new = False
        while x in self.occupied:
            x += 1
        if x > self.max_ever:
            self.max_ever = x
            new = True
        self.occupied.append(x)
        return f"d{x}", new


registry = Registry()


def deallocate(data_name):
    if data_name[0] == "d":
        registry.occupied.remove(int(data_name[1:]))
    print(f"data_deallocate(&{data_name});")


class Loss:
    def __init__(self):
        self.children = []

    def derivative(self, data=None):
        for child in self.children:
            child.derivative("loss_derivative")


class Input:
    def __init__(self):
        self.name = "input"

    def derivative(self, data=None):
        return


class FC:
    def __init__(self, name):
        self.name = name
        self.children = []

    def derivative(self, data=None):
        print()
        bias, new = registry.get()
        if new:
            print(f"Data {bias} = lenet->{self.name}.bias_activation_backward(&lenet->{self.name}, &{data});")
        else:
            print(f"{bias} = lenet->{self.name}.bias_activation_backward(&lenet->{self.name}, &{data});")
        deallocate(data)
        kernels = []
        for child in self.children:
            kernel, new = registry.get()
            kernels.append(kernel)
            if new:
                print(f"Data {kernel} = lenet->{self.name}.madd_backward(&lenet->{self.name}, &lenet->{child.name}.output, &{bias});")
            else:
                print(
                    f"{kernel} = lenet->{self.name}.madd_backward(&lenet->{self.name}, &lenet->{child.name}.output, &{bias});")
        deallocate(bias)
        for i, child in enumerate(self.children):
            child.derivative(kernels[i])


class Conv2D:
    def __init__(self, name):
        self.name = name
        self.children = []

    def derivative(self, data=None):
        print()
        bias, new = registry.get()
        if new:
            print(f"Data {bias} = lenet->{self.name}.bias_activation_backward(&lenet->{self.name}, &{data});")
        else:
            print(f"{bias} = lenet->{self.name}.bias_activation_backward(&lenet->{self.name}, &{data});")
        deallocate(data)
        kernels = []
        for child in self.children:
            if child.name != "input":
                kernel, new = registry.get()
                kernels.append(kernel)
                if new:
                    print(f"Data {kernel} = lenet->{self.name}.kernel_backward(&lenet->{self.name}, &lenet->{child.name}.output, &{bias}, false);")
                else:
                    print(
                        f"{kernel} = lenet->{self.name}.kernel_backward(&lenet->{self.name}, &lenet->{child.name}.output, &{bias}, false);")
            else:
                print(f"lenet->{self.name}.kernel_backward(&lenet->{self.name}, {child.name}, &{bias}, true);")
        deallocate(bias)
        for i, child in enumerate(self.children):
            try:
                child.derivative(kernels[i])
            except IndexError:
                pass


Loss = Loss()
Inp = Input()
FC2 = FC("FC2")
FC1 = FC("FC1")
H2 = [Conv2D(f"H2_{i}") for i in range(1, 13)]
indices = [[1, 2, 4, 5, 7, 8, 10, 11], [1, 2, 4, 5, 7, 8, 10, 11], [1, 2, 4, 5, 7, 8, 10, 11], [1, 2, 4, 5, 7, 8, 10, 11],
           [1, 3, 4, 6, 7, 9, 10, 12], [1, 3, 4, 6, 7, 9, 10, 12], [1, 3, 4, 6, 7, 9, 10, 12], [1, 3, 4, 6, 7, 9, 10, 12],
           [2, 3, 5, 6, 8, 9, 11, 12], [2, 3, 5, 6, 8, 9, 11, 12], [2, 3, 5, 6, 8, 9, 11, 12], [2, 3, 5, 6, 8, 9, 11, 12]]
H1 = [Conv2D(f"H1_{i}") for i in range(1, 13)]
x = {i: 0 for i in range(1, 13)}
for h, idx in zip(H2, indices):
    for i in idx:
        x[i] += 1
    h.children = [H1[i-1] for i in idx]
for h in H1:
    h.children.append(Inp)
assert all([x[i] == 8 for i in range(1, 13)])
Loss.children.append(FC2)
FC2.children.append(FC1)
FC1.children = H2
Loss.derivative()
assert len(registry.occupied) == 0
