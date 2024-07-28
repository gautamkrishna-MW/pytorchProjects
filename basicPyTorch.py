import torch

# t = torch.tensor([[1.,2,3],[4,5,6]])
# # print(t.shape)
# # print(t.ndim)
# # print(t.dtype)
#
# import numpy
# n = numpy.random.randn(3,4)
# # print(n.dtype)
# t = torch.from_numpy(n)
# # print(t.view(2,-1,2))
#
#
# from torch.autograd import Variable
# numpy.random.seed(20)
# n = numpy.floor(numpy.random.random([4,6])*10)
# trArr = torch.tensor(n, requires_grad=True)
# outArr = torch.sum(trArr**2)
# print(trArr)
# print(outArr)
# outArr.backward()
# print(trArr.grad)



# Simple LinReg
torch.manual_seed(256)

inp = torch.rand(12,8)
weights = torch.rand(4,8)
out = torch.rand(12,4)
bias = torch.ones(4)
print(inp)
print(weights)
print(out)
print(bias)

iter = 0
errArr = []
learningRate = 0.01
while iter < 1000:
    weights.requires_grad = True
    bias.requires_grad = True

    outPred = torch.mm(inp, weights.t()) + bias
    diffVal = out - outPred
    errorVal = torch.sum(diffVal.pow(2)/diffVal.numel())
    errArr.append(errorVal.data)
    print(f"Error Val: {errorVal}")
    errorVal.backward()
    iter += 1

    with torch.no_grad():
        weights = weights - (learningRate * weights.grad)
        bias = bias - (learningRate * bias.grad)

from matplotlib import pyplot
pyplot.plot(errArr)
pyplot.title("Error Plot")
pyplot.show()