import math, random

nbExamples = 500

print(nbExamples, 4, 1)
for i in range(nbExamples):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    x3 = random.uniform(-1, 1)
    x4 = random.uniform(-1, 1)
    print(x1, x2, x3, x4, math.sin(x1-x2+x3-x4))
