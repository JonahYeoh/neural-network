from functools import reduce
import random
alist = [ random.randint(0, 100) for i in range(10) ]
print(alist)
helper = lambda x, y: x if x < y else y

mini = reduce(helper, alist)
print(mini)