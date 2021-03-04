import numpy as np


#Creating a blank message:
MESSAGE = ['--'] * 48

# Adding some pumping times...
for x in [1,5,8,9]:
    # influx pump
    time_in = x
    time_out = time_in + 5
    MESSAGE[x] = str(time_in)
    # efflux pump
    MESSAGE[x + 16] = str(time_out)

print("Initial Message", MESSAGE)
#Finding the max pump in message

num_message = [int(x) for x in MESSAGE if x != '--']
print("Number message", num_message)
max_time = max(num_message)
print(max_time)

for index in range(16, 32):
    MESSAGE[index] = str(max_time)

print("Updated pumping message:", MESSAGE)