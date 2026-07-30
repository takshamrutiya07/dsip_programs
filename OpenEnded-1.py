# Q-1:import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt

# n = np.arange(-6, 7)
# x = np.where(n % 2 == 0, 3, 2)

# plt.figure()
# plt.stem(n, x)
# plt.title("Signal 1: Odd = 2, Even = 3")
# plt.xlabel("n")
# plt.ylabel("x[n]")
# plt.grid(True)
# plt.show()

# Q-2:
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 6)
x = np.array([1, 2, 3, 2, 1, 1])

plt.figure()
plt.stem(n, x)
plt.title("Signal 2: Staircase Sequence")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()

#Q-3:
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-5, 6)

x = np.where(
    n < -1,
    -1,
    np.where(n > 1, 1, n)
)

plt.figure()
plt.stem(n, x)
plt.title("Signal 3: Limited Ramp Signal")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()


# Q-4:
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-5, 13)

u_n = np.where(n >= 0, 1, 0)
u_n_minus_3 = np.where(n >= 3, 1, 0)
u_n_minus_7 = np.where(n >= 7, 1, 0)

x = u_n - u_n_minus_3 - 5 * u_n_minus_7

plt.figure()
plt.stem(n, x)
plt.title("Signal 4: Unit-Step Combination")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()

# Q-5:
import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-5, 6)

delta_n = np.where(n == 0, 1, 0)
delta_n_minus_1 = np.where(n == 1, 1, 0)
delta_n_plus_1 = np.where(n == -1, 1, 0)

x = delta_n + 3 * delta_n_minus_1 + 5 * delta_n_plus_1

plt.figure()
plt.stem(n, x)
plt.title("Signal 5: Impulse Combination")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()


