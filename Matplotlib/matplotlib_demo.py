import matplotlib.pyplot as plt
import numpy as np #Visualizing Tons of Data – What Crowded Looks Like
 
for i in range(50):
    plt.plot(np.random.rand(100), linewidth=1)
 
plt.title("Too Much Data Can Be Confusing!")
plt.grid(True)
plt.tight_layout()
plt.show()