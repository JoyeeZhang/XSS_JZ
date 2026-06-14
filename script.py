import numpy as np
import matplotlib.pyplot as plt

# torus parameters
R = 5.0
r = 2.0
d = 4.0   # x = d plane

theta = np.linspace(0, 2*np.pi, 5000)

rho = R + r * np.cos(theta)
z = r * np.sin(theta)

mask = rho**2 >= d**2

y = np.sqrt(rho[mask]**2 - d**2)

y0 = np.concatenate([y, -y[::-1]])
z0 = np.concatenate([z[mask], z[mask][::-1]])

plt.figure(figsize=(5,5))
plt.plot( y0, z0, 'b')
plt.axis('equal')

plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)

plt.xlabel('y')
plt.ylabel('z')
plt.axis('equal')
plt.title('Cross section of torus at x = d')
plt.show()
