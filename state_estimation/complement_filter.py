import math
import numpy as np
import matplotlib.pyplot as plt
class ComplementaryFilter:
    def __init__(self, acc_weight=0.98, gyro_weight=0.02, dt=0.01):
        self.acc_weight = acc_weight
        self.gyro_weight = gyro_weight
        self.dt = dt
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
    
    def update(self, gyro_data, acc_data):
        # Calculate roll and pitch from accelerometer data
        acc_pitch = math.atan2(acc_data[0], math.sqrt(acc_data[1]**2 + acc_data[2]**2))
        acc_roll = math.atan2(-acc_data[1], acc_data[2])
        
        # Integrate gyroscope data to obtain roll and pitch
        gyro_pitch = self.pitch + gyro_data[0] * self.dt
        gyro_roll = self.roll + gyro_data[1] * self.dt
        
        # Combine accelerometer and gyroscope data using complementary filter
        self.pitch = self.acc_weight * acc_pitch + self.gyro_weight * gyro_pitch
        self.roll = self.acc_weight * acc_roll + self.gyro_weight * gyro_roll
        
        # Calculate yaw from gyroscope data (assuming device is not accelerating)
        self.yaw += gyro_data[2] * self.dt
    
    def get_orientation(self):
        # Return current roll, pitch, and yaw
        return np.array([self.roll, self.pitch, self.yaw])



# Generate simulated data
t = np.linspace(0, 10, 1000)
gyro_data = np.zeros((1000, 3))
acc_data = np.zeros((1000, 3))
for i in range(1000):
    gyro_data[i] = [0.1, 0.2, 0.3] + 0.01 * np.random.randn(3)  # simulate gyroscope data
    acc_data[i] = [np.sin(0.5 * t[i]), np.sin(0.3 * t[i]), np.cos(0.2 * t[i])] + 0.1 * np.random.randn(3)  # simulate accelerometer data

# Create Complementary Filter instance
cf = ComplementaryFilter(acc_weight=0.98, gyro_weight=0.02, dt=0.001)

# Run filter on data and store orientation estimates
orientation = np.zeros((1000, 3))
for i in range(1000):
    cf.update(gyro_data[i], acc_data[i])
    orientation[i] = cf.get_orientation()

# Plot results
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=(8, 8))
ax[0].plot(t, orientation[:, 0], label='Roll')
ax[0].set_ylabel('Roll (rad)')
ax[0].grid()
ax[1].plot(t, orientation[:, 1], label='Pitch')
ax[1].set_ylabel('Pitch (rad)')
ax[1].grid()
ax[2].plot(t, orientation[:, 2], label='Yaw')
ax[2].set_ylabel('Yaw (rad)')
ax[2].set_xlabel('Time (s)')
ax[2].grid()
plt.suptitle('Orientation Estimates from Complementary Filter')
plt.show()
