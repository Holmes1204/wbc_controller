import numpy as np
import matplotlib.pyplot as plt

class EKF:
    def __init__(self, x, P, F, Q, H, R):
        self.x = x  # initial state estimate
        self.P = P  # initial state covariance matrix
        self.F = F  # state transition matrix
        self.Q = Q  # process noise covariance matrix
        self.H = H  # measurement matrix
        self.R = R  # measurement noise covariance matrix
    
    def predict(self):
        # predict state estimate and covariance matrix
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, z):
        # update state estimate and covariance matrix based on measurement z
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(self.P.shape[0]) - np.dot(K, self.H), self.P)



# Define system parameters
dt = 0.001  # time step 1kHz
F = np.array([[1, dt], [0, 1]])  # state transition matrix
H = np.array([[1, 0], [0, 1]])  # measurement matrix
Q = np.diag([0.1, 0.1])  # process noise covariance matrix
R = np.diag([1.0, 1.0])  # measurement noise covariance matrix

# Initialize EKF
x0 = np.array([0, 0])  # initial state estimate
P0 = np.diag([1.0, 1.0])  # initial state covariance matrix
ekf = EKF(x0, P0, F, Q, H, R)

# Generate true trajectory
t = np.arange(0, 10, dt)
pos = 10 * np.sin(t)  # true position
vel = 10 * np.cos(t)  # true velocity
true_states = np.vstack((pos, vel))

# Simulate measurements
meas_noise = np.random.multivariate_normal([0, 0], R, len(t)).T
meas = np.dot(H, true_states) + meas_noise

# Run EKF
est_states = np.zeros_like(true_states)
est_covs = np.zeros((2, 2, len(t)))
for i in range(len(t)):
    ekf.predict()
    ekf.update(meas[:, i])
    est_states[:, i] = ekf.x
    est_covs[:, :, i] = ekf.P

# Plot results
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(t,meas[0],label='measure')
ax[0].plot(t, est_states[0], label='estimate')
ax[0].plot(t, true_states[0], label='true',color='r')
ax[0].set_xlabel('time')
ax[0].set_ylabel('position')
ax[0].legend()
ax[0].grid()
ax[1].plot(t,meas[1],label='measure')
ax[1].plot(t, est_states[1], label='estimate')
ax[1].plot(t, true_states[1], label='true',color='r')
ax[1].set_ylabel('velocity')
ax[1].legend()
ax[1].grid()
plt.show()
