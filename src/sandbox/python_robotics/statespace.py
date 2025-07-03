import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Define system parameters
m = 0.1  # Mass of pendulum (kg)
M = 1.0  # Mass of cart (kg)
L = 1.0  # Length of pendulum (m)
g = 9.81  # Gravitational acceleration (m/s²)
F = 0    # Force applied to cart (N)

# State-space representation
def system_dynamics(t, state):
    """
    State space representation of the inverted pendulum on a cart
    state = [x, theta, x_dot, theta_dot]
    where:
        x: cart position
        theta: pendulum angle (0 is upright position)
        x_dot: cart velocity
        theta_dot: pendulum angular velocity
    """
    x, theta, x_dot, theta_dot = state
    
    # Equations of motion (derived from Lagrangian mechanics)
    numerator = F + m*L*theta_dot**2*np.sin(theta) - m*g*np.sin(theta)*np.cos(theta)
    denominator = M + m*(1 - np.cos(theta)**2)
    x_ddot = numerator / denominator
    
    theta_ddot = (g*np.sin(theta) - x_ddot*np.cos(theta)) / L
    
    return [x_dot, theta_dot, x_ddot, theta_ddot]

# Simulation parameters
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)
initial_state = [0, np.pi/6, 0, 0]  # Start with a small angle

# Solve the system
solution = solve_ivp(system_dynamics, t_span, initial_state, t_eval=t_eval)

# Visualization
def create_animation():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid()
    
    cart_width, cart_height = 0.4, 0.2
    
    # Create objects for animation
    cart = plt.Rectangle((0, 0), cart_width, cart_height, fc='b', ec='k')
    pendulum, = ax.plot([], [], 'k-', lw=2)
    mass, = ax.plot([], [], 'ro', ms=10)
    time_text = ax.text(-2.8, 1.8, '', fontsize=12)
    
    def init():
        ax.add_patch(cart)
        return cart, pendulum, mass, time_text
    
    def animate(i):
        x = solution.y[0, i]
        theta = solution.y[1, i]
        
        # Update cart position
        cart.set_xy([x - cart_width/2, -cart_height/2])
        
        # Calculate pendulum endpoint
        x_pend = x + L * np.sin(theta)
        y_pend = L * np.cos(theta)
        
        pendulum.set_data([x, x_pend], [0, y_pend])
        mass.set_data([x_pend], [y_pend])
        time_text.set_text(f'Time: {solution.t[i]:.1f}s')
        
        return cart, pendulum, mass, time_text
    
    anim = FuncAnimation(fig, animate, frames=len(t_eval), 
                         init_func=init, blit=True, interval=20)
    
    plt.close()  # Prevent duplicate display in notebook
    return anim

# Create and display animation
animation = create_animation()
HTML(animation.to_jshtml())