import numpy as np
import matplotlib.pyplot as plt



def set_axes_equal(ax):
    """
    This function normalizes a 3d plot axis
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def rigid_motion_axis(g_SE3):
    # Initial vectors for the x, y, z axes (unit vectors)
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    # decompose rigid motion 
    R = g_SE3[0:3,0:3]
    origin = g_SE3[0:3,3]
    
    # rotate the axis
    x_axis_transform = R@x_axis
    y_axis_transform = R@y_axis
    z_axis_transform = R@z_axis

    # check that the axis is still orthogonal 
    return x_axis_transform, y_axis_transform, z_axis_transform, origin

# x_axis_transform, y_axis_transform, z_axis_transform, origin = rigid_motion_axis(g_SE[4,:,:])


def visualize_frame(g_SE,mcV=None,mcE=None):
    if mcV is None:
        mcV = np.zeros((6,1))

    if mcE is None:
        mcE = np.zeros((6,1))

    # Set up the figure and 3D axes
    fig = plt.figure(size = (16,8))
    ax = fig.add_subplot(111, projection='3d')

    x_axis, y_axis, z_axis, origin_i = rigid_motion_axis(g_SE)


    # Plot the quivers for each axis at each origin
    ax.quiver(*origin_i, *x_axis, color='r', arrow_length_ratio=0.01, label='X' )
    ax.quiver(*origin_i, *y_axis, color='g', arrow_length_ratio=0.01, label='Y' )
    ax.quiver(*origin_i, *z_axis, color='b', arrow_length_ratio=0.01, label='Z' )

     # plot the velocity field
    Omega = mcV[0:3]
    V = mcV[3:6]

    # -- linear velocity field 
    ax.quiver(*origin_i, *V, color='purple', arrow_length_ratio=0.1, label='V' )

    # -- angular velocity field 
    ax.quiver(*origin_i, *Omega, color='orange', arrow_length_ratio=0.1, label='Omega' )

    # plot the velocity field
    Kappa = mcE[0:3]
    Gamma = mcE[3:6]
    
    # -- linear strain field 
    ax.quiver(*origin_i, *Gamma, color='cyan', arrow_length_ratio=0.15, label='Gamma')
    
    # -- angular strain field 
    ax.quiver(*origin_i, *Kappa, color='blue', arrow_length_ratio=0.15, label='Kappa' )
    

    # Set axis limits
    ax.set_xlim([-1, 3])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])

    # Labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    # Show the plot
    plt.show()


def visualize_rod(g_SE,eta=None,xi=None):
    if mcV is None:
        # print(len(g_SE[:,0,0]))
        mcV = np.zeros((len(g_SE[:,0,0]),6))
        # Set up the figure and 3D axes
    if mcE is None:
        mcE = np.zeros((len(g_SE[:,0,0]),6))
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the origin
    origin = np.array([0, 0, 0])

    # Number of transformations (assuming the third dimension represents different transformations)
    num_transforms = len(g_SE[:,0,0])

    # Lists to store origins and axis transforms
    origins = []
    x_axes, y_axes, z_axes = [], [], []

    # Loop through each transformation
    for i in range(num_transforms):
        x_axis, y_axis, z_axis, origin_i = rigid_motion_axis(g_SE[i,:,:])
        origins.append(origin_i)
        x_axes.append(x_axis)
        y_axes.append(y_axis)
        z_axes.append(z_axis)

        # Plot the quivers for each axis at each origin
        ax.quiver(*origin_i, *x_axis, color='r', arrow_length_ratio=0.01, label='X' if i == 0 else "")
        ax.quiver(*origin_i, *y_axis, color='g', arrow_length_ratio=0.01, label='Y' if i == 0 else "")
        ax.quiver(*origin_i, *z_axis, color='b', arrow_length_ratio=0.01, label='Z' if i == 0 else "")
        
        # plot the velocity field
        Omega = mcV[0:3]
        V = mcV[3:6]

        
        # -- linear velocity field 
        ax.quiver(*origin_i, *V, color='purple', arrow_length_ratio=0.1, label='V' if i == 0 else "")

        # -- angular velocity field 
        ax.quiver(*origin_i, *Omega, color='orange', arrow_length_ratio=0.1, label='Omega' if i == 0 else "")

        # plot the velocity field
        Kappa = mcE[0:3]
        Gamma = mcE[3:6]
        
        # -- linear strain field 
        ax.quiver(*origin_i, *Gamma, color='cyan', arrow_length_ratio=0.15, label='Gamma' if i == 0 else "")
        
        # -- angular strain field 
        ax.quiver(*origin_i, *Kappa, color='pink', arrow_length_ratio=0.15, label='Kappa' if i == 0 else "")
        
    print('purple = Linear Velocity')    
    print('orange = Rotational Velocity')
    print('cyan = Linear strain')    
    print('pink = Rotational strain')
    # Convert origins list to numpy array for easier manipulation (optional)
    origins = np.array(origins)

    # Extract x, y, z coordinates of the origins for further use (optional)
    origins_x = origins[:, 0]
    origins_y = origins[:, 1]
    origins_z = origins[:, 2]

    # Set axis limits
    ax.set_xlim([-1, 3])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])

    # Labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)
    # plot the backbone of the rod 
    ax.plot(origins_x, origins_y, origins_z, marker='o', label='Line connecting points')
    # Show the plot
    plt.show()