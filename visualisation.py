import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def visualize_results(mesh, visible_vertices, camera_position, R):
    """
    Visualize the mesh, visible vertices, and camera position in 3D.
    
    :param mesh: trimesh.Trimesh object
    :param visible_vertices: List of indices of visible vertices
    :param camera_position: 3D position of the camera
    :param R: 3x3 rotation matrix of the camera
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all mesh vertices
    ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
               c='lightgray', s=1, alpha=0.5, label='All vertices')

    # Plot visible vertices
    visible_points = mesh.vertices[visible_vertices]
    ax.scatter(visible_points[:, 0], visible_points[:, 1], visible_points[:, 2], 
               c='red', s=20, label='Visible vertices')

    # Plot camera position
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], 
               c='blue', s=100, label='Camera')

    # Plot camera orientation (using the rotation matrix)
    for i in range(3):
        direction = R[:, i]
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                  direction[0], direction[1], direction[2],
                  length=0.5, color=['r', 'g', 'b'][i])

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Visibility Visualization')
    ax.legend()

    # Set aspect ratio to be equal
    ax.set_box_aspect((np.ptp(mesh.vertices[:, 0]),
                       np.ptp(mesh.vertices[:, 1]),
                       np.ptp(mesh.vertices[:, 2])))

    plt.show()


def visualize_results_interactive(mesh, visible_vertices, camera_position, R, t):
    """
    Create an interactive 3D visualization of the mesh, visible vertices, and camera position.
    
    :param mesh: trimesh.Trimesh object
    :param visible_vertices: List of indices of visible vertices
    :param camera_position: 3D position of the camera
    :param R: 3x3 rotation matrix of the camera
    """
    # Create scatter plot for all vertices
    all_vertices = go.Scatter3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.5),
        name='All vertices'
    )

    # Create scatter plot for visible vertices
    visible_points = mesh.vertices[visible_vertices]
    visible_scatter = go.Scatter3d(
        x=visible_points[:, 0], y=visible_points[:, 1], z=visible_points[:, 2],
        mode='markers',
        marker=dict(size=4, color='red'),
        name='Visible vertices'
    )

    # Camera position
    camera_scatter = go.Scatter3d(
        x=[camera_position[0]], y=[camera_position[1]], z=[camera_position[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Camera'
    )

    # Create lines for camera orientation
    line_traces = []
    colors = ['red', 'green', 'blue']
    for i in range(3):
        direction = R[:, i]
        end_point = camera_position + direction * 2.5
        line_traces.append(go.Scatter3d(
            x=[camera_position[0], end_point[0]],
            y=[camera_position[1], end_point[1]],
            z=[camera_position[2], end_point[2]],
            mode='lines',
            line=dict(color=colors[i], width=4),
            name=f'Camera orientation {["X", "Y", "Z"][i]}'
        ))

    # Combine all traces
    data = [all_vertices, visible_scatter, camera_scatter] + line_traces

    max_size_x = abs(mesh.bounds[0][0] - mesh.bounds[1][0])
    max_size_y = abs(mesh.bounds[0][1] - mesh.bounds[1][1])
    max_size_z = abs(mesh.bounds[0][2] - mesh.bounds[1][2])
    
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        xaxis=dict(range=[t[0]-max_size_x/2, t[0]+max_size_x/2]),
        yaxis=dict(range=[t[1]-max_size_y/2, t[1]+max_size_y/2]),
        title='Interactive Mesh Visibility Visualization',
        width=1000,
        height=800
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()