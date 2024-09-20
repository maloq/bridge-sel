import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go



def visualize_results_2d(mesh, visible_vertices, camera_position, R):
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

    # Plot camera orientation
    for i in range(3):
        direction = R[:, i]
        ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                  direction[0], direction[1], direction[2],
                  length=0.5, color=['r', 'g', 'b'][i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mesh Visibility')
    ax.legend()

    ax.set_box_aspect((np.ptp(mesh.vertices[:, 0]),
                       np.ptp(mesh.vertices[:, 1]),
                       np.ptp(mesh.vertices[:, 2])))

    plt.show()



def visualize_results(mesh, visible_vertices, camera_position, R):
    """
    Create an interactive 3D visualization of the mesh, visible vertices, and camera position.
    
    :param mesh: trimesh.Trimesh object
    :param visible_vertices: List of indices of visible vertices
    :param camera_position: 3D position of the camera
    :param R: 3x3 rotation matrix of the camera
    """
    if len(camera_position.shape)>1:
        camera_position = camera_position.squeeze()

    # Create scatter plot for all vertices
    all_vertices = go.Scatter3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        mode='markers',
        marker=dict(size=2, color='grey', opacity=0.5),
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

    data = [all_vertices, visible_scatter, camera_scatter] + line_traces

    max_size_x = abs(mesh.bounds[0][0] - mesh.bounds[1][0])
    max_size_y = abs(mesh.bounds[0][1] - mesh.bounds[1][1])

    offset_x = int(20 + max_size_x/2)
    offset_y = int(20 + max_size_y/2)

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis_range = [-1 * offset_x, offset_x],
            yaxis_range = [-1 * offset_y, offset_y],
            aspectmode='data'
        ),
        title='Mesh Visibility',
        width=1000,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def visualize_projected(vertices_image, image_size):
    plt.figure(figsize=(10, 10))
    plt.scatter(vertices_image[:, 0], vertices_image[:, 1], c='blue', s=1, alpha=0.5, label='All vertices')
    plt.xlim(0, image_size[1])
    plt.ylim(image_size[0], 0)
    plt.legend()
    plt.title('Projected Vertices')
    plt.show()


def visualize_results_rays(mesh, visible_vertices, R, t, rays_origins, rays_directions):


    # Create scatter plot for all vertices
    camera_position = t.squeeze()

    all_vertices = go.Scatter3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        mode='markers',
        marker=dict(size=2, color='grey'),
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

    # Visualize rays
    ray_traces = []
    for i in range(len(rays_origins)):
        start_point = rays_origins[i]
        end_point = start_point + rays_directions[i] * 15
        ray_traces.append(go.Scatter3d(
            x=[start_point[0], end_point[0]],
            y=[start_point[1], end_point[1]],
            z=[start_point[2], end_point[2]],
            mode='lines',
            line=dict(color='yellow', width=1),
            opacity=0.3,
            name=f'Ray_{i}' 
        ))

    data = [all_vertices, visible_scatter, camera_scatter] + line_traces + ray_traces

    max_size_x = abs(mesh.bounds[0][0] - mesh.bounds[1][0])
    max_size_y = abs(mesh.bounds[0][1] - mesh.bounds[1][1])

    offset_x = int(20 + max_size_x/2)
    offset_y = int(20 + max_size_y/2)

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis_range = [-1 * offset_x, offset_x],
            yaxis_range = [-1 * offset_y, offset_y],
            aspectmode='data'
        ),
        title='Mesh Visibility',
        width=1000,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()



def visualize_results_mesh(mesh, visible_vertices, R, t, rays_origins=None, rays_directions=None):
    """
    Create an interactive 3D visualization of the mesh (with faces), visible vertices, camera position, and rays.
    
    :param mesh: trimesh.Trimesh object
    :param visible_vertices: List of indices of visible vertices
    :param R: 3x3 rotation matrix of the camera
    :param t: 3x1 translation vector of the camera
    :param rays_origins: Nx3 array of ray origin points
    :param rays_directions: Nx3 array of ray direction vectors
    """
    camera_position = t.squeeze()

    # Create mesh with faces
    mesh_plot = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        opacity=0.5,
        color='lightblue',
        name='Mesh'
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

    # Visualize rays
    if rays_origins:
        ray_traces = []
        for i in range(len(rays_origins)):
            start_point = rays_origins[i]
            end_point = start_point + rays_directions[i] * 15
            ray_traces.append(go.Scatter3d(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                z=[start_point[2], end_point[2]],
                mode='lines',
                line=dict(color='yellow', width=1),
                opacity=0.3,
                name=f'Ray_{i}' 
            ))

        data = [mesh_plot, visible_scatter, camera_scatter] + line_traces + ray_traces
    else:
        data = [mesh_plot, visible_scatter, camera_scatter] + line_traces

    max_size_x = abs(mesh.bounds[0][0] - mesh.bounds[1][0])
    max_size_y = abs(mesh.bounds[0][1] - mesh.bounds[1][1])

    offset_x = int(20 + max_size_x/2)
    offset_y = int(20 + max_size_y/2)

    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis_range = [-1 * offset_x, offset_x],
            yaxis_range = [-1 * offset_y, offset_y],
            aspectmode='data'
        ),
        title='Mesh Visibility',
        width=1000,
        margin=dict(r=20, l=10, b=10, t=10)
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()




