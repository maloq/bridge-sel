Image Selection Project
Goal
Given a set of images of a bridge taken by a camera, I want to determine the minimum number of images needed to fully visualise/cover the bridge. This is because the images overlap in certain areas, making it necessary to select a subset that covers the entire object.
This problem can be compared to a type of “set cover problem”.

In addition to the images, we also have an .obj file that we created from the images to obtain a textured mesh.

However, we don’t want to use images taken from further away, as they do not allow us to see the details up close. The purpose of these images is to identify defects on the bridge, so it is necessary to view the elements up close. Therefore, we always want to prefer images that are closer to the bridge.
Terminology
Image → the picture taken by the camera’s drone
Pose → the position of the camera in the space when the picture is taken (position and orientation)
Camera →the camera used to take the pictures
Mesh → a surface with a triangular shape made of 3 vertices and a face
Material
Model
TexturedMesh.obj → the textured model of the bridge
Decimated-textured-mesh.obj → the same model but much lighter with fewer vertices and faces
Cameras.xml
ophikappa_poses.txt
RIO_CERVOS_DERECHA
Folder containing all the images taken by the drone

The textured model can be visualised in MeshLab software
Algorithm Definition
Input:
Images folder
File with info about the poses (ophikappa_poses.txt)
File with info about the cameras/sensors (cameras.xml)
Textured mesh (.obj file)
Output:
Subset of the selected images (the filenames of the images)
It would be useful to also have an output graph (a curve) showing the percentage of coverage achieved with a certain number of images.
For example: we achieve a 90% coverage with 1000 images, 95% with 2000 images, and 100% with all images. The coverage percentage could be an input parameter.

Feel free to add any other optional input parameters as needed.

Code Language: Python 3
Libraries to work with meshes: Trimesh (better) or eventually PyMesh or pymeshlab
Possible Strategy
The mesh is viewed as a point cloud composed of its vertices. A mesh is made up of vertices, edges and faces, but for this purpose, we focus on the vertices as points in space.

For each image, select the vertices that fall within the camera’s field of view. This means determining which points of the mesh are within the visible area of the camera when the image was taken.

For each selected vertex, check if it’s visible or occluded:
Ray tracing →draw a ray from the image’s centroid to each vertex.

Intersection check →determine if the nearest intersection of the ray with the mesh is at the selected vertex. If it is, the vertex is visible from the camera pose. If another intersection occurs closer to the camera, the vertex is occluded.

Repeat this process for every image to determine which vertices are visible from each camera position.

Construct a coverage map that indicates which vertices are seen by which images and from what distances.
Image selection →choose the image that covers the most vertices, giving preference to those taken from closer distances to ensure detailed visibility.

