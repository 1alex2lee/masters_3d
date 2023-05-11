import os
import trimesh
import numpy as np


class SurfaceSampler:
    def __init__(self, mesh):
        self.mesh = mesh
        self.correct_faces = {i: 1 for i in range(len(mesh.faces))} # changed
        self.correct_points = np.array([])
        self.pointPositionVectors = np.array([])
        self.connectedFaceIndices = np.array([])
        self.normals_for_SurfacePoints = np.array([])

    def sample_SurfacePoints(self, nObjectPoints=10000):
        # sample points randomly on surface of mesh
        points = trimesh.sample.sample_surface_even(
            self.mesh, count=nObjectPoints)
        self.pointPositionVectors = points[0]
        self.connectedFaceIndices = points[1]

        correct_points = []
        normals_for_SurfacePoints = []
        # self.correct_faces = {i: 1 for i in range(len(mesh.faces))}
        for i, point in enumerate(self.pointPositionVectors):
            if self.correct_faces[self.connectedFaceIndices[i]] == 1:
                correct_points += [point]
                normals_for_SurfacePoints += [self.mesh.face_normals[self.connectedFaceIndices[i]]]

        self.correct_points = np.array(correct_points)
        self.normals_for_SurfacePoints = np.array(normals_for_SurfacePoints)

        print(self.correct_points.shape)

        return self.correct_points, self.normals_for_SurfacePoints

    def uniformlySampleUnitCube(self, nCubePoints=3000):
        X, Y, Z = np.mgrid[0:1:complex(np.cbrt(nCubePoints)), 0:1:complex(
            np.cbrt(nCubePoints)), -0.5:0.5:complex(np.cbrt(nCubePoints))]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        return positions

    def sample_offSurfacePoints(self, alpha1=0.2, alpha2=0.08, nLowNoisePoints=3000, nHighNoisePoints=3000, nVolumePoints=3000):
        basePoints, _ = self.sample_SurfacePoints(nHighNoisePoints)
        noise1 = np.random.normal(
            0, alpha1**2, (basePoints.shape[0], 3))  # changed
        noisy_points1 = basePoints + noise1

        basePoints, _ = self.sample_SurfacePoints(nLowNoisePoints)
        noise2 = np.random.normal(
            0, alpha2**2, (basePoints.shape[0], 3))  # changed
        noisy_points2 = basePoints + noise2

        cubePoints = self.uniformlySampleUnitCube(nCubePoints=nVolumePoints)

        allOffSurfacePoints = np.concatenate(
            (noisy_points1, noisy_points2, cubePoints), axis=0)

        return allOffSurfacePoints  # , noisy_points1, noisy_points2, cubePoints


def generate(file, window=None):
    # load mesh (STL CAD File)
    mesh = trimesh.load_mesh(file, force='mesh')
    # fix mesh if normals are not pointing "up"
    trimesh.repair.fix_inversion(mesh)

    # translate and scale the mesh. Note mesh origin is (0,0,0)
    meshNodes = np.array(mesh.vertices)
    # translate
    meshNodes[:, 0] = meshNodes[:, 0] - meshNodes[:, 0].min()
    meshNodes[:, 1] = meshNodes[:, 1] - meshNodes[:, 1].min()
    meshNodes[:, 2] = meshNodes[:, 2] - meshNodes[:, 2].max()
    # offset Z dimention to be exactly in the centre of a unit cube
    meshNodes[:, 2] = meshNodes[:, 2] + (meshNodes[:, 2].max() - meshNodes[:, 2].min()) / 2

    # scale
    scalingFactor = meshNodes[:, 0].max(
    ) - meshNodes[:, 0].min()  # X side length
    meshNodes = meshNodes/scalingFactor

    mesh.vertices = meshNodes

    nObjectPoints = 9000

    # calculate SDF points: xyz positions and SDF values at those positions
    a = SurfaceSampler(mesh)
    points, normals = a.sample_SurfacePoints(nObjectPoints)
    offSurfacePoints = a.sample_offSurfacePoints(
        nLowNoisePoints=3000, nHighNoisePoints=3000, nVolumePoints=3000)

    # delete points that are outside of unit cube
    pointsOutOfBounds = np.where((offSurfacePoints[:, 0] < 0) | (offSurfacePoints[:, 0] > 1) | (offSurfacePoints[:, 1] < 0) | (
        offSurfacePoints[:, 1] > 1) | (offSurfacePoints[:, 2] < -0.5) | (offSurfacePoints[:, 2] > 0.5))[0]
    offSurfacePoints = np.delete(offSurfacePoints, pointsOutOfBounds, axis=0)

    pointsArray = np.array(points)
    normalsArray = np.array(normals)
    offSurfacePointsArray = np.array(offSurfacePoints)

    export_path = os.path.join("temp", "optimisation", "CADandMesh", "SDF")
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    np.save(os.path.join(export_path, "surface_points.npy"), pointsArray)
    np.save(os.path.join(export_path, "normals.npy"), normalsArray)
    np.save(os.path.join(export_path, "offsurface_points.npy"), offSurfacePointsArray)

    print("-> Points, Normals, and Off-Surface Points generated for ", file)

    return pointsArray, normalsArray, offSurfacePointsArray
