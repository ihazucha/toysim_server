import numpy as np

from pyqtgraph.opengl.MeshData import MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem

from modules.ui.presets import MColors


class CubeMesh(MeshData):
    VERTEXES = np.array(
        [
            [1, 1, 1],  # top front right
            [1, 1, 0],  # bottom front right
            [1, 0, 1],  # top front left
            [1, 0, 0],  # bottom front left
            [0, 1, 1],  # top back right
            [0, 1, 0],  # bottom back right
            [0, 0, 1],  # top back left
            [0, 0, 0],  # bottom back left
        ]
    )
    FACES = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # front
            [4, 6, 5],
            [5, 6, 7],  # back
            [0, 2, 4],
            [2, 6, 4],  # top
            [1, 5, 3],
            [3, 5, 7],  # bottom
            [0, 4, 1],
            [1, 4, 5],  # right
            [2, 3, 6],
            [3, 7, 6],  # left
        ]
    )

    def __init__(self, size: float = 1, *args, **kwargs):
        super().__init__(vertexes=self.VERTEXES * size, faces=self.FACES, *args, **kwargs)


class OpaqueCube(GLMeshItem):
    def __init__(self, parentItem=None, size: float = 0, color=MColors.WHITE, *args, **kwargs):
        self.size = size
        super().__init__(
            parentItem,
            meshdata=CubeMesh(size=size),
            color=color,
            shader="shaded",
            smooth=False,
            glOptions="opaque",
            *args,
            **kwargs,
        )


class CylinderMesh(MeshData):
    """Create a 3D cylinder mesh with customizable parameters."""
    
    def __init__(self, radius=1.0, height=1.0, segments=32, *args, **kwargs):
        """
        Create a cylinder mesh.
        
        Args:
            radius: Radius of the cylinder
            height: Height of the cylinder
            segments: Number of segments (higher = smoother)
        """
        # Generate vertices and faces
        vertices, faces = self._create_cylinder(radius, height, segments)
        super().__init__(vertexes=vertices, faces=faces, *args, **kwargs)
    
    def _create_cylinder(self, radius, height, segments):
        """Generate vertices and faces for a cylinder."""
        vertices = []
        
        # Create top and bottom center vertices
        bottom_center = [0, 0, 0]
        top_center = [0, 0, height]
        vertices.append(bottom_center)  # index 0
        vertices.append(top_center)     # index 1
        
        # Create circle vertices for top and bottom
        for i in range(segments):
            # Calculate angle in radians
            angle = 2 * np.pi * i / segments
            
            # Calculate x and y coordinates on unit circle
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Add bottom and top circle vertices
            vertices.append([x, y, 0])         # bottom rim
            vertices.append([x, y, height])    # top rim
        
        # Convert to numpy array
        vertices = np.array(vertices)
        
        # Create faces
        faces = []
        
        # Bottom faces (connect bottom center to bottom rim)
        for i in range(segments):
            # Get the indices of the rim vertices
            current = 2 + i * 2  # index of current bottom vertex
            next_idx = 2 + ((i + 1) % segments) * 2  # index of next bottom vertex
            
            # Add face (counter-clockwise for correct normals)
            faces.append([0, next_idx, current])
        
        # Top faces (connect top center to top rim)
        for i in range(segments):
            # Get the indices of the rim vertices
            current = 3 + i * 2  # index of current top vertex
            next_idx = 3 + ((i + 1) % segments) * 2  # index of next top vertex
            
            # Add face (clockwise for correct normals)
            faces.append([1, current, next_idx])
        
        # Side faces (connect top and bottom rims)
        for i in range(segments):
            # Get current quad corners
            bottom_current = 2 + i * 2
            bottom_next = 2 + ((i + 1) % segments) * 2
            top_current = bottom_current + 1
            top_next = bottom_next + 1
            
            # Add two triangular faces for the quad
            faces.append([bottom_current, bottom_next, top_current])
            faces.append([bottom_next, top_next, top_current])
        
        # Convert to numpy array
        faces = np.array(faces)
        
        return vertices, faces


class OpaqueCylinder(GLMeshItem):
    """A solid cylinder mesh item with shading."""
    
    def __init__(self, 
                 parentItem=None, 
                 radius=1.0, 
                 height=1.0, 
                 segments=32, 
                 color=MColors.WHITE, 
                 *args, **kwargs):
        
        self.radius = radius
        self.height = height
        self.segments = segments
        
        super().__init__(
            parentItem=parentItem,
            meshdata=CylinderMesh(radius=radius, height=height, segments=segments),
            color=color,
            drawEdges=True,
            drawFaces=False,
            # shader="shaded",
            smooth=True,  # True for smoother shading on curved surface
            # glOptions="opaque",
            *args,
            **kwargs
        )