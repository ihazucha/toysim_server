import numpy as np
from typing import Tuple

from pyqtgraph import Vector
from pyqtgraph.opengl.MeshData import MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader

from modules.ui.presets import MColors

customWorldLight = ShaderProgram('customWorldLight', [
    VertexShader("""
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """
    ),
    FragmentShader("""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            float p = dot(v_normal, normalize(vec3(1.0, 1.0, 1.0)));
            p = p < 0. ? 0. : p * 0.8;
            vec3 rgb = v_color.rgb * (0.2 + p);
            gl_FragColor = vec4(rgb, v_color.a);
        }
    """
    ),
])



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
    
    def _create_cylinder(self, radius, height, segments) -> Tuple[np.ndarray, np.ndarray]:
        """Creates Vertices and Faces"""
        vertices = []
        
        bottom_center = [0, 0, 0]
        top_center = [0, 0, height]
        vertices.append(bottom_center)
        vertices.append(top_center)
        
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Add bottom and top circle rim vertices
            vertices.append([x, y, 0])
            vertices.append([x, y, height])
        
        vertices = np.array(vertices)
        faces = []
        
        # Bottom and Top faces (center -> rim)
        for i in range(segments):
            vertex = 2 + i * 2
            next_vertex = 2 + ((i + 1) % segments) * 2
            faces.append([0, next_vertex, vertex])
                
        for i in range(segments):
            vertex = 3 + i * 2 
            next_vertex = 3 + ((i + 1) % segments) * 2
            faces.append([1, vertex, next_vertex])
        
        # Side faces (top -> bottom rims)
        for i in range(segments):
            bottom_current = 2 + i * 2
            bottom_next = 2 + ((i + 1) % segments) * 2
            top_current = bottom_current + 1
            top_next = bottom_next + 1
            faces.append([bottom_current, bottom_next, top_current])
            faces.append([bottom_next, top_next, top_current])
        
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
                 drawEdges=True,
                 drawFaces=False,
                 shader=customWorldLight,
                 *args, **kwargs):
        
        self.radius = radius
        self.height = height
        self.segments = segments
        
        super().__init__(
            parentItem=parentItem,
            meshdata=CylinderMesh(radius=radius, height=height, segments=segments),
            color=color,
            edgeColor=color,
            drawEdges=drawEdges,
            drawFaces=drawFaces,
            shader=shader,
            smooth=True,
            glOptions="opaque",
            *args,
            **kwargs
        )

    def get_position(self):
        return Vector(self.transform().matrix()[:3, 3])
    
    def get_transform(self):
        return self.transform()