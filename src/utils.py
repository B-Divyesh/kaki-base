import ezdxf
from ezdxf.addons import odafc
import trimesh
import streamlit as st

def validate_dxf(file_path):
    """Validate DXF file before processing"""
    try:
        auditor = odafc.Auditor(file_path)
        auditor.audit()
        return not auditor.has_errors
    except:
        # If ODA File Converter is not available, try basic validation
        try:
            doc = ezdxf.readfile(file_path)
            return True
        except:
            return False

def validate_point(point):
    """Validate and convert point coordinates"""
    try:
        if len(point) >= 2:
            return [float(point[0]), float(point[1])]
        return None
    except (TypeError, ValueError):
        return None

def load_mesh_file(file_path):
    """
    Load 3D mesh file using trimesh with support for multiple formats
    Supports: GLTF, STL, OBJ, PLY
    """
    try:
        # Attempt to load the mesh
        mesh = trimesh.load(str(file_path), force='mesh')
        
        # Validate mesh
        if not isinstance(mesh, trimesh.Trimesh):
            # If loaded as a scene, extract the first mesh
            if hasattr(mesh, 'geometry'):
                mesh = list(mesh.geometry.values())[0]
        
        # Ensure mesh is a valid trimesh object
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Unable to convert to valid mesh")
        
        # Normalize mesh if needed (center and scale)
        mesh.apply_translation(-mesh.centroid)
        
        # Safely scale mesh
        try:
            # Use max absolute coordinate to determine scale
            max_coord = np.max(np.abs(mesh.vertices))
            if max_coord > 0:
                scale = 1.0 / max_coord
                mesh.apply_scale(scale)
        except Exception as scale_error:
            st.warning(f"Could not scale mesh: {scale_error}")
        
        return mesh
    
    except Exception as e:
        st.error(f"Error loading 3D model: {str(e)}")
        return None