import streamlit as st
import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import KDTree
import ezdxf
from .utils import validate_dxf, validate_point, load_mesh_file
from sklearn.neighbors import KernelDensity

class BuildingSimulator:
    def __init__(self):
        self.mesh = None
        self.graph = nx.Graph()
        self.exits = []
        self.people = []
        self.walkable_points = None
        self.heat_map_data = None

    def load_file(self, file_path):
        """Load 3D model file with support for multiple formats"""
        try:
            # Attempt to load mesh using trimesh
            self.mesh = load_mesh_file(file_path)
            
            if self.mesh is None:
                raise ValueError("Unable to load 3D model file")
            
            # Extract walls and potential exits
            self._extract_building_features()
            
            st.success(f"Successfully loaded 3D model")
            return self.mesh
            
        except Exception as e:
            st.error(f"Error loading 3D model file: {str(e)}")
            return None

    def _extract_building_features(self):
        """Extract potential walls and exits from mesh"""
        try:
            # If mesh is too complex, sample points for exits and navigation
            bounds = self.mesh.bounds
            
            # Generate potential exit points around the perimeter
            exit_candidates = [
                [bounds[0, 0], (bounds[0, 1] + bounds[1, 1])/2, bounds[0, 2]],  # Left side
                [bounds[1, 0], (bounds[0, 1] + bounds[1, 1])/2, bounds[0, 2]],  # Right side
                [(bounds[0, 0] + bounds[1, 0])/2, bounds[0, 1], bounds[0, 2]],  # Front side
                [(bounds[0, 0] + bounds[1, 0])/2, bounds[1, 1], bounds[0, 2]]   # Back side
            ]
            
            # Alternative method to check point containment
            def is_point_inside_mesh(point):
                try:
                    # Use ray-based point containment check
                    # Cast rays from the point in multiple directions
                    rays = [
                        [point, [1, 0, 0]],
                        [point, [-1, 0, 0]],
                        [point, [0, 1, 0]],
                        [point, [0, -1, 0]],
                        [point, [0, 0, 1]],
                        [point, [0, 0, -1]]
                    ]
                    
                    # Count intersections
                    intersections = sum(1 for ray in rays if self.mesh.ray.intersects_any([ray]))
                    
                    # If odd number of intersections, point is inside
                    return intersections % 2 == 1
                except Exception:
                    # Fallback to a simpler method if ray intersection fails
                    return False
            
            # Filter exit points 
            self.exits = [
                {'location': point[:2].tolist()}
                for point in exit_candidates
                if not is_point_inside_mesh(point)
            ]
            
            st.info(f"Detected {len(self.exits)} potential exit points")
            
        except Exception as e:
            st.warning(f"Could not automatically extract building features: {str(e)}")
            # Fallback to manual exit points
            self.exits = []
        
    def load_cad(self, file_path):
        """Original DXF loading method, kept for backwards compatibility"""
        try:
            if not validate_dxf(file_path):
                raise ValueError("Invalid or corrupted DXF file")

            doc = ezdxf.readfile(file_path)
            modelspace = doc.modelspace()
            
            walls = []
            doors = []
            
            for entity in modelspace:
                try:
                    if entity.dxftype() == 'LINE':
                        start = validate_point(entity.dxf.start)
                        end = validate_point(entity.dxf.end)
                        
                        if start and end:
                            walls.append({
                                'start': start,
                                'end': end
                            })
                    
                    elif entity.dxftype() == 'INSERT':
                        if any(door_text in entity.dxf.name.upper() 
                              for door_text in ['DOOR', 'D-', 'DR-']):
                            location = validate_point(entity.dxf.insert)
                            if location:
                                doors.append({
                                    'location': location
                                })
                
                except Exception as e:
                    st.warning(f"Skipping invalid entity: {str(e)}")
                    continue
            
            if not walls:
                raise ValueError("No valid walls found in the CAD file")
            
            st.success(f"Successfully loaded {len(walls)} walls and {len(doors)} doors")
            self.mesh = self._create_3d_model(walls, doors)
            return self.mesh
            
        except ezdxf.DXFStructureError:
            st.error("Invalid or corrupted DXF file structure")
            return None
        except Exception as e:
            st.error(f"Error loading CAD file: {str(e)}")
            return None

    def _validate_dxf(self, file_path):
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

    def _validate_point(self, point):
        """Validate and convert point coordinates"""
        try:
            # Ensure point has at least 2 coordinates
            if len(point) >= 2:
                # Convert to float and take first two coordinates
                return [float(point[0]), float(point[1])]
            return None
        except (TypeError, ValueError):
            return None
    
    def _create_3d_model(self, walls, doors):
        """Convert 2D CAD data to 3D model with validation"""
        try:
            vertices = []
            faces = []
            height = 3.0  # Standard wall height in meters
            
            for wall in walls:
                # Validate wall coordinates
                if not all(isinstance(coord, (int, float)) 
                          for point in [wall['start'], wall['end']] 
                          for coord in point):
                    continue
                
                # Create vertical walls
                v_start = len(vertices)
                vertices.extend([
                    [wall['start'][0], wall['start'][1], 0],
                    [wall['start'][0], wall['start'][1], height],
                    [wall['end'][0], wall['end'][1], height],
                    [wall['end'][0], wall['end'][1], 0]
                ])
                
                faces.extend([
                    [v_start, v_start+1, v_start+2],
                    [v_start, v_start+2, v_start+3]
                ])
            
            if not vertices or not faces:
                raise ValueError("No valid geometry created from walls")
            
            # Create mesh using trimesh with error handling
            try:
                mesh = trimesh.Trimesh(vertices=np.array(vertices),
                                     faces=np.array(faces))
                
                # Verify mesh validity
                if not mesh.is_watertight:
                    st.warning("Generated mesh is not watertight - simulation may be affected")
                
                # Store door locations for navigation
                self.exits = doors
                
                return mesh
                
            except Exception as e:
                raise ValueError(f"Error creating 3D mesh: {str(e)}")
            
        except Exception as e:
            st.error(f"Error creating 3D model: {str(e)}")
            return None

    def setup_navigation(self):
        """Create navigation graph for people movement"""
        if not self.mesh:
            st.error("No mesh loaded - please load a CAD file first")
            return
        
        try:
            # Alternative method to check point containment
            def is_point_inside_mesh(point):
                try:
                    # Use ray-based point containment check
                    rays = [
                        [point, [1, 0, 0]],
                        [point, [-1, 0, 0]],
                        [point, [0, 1, 0]],
                        [point, [0, -1, 0]],
                        [point, [0, 0, 1]],
                        [point, [0, 0, -1]]
                    ]
                    
                    # Count intersections
                    intersections = sum(1 for ray in rays if self.mesh.ray.intersects_any([ray]))
                    
                    # If odd number of intersections, point is inside
                    return intersections % 2 == 1
                except Exception:
                    # Fallback to a simpler method if ray intersection fails
                    return False

            # Create grid of walkable points
            bounds = self.mesh.bounds
            grid_spacing = 0.5  # meters
            
            x = np.arange(bounds[0, 0], bounds[1, 0], grid_spacing)
            y = np.arange(bounds[0, 1], bounds[1, 1], grid_spacing)
            
            points = []
            for xi in x:
                for yi in y:
                    point = np.array([xi, yi, 0.1])  # Slightly above ground
                    # Check if point is NOT inside the mesh (meaning it's in walkable space)
                    if not is_point_inside_mesh(point):
                        points.append(point)
            
            if not points:
                raise ValueError("No walkable points generated")
            
            # Clear existing graph and create new one
            self.graph.clear()
            
            # Create navigation graph
            self.walkable_points = np.array(points)
            tree = KDTree(self.walkable_points[:, :2])  # Use only x,y coordinates for distance calculation
            
            # Add nodes first
            for i in range(len(points)):
                self.graph.add_node(i)
            
            # Then add edges
            for i, point in enumerate(points):
                # Find nearby points within radius
                indices = tree.query_ball_point(point[:2], grid_spacing * 1.5)
                for j in indices:
                    if i != j:
                        # Check if path between points intersects with walls
                        try:
                            if not self.mesh.ray.intersects_any([[point, points[j]]]):
                                self.graph.add_edge(i, j)
                        except Exception:
                            # Fallback edge addition if ray intersection fails
                            self.graph.add_edge(i, j)
            
            # Verify graph is properly initialized
            if not self.graph.number_of_nodes():
                raise ValueError("Graph has no nodes")
                
            if not nx.is_connected(self.graph):
                # Find largest connected component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                self.graph = self.graph.subgraph(largest_cc).copy()
                st.warning("Using largest connected component for navigation")
                
            st.success(f"Navigation graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
                
        except Exception as e:
            st.error(f"Error setting up navigation: {str(e)}")
            self.graph.clear()  # Reset graph on error

    def simulate_movement(self, num_people, starting_outside=True):
        """Simulate people movement with improved positioning and pathfinding"""
        if not self.graph or len(self.graph.nodes()) == 0:
            st.error("Navigation graph not properly initialized")
            return []
            
        try:
            np.random.seed(42)  # For reproducibility
            self.people = []
            
            # Determine bounds for people placement
            bounds = self.mesh.bounds
            
            if starting_outside:
                # More sophisticated outside placement
                for _ in range(num_people):
                    # Randomly choose a side of the building
                    side = np.random.randint(4)
                    
                    if side == 0:  # Left side
                        x = bounds[0, 0] - np.random.uniform(1, 3)
                        y = np.random.uniform(bounds[0, 1], bounds[1, 1])
                    elif side == 1:  # Right side
                        x = bounds[1, 0] + np.random.uniform(1, 3)
                        y = np.random.uniform(bounds[0, 1], bounds[1, 1])
                    elif side == 2:  # Front side
                        x = np.random.uniform(bounds[0, 0], bounds[1, 0])
                        y = bounds[0, 1] - np.random.uniform(1, 3)
                    else:  # Back side
                        x = np.random.uniform(bounds[0, 0], bounds[1, 0])
                        y = bounds[1, 1] + np.random.uniform(1, 3)
                    
                    # Find closest accessible node
                    closest_node = self._find_closest_accessible_node([x, y, 0.1])
                    if closest_node is not None:
                        person = self._create_person(closest_node)
                        # Add some variation in starting speed and direction
                        person['speed'] = np.random.uniform(1.0, 2.0)
                        self.people.append(person)
            else:
                # Place people randomly inside the building
                available_nodes = list(self.graph.nodes())
                if not available_nodes:
                    st.error("No valid positions found inside the building")
                    return []
                
                chosen_nodes = np.random.choice(available_nodes, 
                                            min(num_people, len(available_nodes)), 
                                            replace=len(available_nodes) < num_people)
                
                for node in chosen_nodes:
                    person = self._create_person(node)
                    # Add some variation in starting speed
                    person['speed'] = np.random.uniform(0.8, 1.5)
                    self.people.append(person)
            
            # Initialize paths for all people
            self._initialize_paths()
            
            # Initial visualization of people placement
            st.info(f"Placed {len(self.people)} people for evacuation")
            
            return self.people
            
        except Exception as e:
            st.error(f"Error in movement simulation: {str(e)}")
            return []
    def _create_person(self, node_index):
        """Create a person with initial properties"""
        return {
            'position': self.walkable_points[node_index],
            'current_node': node_index,
            'target_node': None,
            'path': [],
            'speed': np.random.uniform(1.2, 1.8),  # Random walking speed (m/s)
            'state': 'moving'  # Can be 'moving', 'waiting', or 'exited'
        }

    def _find_closest_accessible_node(self, point):
        """Find closest accessible node in navigation graph"""
        try:
            distances = np.linalg.norm(
                self.walkable_points - np.array(point),
                axis=1
            )
            sorted_indices = np.argsort(distances)
            
            # Check first 5 closest points for accessibility
            for idx in sorted_indices[:5]:
                if idx in self.graph:
                    return idx
                    
            return sorted_indices[0] if len(sorted_indices) > 0 else None
            
        except Exception as e:
            st.warning(f"Error finding closest node: {str(e)}")
            return None

    def _initialize_paths(self):
        """Initialize movement paths for all people"""
        for person in self.people:
            self._assign_new_target(person)

    def _assign_new_target(self, person):
        """Assign a new target destination for a person with improved exit prioritization"""
        try:
            available_nodes = list(self.graph.nodes())
            if not available_nodes:
                return
            
            # Strong preference for exits
            if self.exits:
                # Sort exits by proximity
                sorted_exits = sorted(
                    self.exits, 
                    key=lambda exit: np.linalg.norm(
                        np.array(exit['location']) - person['position'][:2]
                    )
                )
                
                # Try to find a path to the closest exit
                for exit_point in sorted_exits:
                    target_node = self._find_closest_accessible_node(
                        np.array(exit_point['location'] + [0.1])
                    )
                    
                    if target_node is not None and target_node != person['current_node']:
                        try:
                            path = nx.shortest_path(
                                self.graph, 
                                person['current_node'], 
                                target_node
                            )
                            person['path'] = [self.walkable_points[n] for n in path]
                            person['target_node'] = target_node
                            return
                        except nx.NetworkXNoPath:
                            continue
            
            # Fallback to random target if no exit path found
            target_node = np.random.choice(available_nodes)
            if target_node != person['current_node']:
                try:
                    path = nx.shortest_path(
                        self.graph, 
                        person['current_node'], 
                        target_node
                    )
                    person['path'] = [self.walkable_points[n] for n in path]
                    person['target_node'] = target_node
                except nx.NetworkXNoPath:
                    st.warning("No path found for person")
                    
        except Exception as e:
            st.warning(f"Error assigning target: {str(e)}")

    def identify_exits(self, manual_selection=False):
        """
        Enhanced exit identification method
        
        Args:
            manual_selection (bool): Allow manual exit selection
        """
        if manual_selection:
            # Interactive exit selection
            st.subheader("Exit Selection")
            st.info("Click on the mesh to mark potential exits")
            
            # Placeholder for interactive mesh selection
            # You might want to implement a more sophisticated selection method
            exit_coords = st.text_input("Enter Exit Coordinates (x, y, z)")
            
            if exit_coords:
                try:
                    coords = list(map(float, exit_coords.split(',')))
                    self.exits.append({
                        'location': coords,
                        'is_manual': True
                    })
                except ValueError:
                    st.error("Invalid coordinate format")
        
        if not self.exits:
            # Automatic exit detection
            bounds = self.mesh.bounds
            
            # Detect potential exits based on mesh boundaries
            exit_candidates = [
                # Side exits
                [bounds[0, 0], (bounds[0, 1] + bounds[1, 1])/2, bounds[0, 2]],  # Left
                [bounds[1, 0], (bounds[0, 1] + bounds[1, 1])/2, bounds[0, 2]],  # Right
                [(bounds[0, 0] + bounds[1, 0])/2, bounds[0, 1], bounds[0, 2]],  # Front
                [(bounds[0, 0] + bounds[1, 0])/2, bounds[1, 1], bounds[0, 2]]   # Back
            ]
            
            # Filter and validate exits
            self.exits = [
                {'location': point, 'is_manual': False}
                for point in exit_candidates
                if not self._is_point_inside_mesh(point)
            ]
        
        st.info(f"Detected {len(self.exits)} potential exit points")
        return self.exits

    def generate_heat_map(self, people):
        """
        Generate a more sophisticated heat map showing population density
        
        Args:
            people (list): List of people in the simulation
        
        Returns:
            dict: Heat map data
        """
        try:
            # Extract 2D positions
            positions = np.array([p['position'][:2] for p in people])
            
            # Use Kernel Density Estimation for heat map
            kde = KernelDensity(bandwidth=0.5, metric='euclidean')
            kde.fit(positions)
            
            # Create a grid of points
            bounds = self.mesh.bounds
            x = np.linspace(bounds[0, 0], bounds[1, 0], 50)
            y = np.linspace(bounds[0, 1], bounds[1, 1], 50)
            xx, yy = np.meshgrid(x, y)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Compute density
            density = np.exp(kde.score_samples(grid_points))
            
            # Normalize density
            density = (density - density.min()) / (density.max() - density.min())
            
            # Reshape for visualization
            heat_map = density.reshape(xx.shape)
            
            # Additional metrics
            self.heat_map_data = {
                'x': x,
                'y': y,
                'z': heat_map,
                'max_density': density.max(),
                'avg_density': density.mean(),
                'crowded_areas': np.where(density > density.mean() + density.std(), 1, 0)
            }
            
            return self.heat_map_data
        
        except Exception as e:
            st.warning(f"Heat map generation error: {e}")
            return None

    def _is_point_inside_mesh(self, point):
        """
        Advanced point-in-mesh detection
        
        Args:
            point (list): 3D point coordinates
        
        Returns:
            bool: Whether point is inside the mesh
        """
        try:
            # Ray-based point containment check
            rays = [
                [point, [1, 0, 0]],
                [point, [-1, 0, 0]],
                [point, [0, 1, 0]],
                [point, [0, -1, 0]],
                [point, [0, 0, 1]],
                [point, [0, 0, -1]]
            ]
            
            # Count intersections
            intersections = sum(1 for ray in rays if self.mesh.ray.intersects_any([ray]))
            
            # Odd number of intersections means point is inside
            return intersections % 2 == 1
        
        except Exception:
            return False

    def update_simulation(self, dt=0.1):
        try:
            # Track evacuation progress
            total_people = len(self.people)
            exited_people = 0
            active_people = 0

            for person in self.people:
                # Skip already exited people
                if person['state'] == 'exited':
                    exited_people += 1
                    continue

                # Count active people
                active_people += 1

                # If no path, assign a new target (preferably an exit)
                if not person['path']:
                    self._assign_new_target(person)
                    continue

                # Move toward next point in path
                target = person['path'][0]
                direction = target - person['position']
                distance = np.linalg.norm(direction)

                if distance < 0.2:  # Slightly increased threshold
                    person['position'] = target
                    person['path'].pop(0)

                    # Update current node
                    person['current_node'] = self._find_closest_accessible_node(person['position'])

                    # Check if reached final target
                    if not person['path']:
                        # Check if at an exit
                        for exit_point in self.exits:
                            exit_location = np.array(exit_point['location'] + [0.1])
                            if np.linalg.norm(person['position'][:2] - exit_location[:2]) < 1.0:
                                person['state'] = 'exited'
                                break
                else:
                    # Move in direction of target
                    movement = direction / distance * person['speed'] * dt
                    person['position'] += movement

            # Recalculate exited people
            exited_people = sum(1 for person in self.people if person['state'] == 'exited')

            # Calculate evacuation progress
            evacuation_progress = (exited_people / total_people) * 100 if total_people > 0 else 0

            # Provide status updates
            st.sidebar.markdown("### Evacuation Status")
            st.sidebar.progress(evacuation_progress / 100)
            st.sidebar.write(f"Total People: {total_people}")
            st.sidebar.write(f"Exited People: {exited_people}")
            st.sidebar.write(f"Active People: {active_people}")

            # Generate heat map during simulation
            if self.people:
                self.generate_heat_map(self.people)

            # Check if evacuation is complete with more robust condition
            if exited_people == total_people or active_people == 0:
                st.success("🎉 Evacuation Complete!")
                return False  # Stop simulation

            return True

        except Exception as e:
            st.error(f"Error updating simulation: {str(e)}")
            return False