import streamlit as st
import trimesh
import numpy as np
import time
from pathlib import Path
import plotly.graph_objects as go
from src import BuildingSimulator, create_simulation_plot

def main():
    st.set_page_config(
        page_title="Building Evacuation Simulator", 
        page_icon="🏢", 
        layout="wide"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .reportview-container {
        background: #F0F2F6;
    }
    .sidebar .sidebar-content {
        background: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏢 KAKI")
    st.markdown("Simulate crowd movement and evacuation scenarios for fault analysis")
    
    # Sidebar for configuration
    st.sidebar.header("Simulation Parameters")
    
    # Initialize simulator at the start
    if 'simulator' not in st.session_state:
        st.session_state.simulator = BuildingSimulator()
    
    # File upload with type verification
    uploaded_file = st.file_uploader(
        "Upload CAD file (.dxf) or 3D model", 
        type=['dxf', 'gltf', 'glb', 'stl', 'obj', 'ply'],
        help="Supported formats: DXF, GLTF, GLB, STL, OBJ, PLY"
    )
    
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        # Save uploaded file temporarily
        temp_path = Path(f"temp.{file_ext}")    
        try:
            temp_path.write_bytes(uploaded_file.getvalue())
            
            # Load and validate file
            try:
                if file_ext == "dxf":
                    mesh = st.session_state.simulator.load_cad(temp_path)
                else:
                    mesh = st.session_state.simulator.load_file(temp_path)
                
                if mesh is None:
                    st.error("Failed to load the 3D model. Please check the file.")
                    return
                
            except Exception as load_error:
                st.error(f"Error loading file: {load_error}")
                return
            
            # Setup navigation
            try:
                st.session_state.simulator.setup_navigation()
            except Exception as nav_error:
                st.error(f"Navigation setup failed: {nav_error}")
                return
            
            # Simulation Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                num_people = st.slider(
                    "Number of People", 
                    min_value=1, 
                    max_value=100, 
                    value=10,
                    help="Total number of people in the simulation"
                )
            
            with col2:
                starting_position = st.radio(
                    "Starting Position", 
                    ["Outside building", "Inside building"],
                    help="Choose where people start their evacuation"
                )
            
            # Simulation Controls
            col3, col4 = st.columns(2)
            
            with col3:
                simulation_running = st.checkbox(
                    "Run Continuous Simulation", 
                    help="Continuously update simulation in real-time"
                )
            
            with col4:
                step_simulation = st.button(
                    "Step Simulation", 
                    help="Advance simulation by one step"
                )
            
            # Optional Visualizations
            show_heatmap = st.checkbox(
                "Show Population Density Heat Map", 
                help="Display crowd density visualization"
            )
            
            # Start Evacuation Button
            if st.button("🚨 Start Evacuation"):
                try:
                    # Reset simulation state
                    st.session_state.simulation_active = True
                    st.session_state.running = True
                    st.session_state.simulation_start_time = time.time()
                    
                    # Simulate movement
                    people = st.session_state.simulator.simulate_movement(
                        num_people, 
                        starting_outside=(starting_position == "Outside building")
                    )
                    
                    if people:
                        st.session_state.mesh = mesh
                        st.session_state.people = people
                    else:
                        st.error("Failed to initialize people for simulation")
                
                except Exception as sim_error:
                    st.error(f"Simulation initialization error: {sim_error}")
            
            # Simulation Visualization and Control
            if hasattr(st.session_state, 'simulation_active') and st.session_state.simulation_active:
                try:
                    # Continuous Simulation
                    if simulation_running:
                        # Update simulation state
                        st.session_state.running = st.session_state.simulator.update_simulation()
                        
                        # Visualization
                        placeholder = st.empty()
                        with placeholder.container():
                            fig = create_simulation_plot(
                                st.session_state.mesh, 
                                st.session_state.people, 
                                show_heatmap
                            )
                            
                            # Handle different plot types
                            if fig is not None:
                                if isinstance(fig, tuple):
                                    simulation_fig, heatmap_fig = fig
                                    st.plotly_chart(simulation_fig, use_container_width=True)
                                    if show_heatmap:
                                        st.plotly_chart(heatmap_fig, use_container_width=True)
                                else:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Add a small delay to control update frequency
                        time.sleep(0.5)
                        
                        # Force page rerun to update visualization
                        st.rerun()
                    
                    # Step Simulation
                    elif step_simulation:
                        # Step-by-step simulation logic
                        st.session_state.running = st.session_state.simulator.update_simulation()
                        
                        # Visualization for step
                        fig = create_simulation_plot(
                            st.session_state.mesh, 
                            st.session_state.people, 
                            show_heatmap
                        )
                        
                        # Handle different plot types
                        if fig is not None:
                            if isinstance(fig, tuple):
                                simulation_fig, heatmap_fig = fig
                                st.plotly_chart(simulation_fig, use_container_width=True)
                                if show_heatmap:
                                    st.plotly_chart(heatmap_fig, use_container_width=True)
                            else:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Simulation Completion Check
                    if not st.session_state.running:
                        st.session_state.simulation_active = False
                        st.success("🎉 Evacuation Complete!")
                
                except Exception as update_error:
                    st.error(f"Simulation update error: {update_error}")
                    st.session_state.running = False
                    st.session_state.simulation_active = False
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

if __name__ == "__main__":
    main()