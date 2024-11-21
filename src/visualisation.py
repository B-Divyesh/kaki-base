import plotly.graph_objects as go
import numpy as np
import streamlit as st
from sklearn.neighbors import KernelDensity

def create_simulation_plot(mesh, people, show_heatmap=False):
    """Create a plotly figure for visualization"""
    try:
        # Create main figure
        fig = go.Figure()
        
        # Add building mesh
        if mesh is not None:
            vertices = mesh.vertices
            for face in mesh.faces:
                face_vertices = vertices[face]
                fig.add_trace(go.Mesh3d(
                    x=face_vertices[:, 0],
                    y=face_vertices[:, 1],
                    z=face_vertices[:, 2],
                    color='lightgray',
                    opacity=0.5,
                    showscale=False
                ))
        
        # Add people as 3D scatter
        if people:
            try:
                # Separate people by state
                exited_people = [p for p in people if p['state'] == 'exited']
                active_people = [p for p in people if p['state'] != 'exited']
                
                # Plot active people
                if active_people:
                    active_positions = np.array([p['position'] for p in active_people])
                    fig.add_trace(go.Scatter3d(
                        x=active_positions[:, 0],
                        y=active_positions[:, 1],
                        z=active_positions[:, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='red',
                            opacity=0.8
                        ),
                        name='Active People'
                    ))
                
                # Plot exited people
                if exited_people:
                    exited_positions = np.array([p['position'] for p in exited_people])
                    fig.add_trace(go.Scatter3d(
                        x=exited_positions[:, 0],
                        y=exited_positions[:, 1],
                        z=exited_positions[:, 2],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='green',
                            opacity=0.8
                        ),
                        name='Exited People'
                    ))
            
            except Exception as people_error:
                st.warning(f"Error plotting people: {people_error}")
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title='Building Evacuation Simulation',
            height=600
        )
        
        # Heat Map Generation
        heatmap_fig = None
        if show_heatmap and people:
            try:
                # Generate heat map
                heatmap_fig = generate_heatmap(people)
            except Exception as heatmap_error:
                st.warning(f"Heat map generation error: {heatmap_error}")
        
        # Return figures based on heat map option
        if heatmap_fig:
            return fig, heatmap_fig
        return fig
    
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def generate_heatmap(people):
    """Generate a 2D heat map of people's positions"""
    try:
        # Extract 2D positions
        positions = np.array([p['position'][:2] for p in people])
        
        # Use Kernel Density Estimation
        kde = KernelDensity(bandwidth=0.5, metric='euclidean')
        kde.fit(positions)
        
        # Create grid for heat map
        x = np.linspace(positions[:, 0].min()-1, positions[:, 0].max()+1, 100)
        y = np.linspace(positions[:, 1].min()-1, positions[:, 1].max()+1, 100)
        xx, yy = np.meshgrid(x, y)
        
        # Compute density
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        density = np.exp(kde.score_samples(grid_points))
        density = density.reshape(xx.shape)
        
        # Create heat map figure
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=density,
            x=x,
            y=y,
            colorscale='Viridis'
        ))
        
        heatmap_fig.update_layout(
            title='Population Density Heat Map',
            xaxis_title='X',
            yaxis_title='Y',
            height=600
        )
        
        return heatmap_fig
    
    except Exception as e:
        st.warning(f"Heat map generation error: {e}")
        return None