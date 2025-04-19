import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import logging
import traceback


class Visualizer:
    """
    A modern visualization class using Plotly for interactive 3D visualization
    of cube/pile representations with volume, mass and density information.
    """

    def __init__(self, logging_level=logging.INFO):
        """
        Initialize the CubeVisualizer.

        Args:
            logging_level: The logging level for the visualizer
        """
        # Configure logging
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("CubeVisualizer")

        # Store the latest data for potential reuse
        self._latest_vertices = None
        self._latest_volume = None
        self._latest_density = None
        self._latest_mass = None

        # Initialize figure
        self.fig = None

    def process_data_from_modules(self, segmentation_result=None, processed_image=None, calculation_result=None):
        """
        Process data from other modules and prepare it for visualization.

        Args:
            segmentation_result: Result from segmentation.py (e.g., mask or segmented points)
            processed_image: Result from imageprocessor.py (e.g., processed point cloud)
            calculation_result: Result from calculation.py (e.g., dict with volume, density, mass)

        Returns:
            dict: Processed data ready for visualization
        """
        vertices = None
        volume = None
        density = None
        mass = None

        try:
            # Process segmentation data (if available)
            if segmentation_result is not None:
                if hasattr(segmentation_result, 'points'):
                    vertices = np.array(segmentation_result.points)
                elif isinstance(segmentation_result, dict) and 'points' in segmentation_result:
                    vertices = np.array(segmentation_result['points'])
                self.logger.info(
                    f"Processed segmentation data with {len(vertices) if vertices is not None else 0} points")

            # Process image processing data (if available)
            if processed_image is not None:
                if hasattr(processed_image, 'vertices'):
                    vertices = np.array(processed_image.vertices)
                elif isinstance(processed_image, dict) and 'vertices' in processed_image:
                    vertices = np.array(processed_image['vertices'])
                elif hasattr(processed_image, 'points'):
                    vertices = np.array(processed_image.points)
                elif isinstance(processed_image, dict) and 'points' in processed_image:
                    vertices = np.array(processed_image['points'])
                self.logger.info(f"Processed image data with {len(vertices) if vertices is not None else 0} points")

            # Process calculation data
            if calculation_result is not None:
                if hasattr(calculation_result, 'volume'):
                    volume = calculation_result.volume
                    density = getattr(calculation_result, 'density', None)
                    mass = getattr(calculation_result, 'mass', None)
                elif isinstance(calculation_result, dict):
                    volume = calculation_result.get('volume')
                    density = calculation_result.get('density')
                    mass = calculation_result.get('mass')
                self.logger.info(f"Processed calculation data: volume={volume}, density={density}, mass={mass}")

            # Store the data for potential reuse
            if vertices is not None:
                self._latest_vertices = vertices
            if volume is not None:
                self._latest_volume = volume
            if density is not None:
                self._latest_density = density
            if mass is not None:
                self._latest_mass = mass

            return {
                'vertices': vertices,
                'volume': volume,
                'density': density,
                'mass': mass
            }

        except Exception as e:
            self.logger.error(f"Error processing module data: {str(e)}\n{traceback.format_exc()}")
            return None

    def plot_cube(self, corners_3d=None, volume=None, density=None, mass=None):
        """
        Plot a 3D cube or convex hull based on the given corners and display its properties.

        Args:
            corners_3d: 3D coordinates of cube corners or point cloud, shape (n, 3)
            volume: Volume of the object in cubic meters
            density: Density of the object in kg/m³
            mass: Mass of the object in kg
        """
        # Use latest data if not provided
        if corners_3d is None and self._latest_vertices is not None:
            corners_3d = self._latest_vertices
            self.logger.info("Using latest stored vertices")

        if volume is None and self._latest_volume is not None:
            volume = self._latest_volume
            self.logger.info("Using latest stored volume")

        if density is None and self._latest_density is not None:
            density = self._latest_density
            self.logger.info("Using latest stored density")

        if mass is None and self._latest_mass is not None:
            mass = self._latest_mass
            self.logger.info("Using latest stored mass")

        # Validate required data
        if corners_3d is None:
            self.logger.error("No corner points provided for visualization")
            return

        corners = np.array(corners_3d)

        # Remove duplicate points and ensure we have enough unique points
        unique_corners = np.unique(corners, axis=0)
        if len(unique_corners) < 4:  # Need at least 4 points for a 3D hull
            self.logger.error(f"Insufficient unique points for 3D visualization: {len(unique_corners)} points")
            # Create a scatter plot of the points
            self.fig = go.Figure(data=[
                go.Scatter3d(
                    x=unique_corners[:, 0],
                    y=unique_corners[:, 1],
                    z=unique_corners[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        opacity=0.8
                    ),
                    name='Vertices'
                )
            ])
            return

        try:
            # Create convex hull for visualization
            hull = ConvexHull(unique_corners)

            # Plot points
            points_trace = go.Scatter3d(
                x=unique_corners[:, 0],
                y=unique_corners[:, 1],
                z=unique_corners[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    opacity=0.8
                ),
                name='Vertices'
            )

            # Create mesh data for plotting the hull
            faces = []
            for simplex in hull.simplices:
                faces.append(simplex)

            # Convert faces to the format required by Plotly
            i = []
            j = []
            k = []

            for face in faces:
                if len(face) >= 3:  # Ensure the face has at least 3 points
                    i.append(face[0])
                    j.append(face[1])
                    k.append(face[2])

            # Plot the mesh
            mesh_trace = go.Mesh3d(
                x=unique_corners[:, 0],
                y=unique_corners[:, 1],
                z=unique_corners[:, 2],
                i=i, j=j, k=k,
                opacity=0.3,
                color='blue',
                name='Hull'
            )

            # Calculate and display hull volume if not provided
            if volume is None:
                volume = hull.volume
                self.logger.info(f"Calculated volume from hull: {volume}")
                self._latest_volume = volume

            # Create a figure with both traces
            self.fig = go.Figure(data=[points_trace, mesh_trace])

        except Exception as e:
            self.logger.error(f"Failed to create hull mesh: {str(e)}\n{traceback.format_exc()}")
            # Just plot the points
            self.fig = go.Figure(data=[
                go.Scatter3d(
                    x=unique_corners[:, 0],
                    y=unique_corners[:, 1],
                    z=unique_corners[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        opacity=0.8
                    ),
                    name='Vertices'
                )
            ])

        # Add measurements annotation
        info_text = f'Volume: {volume:.3f} m³<br>' if volume is not None else ''
        if density is not None:
            info_text += f'Density: {density:.2f} kg/m³<br>'
        if mass is not None:
            info_text += f'Mass: {mass:.2f} kg'

        if info_text:
            self.fig.update_layout(
                title=f"3D Pile Reconstruction<br><sub>{info_text}</sub>",
                scene=dict(
                    xaxis_title="X (meters)",
                    yaxis_title="Y (meters)",
                    zaxis_title="Z (meters)",
                    aspectmode='cube'  # Equal aspect ratio
                )
            )
        else:
            self.fig.update_layout(
                title="3D Pile Reconstruction",
                scene=dict(
                    xaxis_title="X (meters)",
                    yaxis_title="Y (meters)",
                    zaxis_title="Z (meters)",
                    aspectmode='cube'  # Equal aspect ratio
                )
            )

    def display_mesh(self, mesh):
        """
        Display a mesh object with its volume.

        Args:
            mesh: The mesh object containing vertices and volume information
        """
        try:
            if not mesh:
                raise ValueError("Mesh object cannot be None")

            # Check for vertices
            vertices = None
            if hasattr(mesh, 'vertices'):
                vertices = np.asarray(mesh.vertices)
            elif hasattr(mesh, 'get_vertices'):
                vertices = np.asarray(mesh.get_vertices())
            elif hasattr(mesh, 'points'):
                vertices = np.asarray(mesh.points)
            else:
                raise ValueError("Mesh object must have vertices attribute or method to get vertices")

            # Check for volume
            volume = None
            if hasattr(mesh, 'volume'):
                volume = mesh.volume
            elif hasattr(mesh, 'compute_volume'):
                volume = mesh.compute_volume()
            elif hasattr(mesh, 'get_volume'):
                volume = mesh.get_volume()
            else:
                # Estimate volume from vertices
                hull = ConvexHull(vertices)
                volume = hull.volume
                self.logger.info(f"Estimated volume from vertices: {volume}")

            # Check for other properties
            density = getattr(mesh, 'density', None)
            mass = getattr(mesh, 'mass', None)

            # Store data for potential reuse
            self._latest_vertices = vertices
            self._latest_volume = volume
            self._latest_density = density
            self._latest_mass = mass

            self.plot_cube(vertices, volume, density, mass)

        except Exception as e:
            self.logger.error(f"Failed to display mesh: {str(e)}\n{traceback.format_exc()}")
            raise

    def visualize_from_calculation_result(self, calculation_result):
        """
        Visualize based on a calculation result from calculation.py

        Args:
            calculation_result: Dict or object with volume, density, mass, and vertices/points
        """
        try:
            # Process the calculation result
            data = self.process_data_from_modules(calculation_result=calculation_result)
            if data and data['vertices'] is not None:
                self.plot_cube(
                    data['vertices'],
                    data['volume'],
                    data['density'],
                    data['mass']
                )
            else:
                self.logger.error("Insufficient data from calculation result for visualization")
        except Exception as e:
            self.logger.error(f"Failed to visualize from calculation result: {str(e)}\n{traceback.format_exc()}")

    def visualize_from_segmentation(self, segmentation_result, calculation_result=None):
        """
        Visualize based on segmentation results from segmentation.py

        Args:
            segmentation_result: Result from segmentation.py with points/vertices
            calculation_result: Optional calculation results with volume, density, mass
        """
        try:
            # Process the segmentation and calculation results
            data = self.process_data_from_modules(
                segmentation_result=segmentation_result,
                calculation_result=calculation_result
            )

            if data and data['vertices'] is not None:
                self.plot_cube(
                    data['vertices'],
                    data['volume'],
                    data['density'],
                    data['mass']
                )
            else:
                self.logger.error("Insufficient data from segmentation for visualization")
        except Exception as e:
            self.logger.error(f"Failed to visualize from segmentation: {str(e)}\n{traceback.format_exc()}")

    def visualize_from_image_processor(self, processed_image, calculation_result=None):
        """
        Visualize based on processed image results from imageprocessor.py

        Args:
            processed_image: Result from imageprocessor.py with points/vertices
            calculation_result: Optional calculation results with volume, density, mass
        """
        try:
            # Process the image processing and calculation results
            data = self.process_data_from_modules(
                processed_image=processed_image,
                calculation_result=calculation_result
            )

            if data and data['vertices'] is not None:
                self.plot_cube(
                    data['vertices'],
                    data['volume'],
                    data['density'],
                    data['mass']
                )
            else:
                self.logger.error("Insufficient data from image processor for visualization")
        except Exception as e:
            self.logger.error(f"Failed to visualize from image processor: {str(e)}\n{traceback.format_exc()}")

    def update_visualization(self, **kwargs):
        """
        Update the visualization with new data.

        Kwargs:
            vertices or points: 3D coordinates for visualization
            volume: Updated volume value
            density: Updated density value
            mass: Updated mass value
        """
        vertices = kwargs.get('vertices') or kwargs.get('points')
        volume = kwargs.get('volume')
        density = kwargs.get('density')
        mass = kwargs.get('mass')

        # Update stored values for any provided parameters
        if vertices is not None:
            self._latest_vertices = np.array(vertices)
        if volume is not None:
            self._latest_volume = volume
        if density is not None:
            self._latest_density = density
        if mass is not None:
            self._latest_mass = mass

        # Plot with latest data (will use stored values for any not provided)
        self.plot_cube(self._latest_vertices, self._latest_volume,
                       self._latest_density, self._latest_mass)

    def show(self, height=800, width=1000):
        """
        Display the visualization.

        Args:
            height: Height of the plot in pixels
            width: Width of the plot in pixels
        """
        if self.fig is None:
            self.logger.error("No visualization to show. Call plot_cube() first.")
            return

        self.fig.update_layout(
            autosize=False,
            width=width,
            height=height
        )
        self.fig.show()

    def save_figure(self, filename, width=1200, height=800):
        """
        Save the visualization to a file.

        Args:
            filename: The filename to save to (include extension like .png, .jpg, .html)
            width: Width of the saved image in pixels
            height: Height of the saved image in pixels
        """
        try:
            if self.fig is None:
                self.logger.error("No visualization to save. Call plot_cube() first.")
                return

            # Set dimensions for the saved figure
            self.fig.update_layout(
                autosize=False,
                width=width,
                height=height
            )

            if filename.endswith('.html'):
                self.fig.write_html(filename)
                self.logger.info(f"Interactive HTML figure saved to {filename}")
            else:
                self.fig.write_image(filename)
                self.logger.info(f"Static image saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save figure: {str(e)}\n{traceback.format_exc()}")