"""
Temporal Trust Field Visualizer

This module provides visualization tools for the Temporal Trust Field component
of ConfidenceID. It enables the visualization of trust fields as dynamic entities
that evolve over time, helping users understand and interpret the temporal
dynamics of trust.

The implementation is based on the temporal trust field theory described in 
claude.metalayer.txt (Layer 8.1).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

from .field_tensors import (
    TrustFieldTensor, TrustFieldSnapshot, 
    TrustFieldTimeSeries, TrustFieldVisualizationConfig
)


class TrustFieldVisualizer:
    """
    Provides visualization tools for trust fields.
    
    This class enables the visualization of trust fields as dynamic entities that
    evolve over time, helping users understand and interpret the temporal dynamics
    of trust through various visualization methods like line charts, heatmaps, and
    3D visualizations.
    """
    
    def __init__(self, config: Optional[TrustFieldVisualizationConfig] = None):
        """
        Initialize the trust field visualizer.
        
        Args:
            config: Optional visualization configuration
        """
        self.config = config or TrustFieldVisualizationConfig()
    
    def visualize_time_series(self, 
                             time_series: TrustFieldTimeSeries, 
                             title: str = "Trust Field Evolution Over Time",
                             fig_size: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Visualize a time series of trust field snapshots as a line chart.
        
        Args:
            time_series: The trust field time series to visualize
            title: Title for the visualization
            fig_size: Size of the figure (width, height) in inches
            
        Returns:
            Matplotlib figure containing the visualization
        """
        # Filter snapshots based on time range in config
        snapshots = time_series.get_snapshots(
            self.config.start_time, 
            self.config.end_time
        )
        
        if not snapshots:
            # Create an empty figure if no snapshots
            fig, ax = plt.subplots(figsize=fig_size)
            ax.text(0.5, 0.5, "No data available for the specified time range",
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return fig
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Prepare data for plotting
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        
        # Plot confidence values if enabled
        if self.config.show_confidence:
            confidence_values = [snapshot.tensor.confidence for snapshot in snapshots]
            ax.plot(timestamps, confidence_values, 
                   color=self.config.confidence_color, 
                   label="Confidence", linewidth=2)
        
        # Plot velocity values if enabled
        if self.config.show_velocity:
            velocity_values = [snapshot.tensor.velocity for snapshot in snapshots]
            ax.plot(timestamps, velocity_values, 
                   color=self.config.velocity_color, 
                   label="Velocity", linewidth=1.5, linestyle="--")
        
        # Plot acceleration values if enabled
        if self.config.show_acceleration:
            acceleration_values = [snapshot.tensor.acceleration for snapshot in snapshots]
            ax.plot(timestamps, acceleration_values, 
                   color=self.config.acceleration_color, 
                   label="Acceleration", linewidth=1, linestyle="-.")
        
        # Plot stability values if enabled
        if self.config.show_stability:
            stability_values = [snapshot.tensor.stability for snapshot in snapshots]
            ax.plot(timestamps, stability_values, 
                   color=self.config.stability_color, 
                   label="Stability", linewidth=1.5, linestyle=":")
        
        # Highlight anomalies if enabled
        if self.config.highlight_anomalies:
            self._highlight_anomalies(ax, snapshots)
        
        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(title)
        
        # Format x-axis to show readable dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    
    def visualize_trust_field_heatmap(self,
                                     time_series: TrustFieldTimeSeries,
                                     dimension: str = "confidence",
                                     title: str = "Trust Field Heatmap",
                                     fig_size: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Visualize a trust field time series as a heatmap.
        
        Args:
            time_series: The trust field time series to visualize
            dimension: Which dimension to visualize ('confidence', 'velocity', 'stability')
            title: Title for the visualization
            fig_size: Size of the figure (width, height) in inches
            
        Returns:
            Matplotlib figure containing the visualization
        """
        # Filter snapshots based on time range in config
        snapshots = time_series.get_snapshots(
            self.config.start_time, 
            self.config.end_time
        )
        
        if not snapshots:
            # Create an empty figure if no snapshots
            fig, ax = plt.subplots(figsize=fig_size)
            ax.text(0.5, 0.5, "No data available for the specified time range",
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return fig
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Prepare data for heatmap
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        
        # Get values for the specified dimension
        if dimension == "confidence":
            values = [snapshot.tensor.confidence for snapshot in snapshots]
            cmap = plt.cm.Greens
            label = "Confidence"
        elif dimension == "velocity":
            values = [snapshot.tensor.velocity for snapshot in snapshots]
            cmap = plt.cm.coolwarm
            label = "Velocity"
        elif dimension == "stability":
            values = [snapshot.tensor.stability for snapshot in snapshots]
            cmap = plt.cm.YlOrRd
            label = "Stability"
        else:
            # Default to confidence
            values = [snapshot.tensor.confidence for snapshot in snapshots]
            cmap = plt.cm.Greens
            label = "Confidence"
        
        # Create a 2D array for the heatmap (using time bins)
        # This example creates a simple heatmap with time on x-axis and a single row
        # More complex heatmaps could include additional dimensions (e.g., content type)
        heatmap_data = np.array([values])
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap)
        
        # Set labels and title
        ax.set_yticks([])  # No y-ticks for single row
        
        # Format x-axis to show readable dates
        num_ticks = min(len(timestamps), 10)  # Limit to avoid overcrowding
        tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([timestamps[i].strftime("%Y-%m-%d %H:%M") for i in tick_indices], 
                          rotation=45)
        
        ax.set_title(f"{title} - {label}")
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(label)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    
    def visualize_trust_field_3d(self,
                                time_series: TrustFieldTimeSeries,
                                title: str = "Trust Field 3D Visualization",
                                fig_size: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Create a 3D visualization of the trust field evolution.
        
        Args:
            time_series: The trust field time series to visualize
            title: Title for the visualization
            fig_size: Size of the figure (width, height) in inches
            
        Returns:
            Matplotlib figure containing the 3D visualization
        """
        # Filter snapshots based on time range in config
        snapshots = time_series.get_snapshots(
            self.config.start_time, 
            self.config.end_time
        )
        
        if not snapshots:
            # Create an empty figure if no snapshots
            fig, ax = plt.subplots(figsize=fig_size)
            ax.text(0.5, 0.5, "No data available for the specified time range",
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return fig
        
        # Create 3D figure and axes
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data for 3D plot
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        # Convert timestamps to numeric values for plotting
        time_values = mdates.date2num(timestamps)
        
        # Get values for different dimensions
        confidence_values = [snapshot.tensor.confidence for snapshot in snapshots]
        velocity_values = [snapshot.tensor.velocity for snapshot in snapshots]
        stability_values = [snapshot.tensor.stability for snapshot in snapshots]
        
        # Plot 3D surface
        ax.plot3D(time_values, confidence_values, stability_values, 
                 color=self.config.confidence_color, linewidth=2, label="Trust Trajectory")
        
        # Add points to highlight each snapshot
        ax.scatter(time_values, confidence_values, stability_values, 
                  color=self.config.confidence_color, s=30)
        
        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Confidence")
        ax.set_zlabel("Stability")
        ax.set_title(title)
        
        # Format x-axis to show readable dates
        date_formatter = mdates.DateFormatter("%Y-%m-%d")
        num_ticks = min(len(timestamps), 5)  # Limit to avoid overcrowding
        tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
        ax.set_xticks([time_values[i] for i in tick_indices])
        ax.set_xticklabels([date_formatter(timestamps[i]) for i in tick_indices])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_ylim(0, 1)  # Confidence range
        ax.set_zlim(0, 1)  # Stability range
        
        # Adjust view angle for better visualization
        ax.view_init(elev=30, azim=45)
        
        # Add legend
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def visualize_beverly_band(self,
                              time_series: TrustFieldTimeSeries,
                              field_width: float = 0.2,
                              title: str = "Beverly Band - Trust Field Stability Envelope",
                              fig_size: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Visualize the Beverly Band - the dynamic stability envelope of the trust field.
        
        The Beverly Band represents the region around the trust field's trajectory
        within which trust can be safely manipulated without destabilizing the system.
        
        Args:
            time_series: The trust field time series to visualize
            field_width: Width of the Beverly Band (as a fraction of the trust value)
            title: Title for the visualization
            fig_size: Size of the figure (width, height) in inches
            
        Returns:
            Matplotlib figure containing the visualization
        """
        # Filter snapshots based on time range in config
        snapshots = time_series.get_snapshots(
            self.config.start_time, 
            self.config.end_time
        )
        
        if not snapshots:
            # Create an empty figure if no snapshots
            fig, ax = plt.subplots(figsize=fig_size)
            ax.text(0.5, 0.5, "No data available for the specified time range",
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return fig
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Prepare data for plotting
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        confidence_values = [snapshot.tensor.confidence for snapshot in snapshots]
        
        # Calculate Beverly Band boundaries
        # The width of the band is influenced by stability, amplification, and decay rate
        upper_band = []
        lower_band = []
        
        for snapshot in snapshots:
            # Calculate band width based on tensor properties
            # Higher stability → narrower band (more precise)
            # Higher amplification → wider band (more potential for growth/change)
            # Higher decay rate → wider band (more volatile)
            tensor = snapshot.tensor
            band_factor = field_width * (1.0 + tensor.amplification * 0.1) * \
                         (1.0 + tensor.decay_rate * 5.0) * (2.0 - tensor.stability)
            
            upper_bound = min(1.0, tensor.confidence + band_factor)
            lower_bound = max(0.0, tensor.confidence - band_factor)
            
            upper_band.append(upper_bound)
            lower_band.append(lower_bound)
        
        # Plot the main confidence line
        ax.plot(timestamps, confidence_values, 
               color=self.config.confidence_color, 
               label="Confidence", linewidth=2)
        
        # Plot the Beverly Band
        ax.fill_between(timestamps, lower_band, upper_band, 
                       color=self.config.confidence_color, alpha=0.2,
                       label="Beverly Band")
        
        # Add annotations for significant changes in the band width
        self._annotate_significant_changes(ax, timestamps, upper_band, lower_band)
        
        # Set labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Confidence")
        ax.set_title(title)
        
        # Format x-axis to show readable dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    
    def visualize_field_components(self,
                                 time_series: TrustFieldTimeSeries,
                                 title: str = "Trust Field Component Analysis",
                                 fig_size: Tuple[int, int] = (12, 10)) -> Figure:
        """
        Visualize all components of the trust field in a multi-panel figure.
        
        Args:
            time_series: The trust field time series to visualize
            title: Title for the visualization
            fig_size: Size of the figure (width, height) in inches
            
        Returns:
            Matplotlib figure containing the visualization
        """
        # Filter snapshots based on time range in config
        snapshots = time_series.get_snapshots(
            self.config.start_time, 
            self.config.end_time
        )
        
        if not snapshots:
            # Create an empty figure if no snapshots
            fig, ax = plt.subplots(figsize=fig_size)
            ax.text(0.5, 0.5, "No data available for the specified time range",
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            return fig
        
        # Create figure and subplots
        fig, axs = plt.subplots(3, 2, figsize=fig_size)
        fig.suptitle(title, fontsize=16)
        
        # Prepare data for plotting
        timestamps = [snapshot.timestamp for snapshot in snapshots]
        
        # Get values for different dimensions
        confidence_values = [snapshot.tensor.confidence for snapshot in snapshots]
        velocity_values = [snapshot.tensor.velocity for snapshot in snapshots]
        acceleration_values = [snapshot.tensor.acceleration for snapshot in snapshots]
        decay_rate_values = [snapshot.tensor.decay_rate for snapshot in snapshots]
        amplification_values = [snapshot.tensor.amplification for snapshot in snapshots]
        stability_values = [snapshot.tensor.stability for snapshot in snapshots]
        
        # 1. Confidence plot
        axs[0, 0].plot(timestamps, confidence_values, 
                      color=self.config.confidence_color, linewidth=2)
        axs[0, 0].set_title("Confidence")
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Velocity plot
        axs[0, 1].plot(timestamps, velocity_values, 
                      color=self.config.velocity_color, linewidth=2)
        axs[0, 1].set_title("Velocity")
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Acceleration plot
        axs[1, 0].plot(timestamps, acceleration_values, 
                      color=self.config.acceleration_color, linewidth=2)
        axs[1, 0].set_title("Acceleration")
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Decay Rate plot
        axs[1, 1].plot(timestamps, decay_rate_values, 
                      color='purple', linewidth=2)
        axs[1, 1].set_title("Decay Rate")
        axs[1, 1].grid(True, alpha=0.3)
        
        # 5. Amplification plot
        axs[2, 0].plot(timestamps, amplification_values, 
                      color='brown', linewidth=2)
        axs[2, 0].set_title("Amplification")
        axs[2, 0].grid(True, alpha=0.3)
        
        # 6. Stability plot
        axs[2, 1].plot(timestamps, stability_values, 
                      color=self.config.stability_color, linewidth=2)
        axs[2, 1].set_title("Stability")
        axs[2, 1].set_ylim(0, 1)
        axs[2, 1].grid(True, alpha=0.3)
        
        # Format x-axis on all subplots
        for i in range(3):
            for j in range(2):
                axs[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                axs[i, j].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for the suptitle
        
        return fig
    
    def _highlight_anomalies(self, ax, snapshots: List[TrustFieldSnapshot]) -> None:
        """
        Highlight anomalies in the trust field visualization.
        
        Args:
            ax: Matplotlib axis to plot on
            snapshots: List of trust field snapshots
        """
        from .field_dynamics_engine import FieldDynamicsEngine
        
        # Create a dynamics engine to detect anomalies
        dynamics_engine = FieldDynamicsEngine()
        
        # Check each snapshot for anomalies
        for i, snapshot in enumerate(snapshots):
            anomalies = dynamics_engine.detect_trust_anomalies(snapshot.tensor)
            
            if anomalies:
                # Mark the point on the visualization
                ax.scatter([snapshot.timestamp], [snapshot.tensor.confidence], 
                          color='red', s=50, zorder=5)
                
                # Add an annotation for the first few anomalies (to avoid cluttering)
                if i < len(snapshots) - 1 and i % max(1, len(snapshots) // 10) == 0:
                    anomaly_text = "\n".join(f"{k}: {v:.2f}" for k, v in anomalies.items())
                    ax.annotate(anomaly_text, 
                               xy=(snapshot.timestamp, snapshot.tensor.confidence),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    def _annotate_significant_changes(self, 
                                     ax, 
                                     timestamps: List[datetime],
                                     upper_band: List[float],
                                     lower_band: List[float]) -> None:
        """
        Annotate significant changes in the Beverly Band width.
        
        Args:
            ax: Matplotlib axis to plot on
            timestamps: List of timestamps
            upper_band: Upper boundary of the Beverly Band
            lower_band: Lower boundary of the Beverly Band
        """
        # Calculate band widths
        band_widths = [u - l for u, l in zip(upper_band, lower_band)]
        
        # Find points of significant change in band width
        if len(band_widths) < 3:
            return
        
        # Calculate the derivative of band width
        width_changes = [band_widths[i+1] - band_widths[i] for i in range(len(band_widths)-1)]
        
        # Find indices of significant changes (using standard deviation as a threshold)
        std_dev = np.std(width_changes)
        significant_indices = [i for i, change in enumerate(width_changes) 
                              if abs(change) > 2 * std_dev]
        
        # Limit to a reasonable number of annotations
        if len(significant_indices) > 5:
            # Sort by absolute magnitude of change and take the top 5
            significant_indices = sorted(significant_indices, 
                                        key=lambda i: abs(width_changes[i]), 
                                        reverse=True)[:5]
        
        # Add annotations
        for idx in significant_indices:
            if width_changes[idx] > 0:
                annotation = "Band Widening"
                color = 'red'
            else:
                annotation = "Band Narrowing"
                color = 'green'
            
            ax.annotate(annotation, 
                       xy=(timestamps[idx+1], (upper_band[idx+1] + lower_band[idx+1]) / 2),
                       xytext=(10, 0), textcoords='offset points',
                       color=color, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    def export_to_html(self, 
                      time_series: TrustFieldTimeSeries,
                      output_path: str,
                      include_3d: bool = True) -> None:
        """
        Export trust field visualizations to an HTML file.
        
        Args:
            time_series: The trust field time series to visualize
            output_path: Path to save the HTML file
            include_3d: Whether to include 3D visualization (can be slow for large datasets)
        """
        import mpld3
        
        # Create visualizations
        fig1 = self.visualize_time_series(time_series)
        fig2 = self.visualize_beverly_band(time_series)
        fig3 = self.visualize_field_components(time_series)
        
        # Combine figures into HTML
        html_content = "<html><head><title>Trust Field Analysis</title>"
        html_content += "<style>body{font-family:Arial;margin:20px;}</style></head><body>"
        html_content += "<h1>Trust Field Analysis</h1>"
        
        # Add time series visualization
        html_content += "<h2>Trust Field Evolution Over Time</h2>"
        html_content += mpld3.fig_to_html(fig1)
        
        # Add Beverly Band visualization
        html_content += "<h2>Beverly Band - Trust Field Stability Envelope</h2>"
        html_content += mpld3.fig_to_html(fig2)
        
        # Add component analysis
        html_content += "<h2>Trust Field Component Analysis</h2>"
        html_content += mpld3.fig_to_html(fig3)
        
        # Add 3D visualization if requested
        if include_3d:
            fig4 = self.visualize_trust_field_3d(time_series)
            html_content += "<h2>3D Trust Field Visualization</h2>"
            html_content += mpld3.fig_to_html(fig4)
        
        # Close HTML
        html_content += "</body></html>"
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"Visualization exported to {output_path}")
