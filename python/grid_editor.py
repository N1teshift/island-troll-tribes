import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, RadioButtons
import sys
import time

# Import the MapAnalyzer class from map_analyzer.py
sys.path.append('.')
from map_analyzer import MapAnalyzer, GRID_WIDTH, GRID_HEIGHT

# Simplified grid values
# 0 = Unwalkable
# 1 = Walkable
# 2 = Spawn Region (for visualization only)

class GridEditor:
    def __init__(self, grid_file=None):
        plt.ion()  # Turn on interactive mode for better responsiveness
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.subplots_adjust(bottom=0.15)
        
        # Initialize the MapAnalyzer
        self.analyzer = MapAnalyzer()
        
        # If a grid file is provided, load it; otherwise, create a new grid
        if grid_file:
            success = self.analyzer.load_grid_from_file(grid_file)
            if not success:
                print("Could not load grid file. Creating a new grid.")
                self.analyzer.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
            else:
                print(f"Grid matrix loaded from {grid_file} with shape {self.analyzer.grid.shape}")
                # Convert the original multi-value grid to binary (0=unwalkable, 1=walkable)
                # Original: 0=unknown, 1=land, 2=water, 3=unwalkable
                walkable_mask = (self.analyzer.grid == 1) | (self.analyzer.grid == 2)
                unwalkable_mask = (self.analyzer.grid == 0) | (self.analyzer.grid == 3)
                self.analyzer.grid = np.zeros_like(self.analyzer.grid)
                self.analyzer.grid[walkable_mask] = 1  # Set walkable areas to 1
        else:
            # Create a new grid
            self.analyzer.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        
        # Current drawing mode (1=walkable, 0=unwalkable)
        self.draw_mode = 1
        
        # Create a colormap for binary grid + spawn regions
        colors = ['red', 'green', 'purple']  # 0=unwalkable(red), 1=walkable(green), 2=spawn(purple)
        bounds = range(4)
        self.cmap = mcolors.ListedColormap(colors)
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)
        
        # Connect events (including mouse release)
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.is_drawing = False
        self.button_pressed = None
        
        # Display parameters
        self.display_grid_lines = True
        self.grid_size = 5  # Size of the drawing brush
        self.last_update_time = time.time()
        self.update_frequency = 0.1  # Update display every 0.1 seconds during continuous drawing
        
        # Store original grid dimensions and offsets for coordinate mapping
        self.original_height = GRID_HEIGHT
        self.original_width = GRID_WIDTH
        self.offset_x = 0
        self.offset_y = 0
        
        # World Editor offset - where (0,0) in our grid corresponds to (17,17) in World Editor
        self.world_editor_offset_x = 17
        self.world_editor_offset_y = 17
        
        # Add UI elements
        self.add_ui_elements()
        
        # Initial drawing of the grid
        self.update_display()
        
        # Store spawn regions
        self.spawn_regions = []
        self.spawn_regions_applied = False
        
        print("Editor initialized. Click and drag to paint. Press ESC to exit.")
        print("Matrix coordinates: (0,0) is at the top-left (array indices)")
        print("Visual coordinates: (0,0) is at the bottom-left (in grid references)")
        print(f"World Editor coordinates: ({self.world_editor_offset_x},{self.world_editor_offset_y}) is at the bottom-left")
    
    def add_ui_elements(self):
        """Add UI elements like buttons and radio buttons"""
        # Add radio buttons for choosing the draw mode (terrain type)
        ax_radio = plt.axes([0.05, 0.05, 0.15, 0.1])
        self.radio = RadioButtons(ax_radio, ('Walkable', 'Unwalkable'))
        self.radio.on_clicked(self.on_radio_clicked)
        
        # Add a button to toggle grid lines
        ax_toggle_grid = plt.axes([0.25, 0.05, 0.15, 0.05])
        self.btn_toggle_grid = Button(ax_toggle_grid, 'Toggle Grid')
        self.btn_toggle_grid.on_clicked(self.toggle_grid_lines)
        
        # Add a button to increase brush size
        ax_increase_size = plt.axes([0.45, 0.05, 0.15, 0.05])
        self.btn_increase_size = Button(ax_increase_size, 'Increase Brush')
        self.btn_increase_size.on_clicked(self.increase_brush_size)
        
        # Add a button to decrease brush size
        ax_decrease_size = plt.axes([0.45, 0.1, 0.15, 0.05])
        self.btn_decrease_size = Button(ax_decrease_size, 'Decrease Brush')
        self.btn_decrease_size.on_clicked(self.decrease_brush_size)
        
        # Add a save button
        ax_save = plt.axes([0.65, 0.05, 0.15, 0.05])
        self.btn_save = Button(ax_save, 'Save Grid')
        self.btn_save.on_clicked(self.save_grid)
        
        # Add an import spawn regions button
        ax_import = plt.axes([0.65, 0.1, 0.15, 0.05])
        self.btn_import = Button(ax_import, 'Import Spawn Regions')
        self.btn_import.on_clicked(self.import_spawn_regions)
        
        # Add a button to apply spawn regions
        ax_apply = plt.axes([0.85, 0.05, 0.1, 0.05])
        self.btn_apply = Button(ax_apply, 'Highlight\nRegions')
        self.btn_apply.on_clicked(self.apply_spawn_regions)
        
        # Add a button to clear spawn region highlighting
        ax_clear = plt.axes([0.85, 0.1, 0.1, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear\nHighlights')
        self.btn_clear.on_clicked(self.clear_spawn_highlights)
        
        # Add a trim button
        ax_trim = plt.axes([0.25, 0.1, 0.15, 0.05])
        self.btn_trim = Button(ax_trim, 'Trim Red Borders')
        self.btn_trim.on_clicked(self.trim_borders)
    
    def on_radio_clicked(self, label):
        """Handle radio button clicks to change the draw mode"""
        if label == 'Walkable':
            self.draw_mode = 1
        elif label == 'Unwalkable':
            self.draw_mode = 0
    
    def toggle_grid_lines(self, event):
        """Toggle grid lines on/off"""
        self.display_grid_lines = not self.display_grid_lines
        self.update_display()
    
    def increase_brush_size(self, event):
        """Increase the brush size"""
        self.grid_size = min(20, self.grid_size + 1)
        print(f"Brush size increased to {self.grid_size}")
    
    def decrease_brush_size(self, event):
        """Decrease the brush size"""
        self.grid_size = max(1, self.grid_size - 1)
        print(f"Brush size decreased to {self.grid_size}")
    
    def save_grid(self, event):
        """Save the grid to a file"""
        # Save only the binary grid (0=unwalkable, 1=walkable)
        # If there's a visualization grid with spawn regions, ignore it for saving
        filename = "edited_grid.npy"
        np.save(filename, self.analyzer.grid)
        print(f"Grid matrix saved to {filename} with shape {self.analyzer.grid.shape}")
        print(f"Matrix contains only values 0 (unwalkable) and 1 (walkable)")
        
        # Save the coordinate offset information
        if self.offset_x > 0 or self.offset_y > 0:
            metadata = {
                'original_width': self.original_width,
                'original_height': self.original_height,
                'offset_x': self.offset_x,
                'offset_y': self.offset_y,
                'world_editor_offset_x': self.world_editor_offset_x,
                'world_editor_offset_y': self.world_editor_offset_y
            }
            np.save("grid_metadata.npy", metadata)
            print(f"Grid metadata saved to grid_metadata.npy")
            print(f"Grid offset: ({self.offset_x}, {self.offset_y})")
            print(f"World Editor offset: ({self.world_editor_offset_x}, {self.world_editor_offset_y})")
            print(f"Original dimensions: {self.original_width}x{self.original_height}")
            print(f"New dimensions: {self.analyzer.grid.shape[1]}x{self.analyzer.grid.shape[0]}")
    
    def import_spawn_regions(self, event):
        """Import spawn regions from predefined coordinates"""
        # Clear any existing spawn regions
        self.spawn_regions = []
        
        # Define spawn regions using bottom-left (x1, y1) and top-right (x2, y2) corner coordinates
        # These are the EXACT coordinates shown when clicking on the map in the grid editor
        # Format: (island_id, region_name, ((x1, y1), (x2, y2)))
        
        # Northwest Island (Blue Herbs)
        # Bottom-left corner at (41, 142) based on actual clicking
        self.spawn_regions.append(("NW", "1-1", ((41, 142), (82, 170))))
        self.spawn_regions.append(("NW", "1-2", ((83, 142), (98, 160))))
        self.spawn_regions.append(("NW", "1-3", ((98, 142), (108, 152))))
        
        # Northeast Island (Orange Herbs)
        self.spawn_regions.append(("NE", "2-1", ((113, 113), (149, 142))))
        self.spawn_regions.append(("NE", "2-2", ((127, 127), (150, 142))))
        self.spawn_regions.append(("NE", "2-3", ((148, 148), (187, 170))))
        
        # Southeast Island (Yellow Herbs)
        self.spawn_regions.append(("SE", "3-3", ((134, 103), (148, 117))))
        self.spawn_regions.append(("SE", "3-1", ((148, 17), (189, 60))))
        self.spawn_regions.append(("SE", "3-2", ((122, 70), (144, 90))))
        
        # Southwest Island (Purple Herbs)
        self.spawn_regions.append(("SW", "4-1", ((35, 35), (113, 113))))
        self.spawn_regions.append(("SW", "4-2", ((120, 70), (137, 95))))
        self.spawn_regions.append(("SW", "4-3", ((100, 50), (120, 70))))
        
        # Adjust spawn regions from World Editor coordinates to our grid coordinates
        for i in range(len(self.spawn_regions)):
            island, name, ((x1, y1), (x2, y2)) = self.spawn_regions[i]
            # Subtract the World Editor offset
            grid_x1 = x1 - self.world_editor_offset_x
            grid_y1 = y1 - self.world_editor_offset_y
            grid_x2 = x2 - self.world_editor_offset_x
            grid_y2 = y2 - self.world_editor_offset_y
            
            # Store in format that works with the rest of the code: (x, y, width, height)
            x = grid_x1
            y = grid_y1
            width = grid_x2 - grid_x1
            height = grid_y2 - grid_y1
            self.spawn_regions[i] = (island, name, (x, y, width, height))
        
        print(f"Imported {len(self.spawn_regions)} spawn regions. Click 'Highlight Regions' to highlight them on the grid.")
        
        # Visualize the grid with the spawn regions
        self.update_display(highlight_regions=True)
    
    def visualize_islands(self):
        """Draw outlines around islands to help visualize map orientation"""
        # No longer drawing island outlines per user request
        pass
    
    def apply_spawn_regions(self, event):
        """Highlight spawn regions on the grid with purple color"""
        if not self.spawn_regions:
            print("No spawn regions to highlight. Click 'Import Spawn Regions' first.")
            return
        
        # Create a backup of the grid if this is the first application
        if not self.spawn_regions_applied:
            self.original_grid = self.analyzer.grid.copy()
            self.spawn_regions_applied = True
        
        # Restore grid from original state to avoid double-highlighting
        self.analyzer.grid = self.original_grid.copy()
        
        # Create a temporary visualization grid for showing spawn regions
        vis_grid = self.analyzer.grid.copy()
        
        # Mark spawn regions with a special value (2 = purple)
        for region in self.spawn_regions:
            island, name, (x, y, width, height) = region
            
            # Apply coordinate offsets if grid has been trimmed
            adj_x = max(0, x - self.offset_x)
            adj_y = max(0, y - self.offset_y)
            
            # Skip if region is completely outside the grid
            if adj_x >= vis_grid.shape[1] or adj_y >= vis_grid.shape[0]:
                print(f"Warning: Region {island}-{name} is outside the trimmed grid. Skipping.")
                continue
            
            # Ensure width and height stay within grid boundaries
            width = min(width, vis_grid.shape[1] - adj_x)
            height = min(height, vis_grid.shape[0] - adj_y)
            
            if width <= 0 or height <= 0:
                print(f"Warning: Region {island}-{name} has invalid dimensions after trimming. Skipping.")
                continue
            
            # Apply the region (only to cells that are walkable)
            for r in range(adj_y, adj_y + height):
                for c in range(adj_x, adj_x + width):
                    if 0 <= r < vis_grid.shape[0] and 0 <= c < vis_grid.shape[1]:
                        if vis_grid[r, c] == 1:  # Only highlight walkable areas
                            vis_grid[r, c] = 2  # Mark as spawn region in visualization
        
        # Update the display with the visualization grid
        self.vis_grid = vis_grid
        print(f"Highlighted spawn regions on the grid.")
        self.update_display(use_vis_grid=True)
    
    def clear_spawn_highlights(self, event):
        """Clear spawn region highlights"""
        if self.spawn_regions_applied and hasattr(self, 'original_grid'):
            # Just update the display without visualization
            print("Cleared spawn region highlights.")
            self.update_display()
        else:
            print("No spawn regions have been highlighted.")
    
    def trim_borders(self, event):
        """Trim continuous red (unwalkable) borders around the grid"""
        print("Analyzing grid for trimming...")
        
        grid = self.analyzer.grid.copy()
        height, width = grid.shape
        
        # Find the first row from top that has non-unwalkable cells
        top_trim = 0
        for r in range(height):
            if np.any(grid[r, :] != 0):
                top_trim = r
                break
        
        # Find the first row from bottom that has non-unwalkable cells
        bottom_trim = height - 1
        for r in range(height - 1, -1, -1):
            if np.any(grid[r, :] != 0):
                bottom_trim = r
                break
        
        # Find the first column from left that has non-unwalkable cells
        left_trim = 0
        for c in range(width):
            if np.any(grid[:, c] != 0):
                left_trim = c
                break
        
        # Find the first column from right that has non-unwalkable cells
        right_trim = width - 1
        for c in range(width - 1, -1, -1):
            if np.any(grid[:, c] != 0):
                right_trim = c
                break
        
        # Calculate how much to trim from each side
        trim_top = top_trim
        trim_bottom = height - bottom_trim - 1
        trim_left = left_trim
        trim_right = width - right_trim - 1
        
        # Only trim if there's something to trim
        if trim_top == 0 and trim_bottom == 0 and trim_left == 0 and trim_right == 0:
            print("No unwalkable borders to trim.")
            return
        
        # Trim the grid
        trimmed_grid = grid[top_trim:bottom_trim+1, left_trim:right_trim+1]
        
        # Update the grid
        self.analyzer.grid = trimmed_grid
        
        # Update offsets for coordinate mapping
        self.offset_x += left_trim
        self.offset_y += top_trim
        
        # If spawn regions were applied, update the original grid too
        if self.spawn_regions_applied and hasattr(self, 'original_grid'):
            self.original_grid = self.original_grid[top_trim:bottom_trim+1, left_trim:right_trim+1]
        
        print(f"Grid trimmed: removed {trim_top} rows from top, {trim_bottom} from bottom, {trim_left} from left, {trim_right} from right")
        print(f"New grid shape: {trimmed_grid.shape}")
        print(f"Coordinate offset is now: ({self.offset_x}, {self.offset_y})")
        
        # Update the display
        self.update_display()
    
    def on_click(self, event):
        """Handle mouse clicks on the grid"""
        if event.inaxes != self.ax:
            return
        
        # Record which button was pressed
        self.button_pressed = event.button
        
        # Start drawing only with left button
        if self.button_pressed == 1:  # Left mouse button
            self.is_drawing = True
            
            # Map mouse coordinates to grid coordinates (with 0,0 at bottom left)
            visual_y = int(event.ydata) if event.ydata is not None else None
            visual_x = int(event.xdata) if event.xdata is not None else None
            
            if visual_x is not None and visual_y is not None:
                # Convert to World Editor coordinates by adding offset
                world_x = visual_x + self.offset_x + self.world_editor_offset_x
                # Flip the y-axis to match World Editor's coordinate system (y=0 is at the bottom)
                world_y = self.original_height - 1 - visual_y + self.world_editor_offset_y
                
                # Display coordinates for user reference
                print(f"Click at World Editor coordinates: ({world_x}, {world_y}) [Matrix indices: ({visual_y}, {visual_x})]")
                
                # Draw at the visual position
                self.draw_at_position(event.ydata, event.xdata)
    
    def on_release(self, event):
        """Handle mouse release events to stop drawing"""
        # Stop drawing only if the same button that started drawing is released
        if self.button_pressed == 1:  # Left mouse button
            self.is_drawing = False
            # Force a full update on release
            self.update_display()
        self.button_pressed = None
    
    def on_motion(self, event):
        """Handle mouse motion on the grid"""
        # Only process if left button is being held down
        if not self.is_drawing or self.button_pressed != 1:
            return
        
        # Continue drawing if still pressed and inside the axis
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.draw_at_position(event.ydata, event.xdata)
    
    def draw_at_position(self, y, x):
        """Draw on the grid at the specified position"""
        if y is None or x is None:
            return
            
        current_height, current_width = self.analyzer.grid.shape
        if y < 0 or y >= current_height or x < 0 or x >= current_width:
            return
        
        # Convert to integer grid coordinates (in visual space)
        row, col = int(y), int(x)
        
        # Track whether any changes were made
        changes_made = False
        
        # Apply the brush
        half_size = self.grid_size // 2
        for r in range(max(0, row - half_size), min(current_height, row + half_size + 1)):
            for c in range(max(0, col - half_size), min(current_width, col + half_size + 1)):
                if ((r - row) ** 2 + (c - col) ** 2) <= (half_size ** 2):  # Circular brush
                    # Only modify if different from current mode
                    if self.analyzer.grid[r, c] != self.draw_mode:
                        self.analyzer.grid[r, c] = self.draw_mode
                        changes_made = True
        
        # Only update display occasionally for performance during continuous drawing
        current_time = time.time()
        if changes_made and (current_time - self.last_update_time) > self.update_frequency:
            self.update_display(refresh_only=True)
            self.last_update_time = current_time
    
    def update_display(self, highlight_regions=False, refresh_only=False, use_vis_grid=False):
        """Update the grid display"""
        if not refresh_only:
            # Full redraw
            self.ax.clear()
            
            # Choose which grid to display
            if use_vis_grid and hasattr(self, 'vis_grid'):
                display_grid = self.vis_grid
            else:
                display_grid = self.analyzer.grid
            
            # Display the grid - keep original visual orientation
            self.img = self.ax.imshow(display_grid, cmap=self.cmap, norm=self.norm)
            
            # Display grid lines if enabled
            if self.display_grid_lines:
                # Add gridlines
                self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
                
                current_height, current_width = display_grid.shape
                
                # For better visibility, only show major grid lines
                step = 20
                self.ax.set_xticks(np.arange(-0.5, current_width, step), minor=False)
                self.ax.set_yticks(np.arange(-0.5, current_height, step), minor=False)
                
                # Set x-axis labels with offset (0 at left)
                self.ax.set_xticklabels([str(i + self.offset_x + self.world_editor_offset_x) 
                                         for i in range(0, current_width + 1, step)])
                
                # Set y-axis labels with offset (0 at bottom)
                self.ax.set_yticklabels([str(self.world_editor_offset_y + self.original_height - (i + self.offset_y) - 1) 
                                         for i in range(0, current_height + 1, step)])
                
            # If highlighting spawn regions with rectangles
            if highlight_regions and self.spawn_regions:
                for region in self.spawn_regions:
                    island, name, (x, y, width, height) = region
                    
                    # Apply coordinate offsets if grid has been trimmed
                    adj_x = max(0, x - self.offset_x)
                    adj_y = max(0, y - self.offset_y)
                    
                    # Skip if region is completely outside the grid
                    if adj_x >= self.analyzer.grid.shape[1] or adj_y >= self.analyzer.grid.shape[0]:
                        continue
                    
                    # Ensure width and height stay within grid boundaries
                    width = min(width, self.analyzer.grid.shape[1] - adj_x)
                    height = min(height, self.analyzer.grid.shape[0] - adj_y)
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Determine island-specific color
                    if island == "NW":
                        color = 'blue'
                    elif island == "NE":
                        color = 'orange'
                    elif island == "SE":
                        color = 'yellow'
                    else:  # SW
                        color = 'purple'
                    
                    # Draw a semi-transparent rectangle
                    self.ax.add_patch(plt.Rectangle((adj_x - 0.5, adj_y - 0.5), width, height,
                                                  fill=True, alpha=0.3, color=color,
                                                  linestyle='-', linewidth=2))
                    
                    # Add region name
                    self.ax.text(adj_x + width/2, adj_y + height/2, f"{island}-{name}",
                                ha='center', va='center', color='black',
                                fontsize=12, fontweight='bold')
            
            # Add a title showing current mode and brush size
            current_mode = "Walkable (1)" if self.draw_mode == 1 else "Unwalkable (0)"
            current_height, current_width = display_grid.shape
            self.ax.set_title(f'Grid Editor - {current_width}x{current_height} - Brush Size: {self.grid_size} - Mode: {current_mode}')
        
        else:
            # Partial update - just update the image data
            if use_vis_grid and hasattr(self, 'vis_grid'):
                self.img.set_data(self.vis_grid)
            else:
                self.img.set_data(self.analyzer.grid)
        
        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()  # Process any pending events
    
    def run(self):
        """Run the grid editor"""
        plt.ioff()  # Turn off interactive mode for normal plotting
        plt.show(block=True)

if __name__ == "__main__":
    # Check if a grid file was provided
    if len(sys.argv) > 1:
        grid_file = sys.argv[1]
        print(f"Loading grid file: {grid_file}")
        editor = GridEditor(grid_file)
    else:
        print("No grid file provided. Creating a new grid.")
        editor = GridEditor()
    
    editor.run() 