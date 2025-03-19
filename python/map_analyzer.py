import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# Based on user input, the map is a 190x190 grid
# Bottom-left corner: (17, 17), Top-right corner: (207, 207)
GRID_WIDTH = 190
GRID_HEIGHT = 190
GRID_ORIGIN_X = 17
GRID_ORIGIN_Y = 17

# Define the map grid
# 0 = Unwalkable/Unknown
# 1 = Walkable Land (Black)
# 2 = Shallow Water (Blue)
# 3 = Deep Water/Unwalkable (Purple/Pink)
# 4 = Grid line (White/Yellow)

class MapAnalyzer:
    def __init__(self):
        # Initialize the grid with map boundaries
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        
        # Initialize an empty graph (will be populated later)
        self.graph = None
        
        # Store spawn rectangles (to be filled later)
        self.spawn_rectangles = {
            "NW": [],
            "NE": [],
            "SE": [],
            "SW": []
        }
        
        # Initialize the grid with map boundaries
        # The map is from -12672 to 11648 in both x and y directions
        self.origin_x = 12672
        self.origin_y = 12672
        
        # Define spawn rectangles
        # These would be extracted from the game, but for now we'll hard-code examples
        self.spawn_rectangles = {
            "NW": [
                {"name": "spawn_area_1_1", "coords": (-5000, 2000, 1500, 1200), "weight": 65.4},
                {"name": "spawn_area_1_2", "coords": (-7000, 3000, 800, 900), "weight": 21.6},
                {"name": "spawn_area_1_3", "coords": (-6000, 5000, 600, 700), "weight": 13.1}
            ],
            "NE": [
                {"name": "spawn_area_2_1", "coords": (3000, 3000, 1000, 1000), "weight": 30.9},
                {"name": "spawn_area_2_2", "coords": (5000, 4000, 1000, 1000), "weight": 30.9},
                {"name": "spawn_area_2_3", "coords": (4000, 6000, 1200, 1000), "weight": 38.2}
            ],
            "SE": [
                {"name": "spawn_area_3_1", "coords": (5000, -5000, 1500, 1200), "weight": 63.3},
                {"name": "spawn_area_3_2", "coords": (3000, -3000, 500, 400), "weight": 6.3},
                {"name": "spawn_area_3_3", "coords": (4000, -4000, 1000, 900), "weight": 30.4}
            ],
            "SW": [
                {"name": "spawn_area_4_1", "coords": (-5000, -5000, 2000, 1800), "weight": 90.9},
                {"name": "spawn_area_4_2", "coords": (-3000, -4000, 400, 300), "weight": 3.2},
                {"name": "spawn_area_4_3", "coords": (-4000, -3000, 500, 600), "weight": 5.8}
            ]
        }
        
        # Restricted areas (thief bush cliffs, etc.)
        self.restricted_areas = [
            {"name": "Thief_Bush_Cliff_NW", "coords": (-6500, 5500, 300, 300)},
            {"name": "Thief_Bush_Cliff_NE", "coords": (6500, 6000, 300, 300)},
            {"name": "Thief_Bush_Cliff_SE", "coords": (5500, -6000, 300, 300)},
            {"name": "Thief_Bush_Cliff_SW_1", "coords": (-4500, -5500, 300, 300)},
            {"name": "Thief_Bush_Cliff_SW_2", "coords": (-5500, -4500, 300, 300)}
        ]
        
        # Special areas (bushes, waterfall entrances, etc.)
        self.special_areas = [
            {"name": "Thief_Bush_NW_A_In", "coords": (-6300, 5300, 100, 100)},
            {"name": "Thief_Bush_NE_A_In", "coords": (6300, 5800, 100, 100)},
            {"name": "Thief_Bush_SE_A_In", "coords": (5300, -5800, 100, 100)},
            {"name": "Thief_Bush_SW_A_In", "coords": (-4300, -5300, 100, 100)},
            {"name": "Thief_Bush_SW_C_In", "coords": (-5300, -4300, 100, 100)},
            {"name": "Oasis_waterfall_In", "coords": (-4000, 4000, 150, 150)},
            {"name": "Pantsu_Waterfall_Left_In", "coords": (4000, -4000, 150, 150)},
            {"name": "Pantsu_Waterfall_Right_In", "coords": (4200, -4000, 150, 150)}
        ]
        
    def game_to_grid(self, x, y):
        """Convert game coordinates to grid coordinates"""
        grid_x = int((x + self.origin_x) / RESOLUTION)
        grid_y = int((y + self.origin_y) / RESOLUTION)
        return grid_x, grid_y
    
    def fill_rectangle(self, x1, y1, width, height, value):
        """Fill a rectangle in the grid with the specified value"""
        gx1, gy1 = self.game_to_grid(x1, y1)
        gx2, gy2 = self.game_to_grid(x1 + width, y1 + height)
        self.grid[gy1:gy2, gx1:gx2] = value
        
    def fill_spawn_areas(self):
        """Fill spawn areas in the grid"""
        # NW Island (value = 4)
        for rect in self.spawn_rectangles["NW"]:
            x, y, w, h = rect["coords"]
            self.fill_rectangle(x, y, w, h, 4)
            
        # NE Island (value = 5)
        for rect in self.spawn_rectangles["NE"]:
            x, y, w, h = rect["coords"]
            self.fill_rectangle(x, y, w, h, 5)
            
        # SE Island (value = 6)
        for rect in self.spawn_rectangles["SE"]:
            x, y, w, h = rect["coords"]
            self.fill_rectangle(x, y, w, h, 6)
            
        # SW Island (value = 7)
        for rect in self.spawn_rectangles["SW"]:
            x, y, w, h = rect["coords"]
            self.fill_rectangle(x, y, w, h, 7)
    
    def fill_restricted_areas(self):
        """Fill restricted areas in the grid"""
        for area in self.restricted_areas:
            x, y, w, h = area["coords"]
            self.fill_rectangle(x, y, w, h, 8)
    
    def fill_special_areas(self):
        """Fill special areas in the grid"""
        for area in self.special_areas:
            x, y, w, h = area["coords"]
            self.fill_rectangle(x, y, w, h, 9)
    
    def simulate_land_water(self):
        """
        Simulate land and water distribution
        In a real implementation, this would use the extracted walkability data
        """
        # Create islands
        # NW Island
        center_x, center_y = -5000, 4000
        radius = 3000
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                game_x = (x * RESOLUTION) - self.origin_x
                game_y = (y * RESOLUTION) - self.origin_y
                dist = np.sqrt((game_x - center_x)**2 + (game_y - center_y)**2)
                if dist < radius:
                    self.grid[y, x] = max(self.grid[y, x], 1)  # Land
                    
        # NE Island
        center_x, center_y = 5000, 4000
        radius = 3000
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                game_x = (x * RESOLUTION) - self.origin_x
                game_y = (y * RESOLUTION) - self.origin_y
                dist = np.sqrt((game_x - center_x)**2 + (game_y - center_y)**2)
                if dist < radius:
                    self.grid[y, x] = max(self.grid[y, x], 1)  # Land
        
        # SE Island
        center_x, center_y = 5000, -4000
        radius = 3000
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                game_x = (x * RESOLUTION) - self.origin_x
                game_y = (y * RESOLUTION) - self.origin_y
                dist = np.sqrt((game_x - center_x)**2 + (game_y - center_y)**2)
                if dist < radius:
                    self.grid[y, x] = max(self.grid[y, x], 1)  # Land
        
        # SW Island
        center_x, center_y = -5000, -4000
        radius = 3500  # Larger island
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                game_x = (x * RESOLUTION) - self.origin_x
                game_y = (y * RESOLUTION) - self.origin_y
                dist = np.sqrt((game_x - center_x)**2 + (game_y - center_y)**2)
                if dist < radius:
                    self.grid[y, x] = max(self.grid[y, x], 1)  # Land
                    
        # Add water around
        # Everything that's not land is deep water
        water_mask = (self.grid == 0)
        self.grid[water_mask] = 3
        
        # Add shallow water around islands
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y, x] == 3:  # Deep water
                    # Check if there's land nearby
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                                if self.grid[ny, nx] == 1:  # Land
                                    self.grid[y, x] = 2  # Shallow water
                                    break
    
    def simulate_trees(self, tree_density=0.2):
        """
        Simulate trees on land
        In a real implementation, this would use actual tree data
        """
        land_mask = (self.grid == 1)
        tree_count = int(np.sum(land_mask) * tree_density)
        land_indices = np.where(land_mask)
        land_coords = list(zip(land_indices[0], land_indices[1]))
        
        # Randomly select tree_count coordinates
        tree_coords = np.random.choice(len(land_coords), tree_count, replace=False)
        for idx in tree_coords:
            y, x = land_coords[idx]
            self.grid[y, x] = 9  # Tree
    
    def build_graph(self):
        """
        Build a graph for pathfinding based on the normalized grid
        Nodes are walkable cells, edges connect neighboring walkable cells
        """
        import networkx as nx
        
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return
            
        # Reset the graph
        self.graph = nx.Graph()
        
        # Define walkable terrain (land and shallow water)
        walkable = (self.grid == 1) | (self.grid == 2)
        
        # Create nodes for all walkable cells
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if walkable[y, x]:
                    self.graph.add_node((y, x))
        
        # Connect neighboring walkable cells
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if walkable[y, x]:
                    # Check all 8 neighboring cells
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH and walkable[ny, nx]:
                                # Diagonal neighbors have sqrt(2) distance
                                weight = 1.414 if dy != 0 and dx != 0 else 1.0
                                
                                # Land-to-water transitions have higher weight
                                if self.grid[y, x] == 1 and self.grid[ny, nx] == 2:
                                    weight *= 1.5  # Penalty for entering water
                                elif self.grid[y, x] == 2 and self.grid[ny, nx] == 1:
                                    weight *= 1.2  # Smaller penalty for exiting water
                                    
                                self.graph.add_edge((y, x), (ny, nx), weight=weight)
        
        print(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        
    def find_path(self, start, goal):
        """
        Find the shortest path between two points in grid coordinates.
        start and goal are in grid coordinates (y, x) where y is row, x is column.
        """
        import networkx as nx
        
        if not hasattr(self, 'graph') or len(self.graph) == 0:
            print("Graph not built. Run build_graph first.")
            return None
        
        # Check if start and goal are in the graph
        if start not in self.graph:
            print(f"Start point {start} is not walkable.")
            return None
        
        if goal not in self.graph:
            print(f"Goal point {goal} is not walkable.")
            return None
        
        try:
            path = nx.shortest_path(self.graph, start, goal, weight='weight')
            return path
        except nx.NetworkXNoPath:
            print(f"No path found between {start} and {goal}")
            return None
    
    def grid_to_game_coords(self, grid_y, grid_x):
        """Convert grid coordinates to game coordinates"""
        game_x = grid_x + GRID_ORIGIN_X
        game_y = grid_y + GRID_ORIGIN_Y
        return (game_x, game_y)
    
    def game_to_grid_coords(self, game_x, game_y):
        """Convert game coordinates to grid coordinates"""
        grid_x = game_x - GRID_ORIGIN_X
        grid_y = game_y - GRID_ORIGIN_Y
        return (grid_y, grid_x)  # Note: y is row, x is column
    
    def find_valid_walkable_points(self, num_points=2):
        """
        Find valid walkable points in the grid for pathfinding.
        Returns a list of points in grid coordinates (y, x).
        """
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return []
            
        # Find walkable cells (land and shallow water)
        walkable_indices = np.where((self.grid == 1) | (self.grid == 2))
        walkable_coords = list(zip(walkable_indices[0], walkable_indices[1]))
        
        if len(walkable_coords) == 0:
            print("No walkable areas found in the grid.")
            return []
            
        # Randomly select num_points coordinate pairs
        selected_indices = np.random.choice(len(walkable_coords), min(num_points, len(walkable_coords)), replace=False)
        selected_coords = [walkable_coords[i] for i in selected_indices]
            
        return selected_coords
    
    def visualize_path(self, path, output_path="path_visualization.png"):
        """Visualize a path on the grid"""
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return
        
        if path is None or len(path) == 0:
            print("No path to visualize.")
            return
        
        plt.figure(figsize=(12, 12))
        
        # Define colors for each cell type
        colors = {
            0: 'white',      # Unknown/grid
            1: 'green',      # Land
            2: 'blue',       # Shallow water 
            3: 'red',        # Unwalkable
            4: 'yellow'      # Grid line
        }
        
        # Create a color map
        cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(5)])
        bounds = range(6)
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the grid
        plt.imshow(self.grid, cmap=cmap, norm=norm)
        
        # Add gridlines
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, GRID_WIDTH, 20), [str(i + GRID_ORIGIN_X) for i in range(0, GRID_WIDTH + 1, 20)])
        plt.yticks(np.arange(-0.5, GRID_HEIGHT, 20), [str(i + GRID_ORIGIN_Y) for i in range(0, GRID_HEIGHT + 1, 20)])
        
        # Plot the path
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2)
        
        # Mark start and end points
        plt.plot(path_x[0], path_y[0], 'go', markersize=10)  # Start point
        plt.plot(path_x[-1], path_y[-1], 'ro', markersize=10)  # End point
        
        # Add labels
        plt.title(f'Path Visualization ({len(path)} steps)')
        
        # Save the figure
        plt.savefig(output_path, dpi=150)
        print(f"Path visualization saved to {output_path}")
    
    def visualize_map(self, file_name="map_analysis.png"):
        """Visualize the map grid"""
        plt.figure(figsize=(20, 20))
        
        # Define colors for each cell type
        colors = {
            0: 'black',       # Unwalkable/Unknown
            1: 'green',       # Walkable Land
            2: 'lightblue',   # Shallow Water
            3: 'blue',        # Deep Water
            4: 'yellow',      # NW Island Spawn
            5: 'orange',      # NE Island Spawn
            6: 'pink',        # SE Island Spawn
            7: 'red',         # SW Island Spawn
            8: 'purple',      # Restricted spawn area
            9: 'darkgreen'    # Trees/obstacles
        }
        
        # Create a color map
        cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(10)])
        bounds = range(11)
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the grid
        plt.imshow(self.grid, cmap=cmap, norm=norm)
        
        # Add labels
        plt.title('Map Analysis')
        plt.colorbar(ticks=range(10), label='Terrain Type')
        
        # Save the figure
        plt.savefig(file_name, dpi=300)
        print(f"Map visualization saved to {file_name}")
    
    def analyze_resource_density(self):
        """
        Analyze the resource density in each spawn rectangle
        In a real implementation, this would calculate actual spawn probabilities
        """
        results = []
        
        for island, rectangles in self.spawn_rectangles.items():
            for rect in rectangles:
                x, y, w, h = rect["coords"]
                area = w * h
                
                # Calculate walkable area percentage within this rectangle
                gx1, gy1 = self.game_to_grid(x, y)
                gx2, gy2 = self.game_to_grid(x + w, y + h)
                
                walkable_cells = np.sum((self.grid[gy1:gy2, gx1:gx2] == 1) | 
                                       (self.grid[gy1:gy2, gx1:gx2] == 4) |
                                       (self.grid[gy1:gy2, gx1:gx2] == 5) |
                                       (self.grid[gy1:gy2, gx1:gx2] == 6) |
                                       (self.grid[gy1:gy2, gx1:gx2] == 7))
                total_cells = (gy2 - gy1) * (gx2 - gx1)
                walkable_percentage = walkable_cells / total_cells if total_cells > 0 else 0
                
                # Calculate trees per walkable area
                tree_cells = np.sum(self.grid[gy1:gy2, gx1:gx2] == 9)
                trees_per_area = tree_cells / walkable_cells if walkable_cells > 0 else 0
                
                # In a real implementation, we would calculate actual spawn rates
                # For now, we'll use the rectangle weight as a proxy
                weight = rect["weight"]
                
                results.append({
                    "island": island,
                    "rectangle": rect["name"],
                    "area": area,
                    "weight": weight,
                    "walkable_percentage": walkable_percentage * 100,
                    "trees_per_area": trees_per_area
                })
        
        return results
    
    def run_analysis(self):
        """Run the complete analysis"""
        # Fill in the map data
        self.simulate_land_water()
        self.fill_spawn_areas()
        self.fill_restricted_areas()
        self.fill_special_areas()
        self.simulate_trees()
        
        # Build the graph for pathfinding
        self.build_graph()
        
        # Visualize the map
        self.visualize_map()
        
        # Analyze resource density
        density_results = self.analyze_resource_density()
        
        # Print resource density analysis
        print("\nResource Density Analysis:")
        print("=========================")
        for result in density_results:
            print(f"Island: {result['island']}, Rectangle: {result['rectangle']}")
            print(f"  Area: {result['area']} square units")
            print(f"  Weight: {result['weight']}%")
            print(f"  Walkable Percentage: {result['walkable_percentage']:.2f}%")
            print(f"  Trees per Walkable Area: {result['trees_per_area']:.4f}")
            print()
        
        # Example pathfinding
        print("\nPathfinding Examples:")
        print("====================")
        # Path from one point in NW island to another point in NW island
        print("Testing path within same island:")
        start = (-5000, 4000)
        goal = (-6000, 3000)
        path = self.find_path(start, goal)
        if path:
            print(f"Path within NW island found with {len(path)} steps")
        else:
            print("No path found within NW island")
            
        # Try to find path between islands - will fail because islands are separated by water
        print("\nTesting paths between islands (should fail because of water):")
        start = (-5000, 4000)
        goal = (5000, 4000)  
        path = self.find_path(start, goal)
        if path:
            print(f"Path from NW to NE island found with {len(path)} steps")
        else:
            print("No path found from NW to NE island (expected)")
        
        return density_results

    def clean_stitched_image(self, input_image_path, roi, output_image_path):
        """Clean the stitched map image by masking out the undesired region (ROI) and filling it with white.
        roi is a tuple (x, y, width, height) in pixels."""
        img = cv2.imread(input_image_path)
        if img is None:
            print(f"Error: Could not load image from {input_image_path}")
            return
        x, y, w, h = roi
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), thickness=-1)
        cv2.imwrite(output_image_path, img)
        print(f"Cleaned image saved to {output_image_path}")

    def extract_map_from_image(self, image_path, output_path="extracted_map.png"):
        """
        Extract walkable and non-walkable areas from the provided grid map image.
        
        Parameters:
            image_path: Path to the input image
            output_path: Path to save the processed binary map
            
        Returns:
            A 2D numpy array representing the walkable (1) and non-walkable (0) areas
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
            
        print(f"Loaded image with shape: {img.shape}")
        height, width = img.shape[:2]
        
        # Save the original image dimensions
        self.original_image_height = height
        self.original_image_width = width
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks for different colors
        # Purple/Pink (non-walkable)
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Blue (shallow water)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Black/Green (land)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Yellow/White (grid lines)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine grid masks
        grid_mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Create an RGB visualization
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)
        colored_map[black_mask > 0] = [0, 255, 0]    # Green for land
        colored_map[blue_mask > 0] = [255, 0, 0]     # Blue for shallow water
        colored_map[purple_mask > 0] = [0, 0, 255]   # Red for unwalkable
        colored_map[grid_mask > 0] = [255, 255, 255] # White for grid
        
        # Save the visualization
        cv2.imwrite(output_path, colored_map)
        print(f"Extracted map saved to {output_path}")
        
        # Detect the grid pattern to create the normalized grid
        # Store the masks for later processing
        self.land_mask = black_mask
        self.water_mask = blue_mask
        self.unwalkable_mask = purple_mask
        self.grid_mask = grid_mask
        
        return colored_map
    
    def create_grid_matrix(self):
        """Create a normalized 190x190 grid matrix from the image masks"""
        if not hasattr(self, 'land_mask') or not hasattr(self, 'water_mask'):
            print("Image masks not available. Run extract_map_from_image first.")
            return
            
        # Create a new grid of the exact dimensions we want
        grid_matrix = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        
        # Calculate the cell size in pixels
        cell_height = self.original_image_height / GRID_HEIGHT
        cell_width = self.original_image_width / GRID_WIDTH
        
        # For each cell in our target grid, determine the predominant terrain type
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Calculate the pixel region this cell corresponds to
                start_y = int(y * cell_height)
                end_y = int((y + 1) * cell_height)
                start_x = int(x * cell_width)
                end_x = int((x + 1) * cell_width)
                
                # Get the corresponding regions in the masks
                land_region = self.land_mask[start_y:end_y, start_x:end_x]
                water_region = self.water_mask[start_y:end_y, start_x:end_x]
                unwalkable_region = self.unwalkable_mask[start_y:end_y, start_x:end_x]
                
                # Calculate the percentage of each terrain type
                land_percent = np.sum(land_region > 0) / land_region.size if land_region.size > 0 else 0
                water_percent = np.sum(water_region > 0) / water_region.size if water_region.size > 0 else 0
                unwalkable_percent = np.sum(unwalkable_region > 0) / unwalkable_region.size if unwalkable_region.size > 0 else 0
                
                # Determine the predominant terrain type
                if land_percent > 0.5:
                    grid_matrix[y, x] = 1  # Land
                elif water_percent > 0.5:
                    grid_matrix[y, x] = 2  # Shallow water
                elif unwalkable_percent > 0.5:
                    grid_matrix[y, x] = 3  # Unwalkable
                else:
                    # If no clear terrain type, default to unwalkable
                    grid_matrix[y, x] = 3  # Unwalkable
        
        # Store the normalized grid
        self.grid = grid_matrix
        print(f"Created normalized grid matrix with shape: {grid_matrix.shape}")
        
        return grid_matrix
    
    def visualize_grid_matrix(self, output_path="normalized_grid.png", unify_playable=True):
        """
        Visualize the normalized grid matrix
        
        Parameters:
            output_path: Path to save the visualization
            unify_playable: If True, color all playable areas (land and shallow water) as green
        """
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return
            
        plt.figure(figsize=(12, 12))
        
        # Define colors for each cell type
        if unify_playable:
            # Use the same color for land and shallow water if unify_playable is True
            colors = {
                0: 'red',        # Unknown/grid -> mark as unwalkable (red)
                1: 'green',      # Land (playable)
                2: 'green',      # Shallow water (playable)
                3: 'red',        # Unwalkable/Deep water
                4: 'yellow'      # Grid line
            }
        else:
            colors = {
                0: 'white',      # Unknown/grid
                1: 'green',      # Land
                2: 'blue',       # Shallow water 
                3: 'red',        # Unwalkable
                4: 'yellow'      # Grid line
            }
        
        # Create a color map
        cmap = plt.matplotlib.colors.ListedColormap([colors[i] for i in range(5)])
        bounds = range(6)
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create a copy of the grid to visualize
        viz_grid = self.grid.copy()
        
        # Mark all unclassified cells (0) as unwalkable (3)
        viz_grid[viz_grid == 0] = 3
        
        # Plot the grid
        plt.imshow(viz_grid, cmap=cmap, norm=norm)
        
        # Add gridlines
        plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, GRID_WIDTH, 20), [str(i + GRID_ORIGIN_X) for i in range(0, GRID_WIDTH + 1, 20)])
        plt.yticks(np.arange(-0.5, GRID_HEIGHT, 20), [str(i + GRID_ORIGIN_Y) for i in range(0, GRID_HEIGHT + 1, 20)])
        
        # Add labels
        plt.title(f'Normalized Grid Matrix ({GRID_WIDTH}x{GRID_HEIGHT})')
        
        # Create a custom legend
        from matplotlib.patches import Patch
        legend_elements = []
        if unify_playable:
            legend_elements = [
                Patch(facecolor='green', edgecolor='black', label='Playable (Land + Shallow Water)'),
                Patch(facecolor='red', edgecolor='black', label='Unwalkable')
            ]
        else:
            legend_elements = [
                Patch(facecolor='green', edgecolor='black', label='Land'),
                Patch(facecolor='blue', edgecolor='black', label='Shallow Water'),
                Patch(facecolor='red', edgecolor='black', label='Unwalkable'),
                Patch(facecolor='white', edgecolor='black', label='Unknown/Grid')
            ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Save the figure
        plt.savefig(output_path, dpi=150)
        print(f"Normalized grid visualization saved to {output_path}")
        
    def save_grid_to_file(self, output_path="grid_matrix.npy"):
        """Save the current grid matrix to a file for later editing"""
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return
            
        np.save(output_path, self.grid)
        print(f"Grid matrix saved to {output_path}")
        
    def load_grid_from_file(self, input_path="grid_matrix.npy"):
        """Load a grid matrix from a file"""
        try:
            self.grid = np.load(input_path)
            print(f"Grid matrix loaded from {input_path} with shape {self.grid.shape}")
            return True
        except Exception as e:
            print(f"Error loading grid matrix: {e}")
            return False
            
    def edit_grid_cell(self, row, col, value):
        """
        Edit a single cell in the grid matrix
        
        Parameters:
            row: Row index (0-189)
            col: Column index (0-189)
            value: New cell value (1=land, 2=shallow water, 3=unwalkable)
        """
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return False
            
        if row < 0 or row >= GRID_HEIGHT or col < 0 or col >= GRID_WIDTH:
            print(f"Invalid cell coordinates: ({row}, {col})")
            return False
            
        if value not in [1, 2, 3]:
            print(f"Invalid cell value: {value}. Must be 1 (land), 2 (shallow water), or 3 (unwalkable)")
            return False
            
        self.grid[row, col] = value
        print(f"Cell ({row}, {col}) updated to value {value}")
        return True
        
    def edit_grid_region(self, start_row, start_col, end_row, end_col, value):
        """
        Edit a rectangular region in the grid matrix
        
        Parameters:
            start_row: Starting row index
            start_col: Starting column index
            end_row: Ending row index (inclusive)
            end_col: Ending column index (inclusive)
            value: New cell value (1=land, 2=shallow water, 3=unwalkable)
        """
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return False
            
        if (start_row < 0 or start_row >= GRID_HEIGHT or 
            start_col < 0 or start_col >= GRID_WIDTH or
            end_row < 0 or end_row >= GRID_HEIGHT or
            end_col < 0 or end_col >= GRID_WIDTH):
            print(f"Invalid region coordinates: ({start_row}, {start_col}) to ({end_row}, {end_col})")
            return False
            
        if value not in [1, 2, 3]:
            print(f"Invalid cell value: {value}. Must be 1 (land), 2 (shallow water), or 3 (unwalkable)")
            return False
            
        # Ensure start <= end
        start_row, end_row = min(start_row, end_row), max(start_row, end_row)
        start_col, end_col = min(start_col, end_col), max(start_col, end_col)
        
        # Update the region
        self.grid[start_row:end_row+1, start_col:end_col+1] = value
        
        num_cells = (end_row - start_row + 1) * (end_col - start_col + 1)
        print(f"Region from ({start_row}, {start_col}) to ({end_row}, {end_col}) updated to value {value} ({num_cells} cells)")
        return True
        
    def apply_spawn_region(self, island, name, coords, is_walkable=True):
        """
        Apply a spawn region to the grid
        
        Parameters:
            island: Island identifier ("NW", "NE", "SE", "SW")
            name: Region name
            coords: (row_start, col_start, width, height) in grid coordinates
            is_walkable: Whether the region should be marked as walkable
        """
        if not hasattr(self, 'grid'):
            print("Grid not initialized. Run create_grid_matrix first.")
            return False
            
        if island not in ["NW", "NE", "SE", "SW"]:
            print(f"Invalid island: {island}")
            return False
            
        row_start, col_start, width, height = coords
        
        # Calculate region boundaries
        row_end = row_start + height - 1
        col_end = col_start + width - 1
        
        # Check if the region is within grid bounds
        if (row_start < 0 or row_start >= GRID_HEIGHT or 
            col_start < 0 or col_start >= GRID_WIDTH or
            row_end < 0 or row_end >= GRID_HEIGHT or
            col_end < 0 or col_end >= GRID_WIDTH):
            print(f"Invalid region coordinates: ({row_start}, {col_start}) to ({row_end}, {col_end})")
            return False
            
        # Mark the region as walkable if specified
        if is_walkable:
            # Create a mask for the region
            mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
            mask[row_start:row_end+1, col_start:col_end+1] = True
            
            # Only update cells that are currently unwalkable (value 3)
            update_mask = mask & (self.grid == 3)
            self.grid[update_mask] = 1  # Mark as land
        
        # Add the region to spawn rectangles
        self.spawn_rectangles[island].append({
            "name": name,
            "coords": (row_start, col_start, width, height),
            "is_walkable": is_walkable
        })
        
        print(f"Spawn region {name} added to {island} island at ({row_start}, {col_start}) with size {width}x{height}")
        return True


if __name__ == "__main__":
    import sys
    
    print("Island Troll Tribes Map Analysis Tool")
    print("====================================")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\nUsage:")
            print("  python map_analyzer.py [options] [map_image.png]")
            print("\nOptions:")
            print("  --help, -h            Show this help message")
            print("  --edit <grid_file>    Edit an existing grid matrix")
            print("  --view <grid_file>    View an existing grid matrix")
            print("\nExamples:")
            print("  python map_analyzer.py map_image.png          # Process a new map image")
            print("  python map_analyzer.py --edit grid_matrix.npy  # Edit an existing grid")
            print("  python map_analyzer.py --view grid_matrix.npy  # View an existing grid")
            sys.exit(0)
        
        # Check if we're editing an existing grid
        if sys.argv[1] == "--edit" and len(sys.argv) > 2:
            print(f"Editing grid file: {sys.argv[2]}")
            analyzer = MapAnalyzer()
            if analyzer.load_grid_from_file(sys.argv[2]):
                # Example: Edit a specific region
                print("\nExample: Editing a region to make it walkable")
                
                # Edit a region (e.g., make a 5x5 region walkable)
                # In practice, you'd use user input here
                analyzer.edit_grid_region(50, 50, 55, 55, 1)  # 1 = land (walkable)
                
                # Visualize the edited grid
                analyzer.visualize_grid_matrix("edited_grid.png")
                
                # Save the edited grid
                analyzer.save_grid_to_file("edited_" + sys.argv[2])
                
                print("\nGrid editing complete! Edited grid saved to 'edited_" + sys.argv[2] + "'")
                print("Visualization saved to 'edited_grid.png'")
            sys.exit(0)
            
        # Check if we're viewing an existing grid
        if sys.argv[1] == "--view" and len(sys.argv) > 2:
            print(f"Viewing grid file: {sys.argv[2]}")
            analyzer = MapAnalyzer()
            if analyzer.load_grid_from_file(sys.argv[2]):
                # Visualize the grid
                analyzer.visualize_grid_matrix("viewed_grid.png")
                print("\nVisualization saved to 'viewed_grid.png'")
            sys.exit(0)
        
        # Assume it's an image file
        image_path = sys.argv[1]
        print(f"Using map image: {image_path}")
        use_image = True
    else:
        print("No map image provided. Using simulated map data.")
        print("To use a map image, run: python map_analyzer.py path_to_image.png")
        print("For help, run: python map_analyzer.py --help")
        use_image = False
    
    analyzer = MapAnalyzer()
    
    if use_image:
        # Process the map image
        print("\nProcessing map image...")
        analyzer.extract_map_from_image(image_path)
        
        # Create the normalized grid matrix
        print("\nCreating normalized 190x190 grid...")
        analyzer.create_grid_matrix()
        
        # Visualize the normalized grid
        print("\nVisualizing grid...")
        # Save both visualizations - one with unified playable areas and one with distinct terrain types
        analyzer.visualize_grid_matrix("normalized_grid_unified.png", unify_playable=True)
        analyzer.visualize_grid_matrix("normalized_grid_detailed.png", unify_playable=False)
        
        # Save the grid matrix for later editing
        analyzer.save_grid_to_file("grid_matrix.npy")
        print("\nGrid matrix saved to 'grid_matrix.npy'. You can edit it with:")
        print("  python map_analyzer.py --edit grid_matrix.npy")
        
        # Build the graph for pathfinding
        print("\nBuilding pathfinding graph...")
        analyzer.build_graph()
        
        # Find valid walkable points for pathfinding example
        valid_points = analyzer.find_valid_walkable_points(5)
        
        if len(valid_points) >= 2:
            print("\nPathfinding Example:")
            print("===================")
            
            # Select two points that should be on different islands
            # Find points that are far apart
            max_distance = 0
            start = None
            goal = None
            
            for i in range(len(valid_points)):
                for j in range(i+1, len(valid_points)):
                    p1 = valid_points[i]
                    p2 = valid_points[j]
                    dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                    if dist > max_distance:
                        max_distance = dist
                        start = p1
                        goal = p2
            
            if start and goal:
                # Convert to game coordinates for display
                start_game = analyzer.grid_to_game_coords(start[0], start[1])
                goal_game = analyzer.grid_to_game_coords(goal[0], goal[1])
                
                print(f"Finding path from grid: {start} (game: {start_game})")
                print(f"                to grid: {goal} (game: {goal_game})")
                
                path = analyzer.find_path(start, goal)
                
                if path:
                    print(f"Path found with {len(path)} steps")
                    # Visualize the path
                    analyzer.visualize_path(path)
                    print("Path visualization saved to path_visualization.png")
                else:
                    print("No path found - trying another pair of points")
                    
                    # Try with points that are closer
                    for i in range(len(valid_points)):
                        if valid_points[i] != start:
                            new_goal = valid_points[i]
                            new_goal_game = analyzer.grid_to_game_coords(new_goal[0], new_goal[1])
                            print(f"\nTrying with goal: {new_goal} (game: {new_goal_game})")
                            
                            path = analyzer.find_path(start, new_goal)
                            if path:
                                print(f"Path found with {len(path)} steps")
                                analyzer.visualize_path(path, "path_visualization_alternative.png")
                                print("Path visualization saved to path_visualization_alternative.png")
                                break
        else:
            print("Could not find enough valid walkable points for pathfinding.")
    else:
        # Run the standard analysis with simulated data
        analyzer.run_analysis()
    
    print("\nAnalysis complete!") 