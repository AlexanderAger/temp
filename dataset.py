import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


class MazeTransform(data.Dataset):
    
    def __init__(self, file, dims, train=True, transform=None, target_transform=None):
        self.file = file
        self.dims = dims
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  #Training set or test set
        
        self.grid, self.x_coord, self.y_coord, self.target_actions, self.path_lengths = self.load_dataset(file, self.train)

    def __getitem__(self, index):
        #Returns the specified maze
        grid = self.grid[index]
        x_coord = self.x_coord[index]
        y_coord = self.y_coord[index]
        target_action = self.target_actions[index]
        path_length = self.path_lengths[index]
        
        #Apply transform or convert to tensor
        if self.transform is not None:
            grid = self.transform(grid)
        else: 
            grid = torch.from_numpy(grid).float()  
            
        return grid, x_coord, y_coord, target_action, path_length
        
    def __len__(self):
        return len(self.path_lengths)
        
    def load_dataset(self, file, train):
        data_container = np.load(file, allow_pickle=True)
        
        #Check if the file is in the expected format, two channels, train/test.
        if 'train' in data_container and 'test' in data_container:
            dataset = data_container['train'] if train else data_container['test']
        else:
            print("unexpected channel format")
        
        data_len = len(dataset)

        # Calculating maze dimensions
        grid_size = self.dims ** 2
        maze_dims = (self.dims, self.dims)
        
        #Initialize np arrays
        grid_img = np.zeros((data_len, 2, *maze_dims), dtype=np.float32)
        x_coord = np.zeros(data_len, dtype=np.int64)
        y_coord = np.zeros(data_len, dtype=np.int64)
        target_actions = np.zeros(data_len, dtype=np.int64)
        path_lengths = np.zeros(data_len, dtype=np.int64)
        
        # For each maze
        for index in range(data_len):
            maze = dataset[index]
            x_coord[index] = maze[0]  # x start coordinate
            y_coord[index] = maze[1]  # y start coordinate
            target_actions[index] = maze[2]  # optimal action for next step
            
            #Extract obstacle map and goal map
            if self.dims == 100:  #16x16 maze dimensions
                #Obstacle map is 16x16 items starting from index 3
                obstacle_map = np.array(maze[3:3+self.dims*self.dims]).reshape(self.dims, self.dims)
                
                #Goal map is 16x16 items starting after the obstacle map
                goal_map_start = 3 + self.dims*self.dims
                goal_map = np.array(maze[goal_map_start:goal_map_start+self.dims*self.dims]).reshape(self.dims, self.dims)
                
                #Path length is after the goal map
                path_length_index = goal_map_start + self.dims*self.dims
                path_lengths[index] = maze[path_length_index]
            
            #Store obstacle and goal maps
            grid_img[index, 0] = obstacle_map
            grid_img[index, 1] = goal_map
        
        return grid_img, x_coord, y_coord, target_actions, path_lengths

    def get_raw_maze(self, index):
        """Get raw maze for visualization"""
        return self.grid[index]
        
