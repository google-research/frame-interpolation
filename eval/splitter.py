"""Utility functions to split and stitch frames."""
import numpy as np

def splitFrame(frame, tiles, overlapping):
    """Split a frame in equal parts and return an array.
    Args:
    frame: numpy array of the frame
    tiles: number of tiles to split the frame into
    overlapping: number of extra pixels to overlap between tiles
    Returns:
    A numpy array of the split frame
    """
    # Split the frame into tiles
    height = frame.shape[0]
    width = frame.shape[1]
    height_split = int(height / tiles)
    width_split = int(width / tiles)
    # Add a frame to the image the size of the overlapping pixels
    frame = np.pad(frame, ((overlapping, overlapping), (overlapping, overlapping), (0, 0)), 'constant')
    split_frame = np.zeros((tiles**2, height_split+(2*overlapping), width_split+(2*overlapping), 3))
    tile = 0
    for i in range(tiles):
        for j in range(tiles):
            split_frame[tile, :, :, :] = frame[i*height_split:(i+1)*height_split+(2*overlapping), j*width_split:(j+1)*width_split+(2*overlapping), :]
            tile += 1
    return split_frame

def stitchFrame(split_frame, overlapping):
    """Stitch a frame from an array of tiles.
    Args:
    split_frame: numpy array of the split frame
    overlapping: number of extra pixels to overlapping between tiles
    Returns:
    A numpy array of the stitched frame
    """
    # Stitch the frame from the split tiles considering the overlapping pixels
    side_tiles = int(split_frame.shape[0]**.5)
    height = (side_tiles * split_frame.shape[1]) - (2*overlapping*side_tiles)
    width = (side_tiles * split_frame.shape[2]) - (2*overlapping*side_tiles)
    stitched_frame = np.zeros((height, width, 3))
    # Get width and height of the tiles
    height_split = split_frame.shape[1] - (2*overlapping)
    width_split = split_frame.shape[2] - (2*overlapping)
    stitched_width = side_tiles * width_split-(2*overlapping*side_tiles)
    stitched_height = side_tiles * height_split-(2*overlapping*side_tiles)
    tile = 0
    for i in range(side_tiles):
        for j in range(side_tiles):
            stitched_frame[height_split*i:height_split*(i+1), width_split*j:width_split*(j+1), :] = split_frame[tile,overlapping:height_split+overlapping,overlapping:width_split+overlapping,:]
            tile += 1
    return stitched_frame