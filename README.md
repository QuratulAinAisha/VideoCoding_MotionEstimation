

### Motion Estimation and Compensation with YUV Frames

This project reads a YUV video file, extracts grayscale frames, and performs basic motion estimation and compensation between two consecutive frames. It also calculates simple difference and residual images, showing the difference between the two frames.

### Step-by-Step Explanation

#### 1. **Reading YUV Frames**
The function `read_yuv_frames(filename, width, height)` reads YUV video frames from a file and converts the Y (luminance) channel to grayscale. The frames are stored as NumPy arrays in a list.

- **`filename`**: The path to the YUV file.
- **`width` & `height`**: Dimensions of the video.
- It reads the video in chunks corresponding to each frame and extracts the Y channel (grayscale).

```python
def read_yuv_frames(filename, width, height):
    frames = []  # Store frames

    num_bytes_per_frame = width * height * 1.5  # Estimate memory for YUV frame

    with open(filename, 'rb') as f:
        while True:
            yuv_data = f.read(int(num_bytes_per_frame))
            if not yuv_data:
                break

            y = yuv_data[0: width * height]  # Extract Y channel
            y = np.reshape(np.frombuffer(y, dtype=np.uint8), (height, width))  # Reshape to 2D array
            frames.append(y)  # Add grayscale frame to list

    return frames
```

#### 2. **Loading and Processing Frames**
The Y channel frames are loaded from the video. Two frames are extracted, converted to integers, and a simple absolute difference between the two frames is calculated.

```python
frames = read_yuv_frames("foreman.yuv", 352, 288)  # Load video frames
frame1 = frames[1]  # First frame
frame2 = frames[2]  # Second frame
intframe1 = frame1.astype(int)
intframe2 = frame2.astype(int)
simple_diff = np.abs(intframe1 - intframe2)  # Simple absolute difference
```

#### 3. **Block Matching Motion Estimation**
The code divides the frames into blocks and searches for the best matching block from `frame2` for each block in `frame1`. This is done using **Sum of Squared Differences (SSD)**.

- **Block Size**: 32x32 pixels.
- **Search Area**: ±16 pixels around each block.

For each block in `frame1`, the code searches within the defined area in `frame2` to find the block with the minimum SSD. It then stores the motion vector for each block.

```python
block_size = 32
search_area = 16
motion_vectors = []
residual_image = np.zeros_like(intframe1)

for y in range(0, frame1.shape[0] - block_size + 1, block_size):
    for x in range(0, frame1.shape[1] - block_size + 1, block_size):
        min_ssd = float('inf')
        best_match = (0, 0)
        block1 = frame1[y:y + block_size, x:x + block_size]

        for dy in range(-search_area, search_area + 1):
            for dx in range(-search_area, search_area + 1):
                y_start = y + dy
                y_end = y + dy + block_size
                x_start = x + dx
                x_end = x + dx + block_size
                block2 = frame2[y_start:y_end, x_start:x_end]

                if y_start >= 0 and y_end <= frame2.shape[0] and x_start >= 0 and x_end <= frame2.shape[1]:
                    ssd = np.sum((block1 - block2) ** 2)  # Calculate SSD
                    if ssd < min_ssd:
                        min_ssd = ssd
                        best_match = (dy, dx)

        motion_vectors.append(best_match)
        residual_image[y:y + block_size, x:x + block_size] = intframe1[y:y + block_size, x:x + block_size] - intframe2[y + best_match[0]:y + block_size + best_match[0], x + best_match[1]:x + block_size + best_match[1]]
```

#### 4. **Motion Compensation**
Using the calculated motion vectors, the code generates a **motion-compensated frame** by shifting blocks in `frame2` based on their corresponding motion vectors.

```python
frame_compensated = np.zeros_like(frame1)
for i, (dy, dx) in enumerate(motion_vectors):
    y = (i // (frame1.shape[1] // block_size)) * block_size
    x = (i % (frame1.shape[1] // block_size)) * block_size
    frame_compensated[y:y + block_size, x:x + block_size] = frame2[y + dy:y + dy + block_size, x + dx:x + dx + block_size]
```

#### 5. **Visualizing Results**
The code visualizes two images:
- **Simple Difference Image**: The absolute pixel differences between `frame1` and `frame2`.
- **Residual Image**: The difference between `frame1` and the motion-compensated frame.

```python
plt.imshow(simple_diff, cmap='gray')  # Display simple difference
plt.title('Simple Difference Image')
plt.show()

plt.imshow(np.abs(residual_image), cmap='gray')  # Display residual image
plt.title('Residual Image')
plt.show()
```

#### 6. **Printing Frames**
Finally, it prints out the frames for comparison.

```python
print("Frame2:", frame2)
print("FrameCompensated:", frame_compensated)
print("Residual Image:", residual_image)
print("Simple Diff:", simple_diff)
```

### Summary:
- **YUV Frames** are read, and two consecutive frames are processed.
- **Motion Estimation** is performed using block matching and SSD.
- **Motion Compensation** reconstructs the current frame using motion vectors.
- **Simple Difference** and **Residual Images** visualize the difference between the original and compensated frames.
