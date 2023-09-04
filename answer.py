import numpy as np
import matplotlib.pyplot as plt
# from utils.plot import plot_image

''' Saving a image in the disk '''
def save_image(filename, image, gray=False):
    temp = np.array(image)
    temp = temp.astype(np.uint8)
    # Save the image
    path = image_path + filename   # separate folder created for output images
    if gray:    
        plt.imsave(path, temp, cmap='gray')
    else:       
        plt.imsave(path, temp)
    print(f"Image saved at {path}")
    return


#region 1: Reading in images
image_path = 'images/'

image_rgb = plt.imread(image_path+'original.jpg')

# print('image_rgb.shape: ', image_rgb.shape)      #output : (height, width, channels) = (h=200, w=200, 3)
height, width, channels = image_rgb.shape
#endregion



#region 2: 
# Convert the image to gray-scale (8bpp format) 
# print (image_rgb)
image_gray = [[[0 for k in range(3)] for j in range(width)] for i in range(height)]

for i in range(height):
    for j in range(width):
        # Extracting the RGB values
        r, g, b = image_rgb[i][j]
        # Converting to grayscale considering the Luminance level as it was widely use than others in as in YUV and YCrCb formats
        # Formula: Y = 0.299 R + 0.587 G + 0.114 B
        image_gray[i][j] = int(0.299*r + 0.587*g + 0.114*b)

        ### Another methods to convert to grayscale:
        ## 1. Average method
        # image_rgb[i][j] = int(sum(r + g + b)/3)
        ## 2. Lightness method
        # image_rgb[i][j] = int((max(r, g, b) + min(r, g, b))/2)
else:
    print('Image converted to grayscale successfully!')

#Testing:
# plot_image(image_gray, 'Grayscale Image', True)

save_image('gray.jpg', image_gray, True)
#endregion




#region 3: 
# Re-sample the image such that the size is 0.7 times it original dimensions using linear interpolation method and save the image.
image_resampled = [[[0 for k in range(3)] for j in range(int(width*0.7))] for i in range(int(height*0.7))]

for y in range(int(height*0.7)):
    for x in range(int(width*0.7)):
        # let pixel on the resampled image = (x,y)
        # Compared the pixel on the original image would be = (x/0.7, y/0.7) as we are doing down-sampling of 70%
        # But x/0.7 and y/0.7 are not integers, so we need to interpolate the values
        # We can use linear interpolation method to find the value of the pixel on the original image
        # Linear Interpolation Formula: f(x) = f(x1) + (x - x1) * (f(x2) - f(x1)) / (x2 - x1)
        # where x1 = floor(x/0.7) and x2 = ceil(x/0.7)
        X = x/0.7
        x1 = int(x/0.7)
        x2 = x1 + 1

        Y = y/0.7
        y1 = int(y/0.7)
        y2 = y1 + 1
        # print(x1, x2, y1, y2)
        # print(image_rgb[x1][y1], image_rgb[x2][y1], image_rgb[x1][y2], image_rgb[x2][y2])
        try:
            a = X - x1  
            # let, difference to the nearest floor pixel (the pixel on the zero side of the x axis) on the original image = a
            # then, difference to the nearest ceil pixel on the original image = (1-a)
            # difference on the x-axis for the nearest pixels on the original image will be a and (1-a)

            b = Y - y1
            # Similarly for y axis, difference to the nearest floor pixel (the pixel on the zero side of the y axis) on the original image = b

            ## Now, we can use the linear interpolation formula to find the value of the pixel on the original image
            # Formula: f(x,y) = (1-a)*(1-b)*f(x1,y1) + a*(1-b)*f(x2,y1) + (1-a)*b*f(x1,y2) + a*b*f(x2,y2) 
            
            ## Red channel
            r = (1-b)*(1-a)*image_rgb[y1][x1][0] + (1-b)*a*image_rgb[y1][x2][0] + b*(1-a)*image_rgb[y2][x1][0] + b*a*image_rgb[y2][x2][0] 
            ## Green channel
            g = (1-b)*(1-a)*image_rgb[y1][x1][1] + (1-b)*a*image_rgb[y1][x2][1] + b*(1-a)*image_rgb[y2][x1][1] + b*a*image_rgb[y2][x2][1] 
            ## Blue channel
            b = (1-b)*(1-a)*image_rgb[y1][x1][2] + (1-b)*a*image_rgb[y1][x2][2] + b*(1-a)*image_rgb[y2][x1][2] + b*a*image_rgb[y2][x2][0] 

            # print(r, g, b)
            # Applying the calculated values to the resampled image
            image_resampled[y][x] = [int(r), int(g), int(b)]

        except:
            # Error case for edge pixels
            image_resampled[y][x] = image_rgb[y1][x1]
        
        # image_resampled[y][x] = [int(image_rgb[y1][x1][k] + (x/0.7 - x1) * (image_rgb[y2][x1][k] - image_rgb[y1][x1][k]) / (x2 - x1)) for k in range(3)]
else:
    print('Image downscaled successfully!')

#Test      
# plot_image(image_resampled, 'Downscaled Image')

save_image('downscaled.jpg', image_resampled)
#endregion




#region 4: 
# Re-sample the image created in (step 3) back to its original size and save the image.
image_upscaled = [[[0 for k in range(3)] for j in range(width)] for i in range(height)]

for y in range(height):
    for x in range(width):
        # Here, we are doing up-sampling to 142.857% (100/70) of the dowscaled image

        # let pixel on the resampled image = (x,y)
        # Compared the pixel on the original image would be = (x*0.7, y*0.7) as we are doing up-sampling of 70%
        # But x*0.7 and y*0.7 are not always integers, so we need to interpolate the values
        
        # where x1 = floor(x*0.7) and x2 = ceil(x*0.7)
        X = x*0.7
        x1 = int(x*0.7)
        x2 = x1 + 1

        Y = y*0.7
        y1 = int(y*0.7)
        y2 = y1 + 1

        try:
            a = X - x1  
            # let, difference to the nearest floor pixel (the pixel on the zero side of the x axis) on the original image = a
            # then, difference to the nearest ceil pixel on the original image = (1-a)
            # difference on the x-axis for the nearest pixels on the original image will be a and (1-a)

            b = Y - y1
            # Similarly for y axis, difference to the nearest floor pixel (the pixel on the zero side of the y axis) on the original image = b

            ## Now, we can use the linear interpolation formula to find the value of the pixel on the original image
            # Formula: f(x,y) = (1-a)*(1-b)*f(x1,y1) + a*(1-b)*f(x2,y1) + (1-a)*b*f(x1,y2) + a*b*f(x2,y2) 
            
            ## Red channel
            r = (1-b)*(1-a)*image_resampled[y1][x1][0] + (1-b)*a*image_resampled[y1][x2][0] + b*(1-a)*image_resampled[y2][x1][0] + b*a*image_resampled[y2][x2][0]   

            ## Green channel
            g = (1-b)*(1-a)*image_resampled[y1][x1][1] + (1-b)*a*image_resampled[y1][x2][1] + b*(1-a)*image_resampled[y2][x1][1] + b*a*image_resampled[y2][x2][1]

            ## Blue channel
            b = (1-b)*(1-a)*image_resampled[y1][x1][2] + (1-b)*a*image_resampled[y1][x2][2] + b*(1-a)*image_resampled[y2][x1][2] + b*a*image_resampled[y2][x2][2]

            # Applying the calculated values to the resampled image
            image_upscaled[y][x] = [int(r), int(g), int(b)]

        except:
            # Error case for edge pixels
            image_upscaled[y][x] = image_rgb[y1][x1]
else:
    print('Image upscaled successfully!')
        
#Testing:
# plot_image(image_upscaled, 'Upscaled Image')

save_image('upscaled.jpg', image_upscaled)
#endregion 


''' Image plotting : (Used for testing purposes only) '''
fig, axs = plt.subplots( 2, 2, figsize=(5,5) )
axs[0,0].imshow(image_rgb)
axs[0,0].set_title('Original Image')
axs[0,1].imshow(image_gray, cmap='gray')
axs[0,1].set_title('Grayscale Image')
axs[1,0].imshow(image_resampled)
axs[1,0].set_title('Downscaled Image')
axs[1,1].imshow(image_upscaled)
axs[1,1].set_title('Upscaled Image')
for ax in axs.flat:
    ax.axis('on')       # visibility of x- and y-axes
    ax.grid(True)       # show gridlines

plt.tight_layout()
plt.show()



#region 5: 
# Compute the sum of the average of the squared difference between pixels in the original image (in step 2) and the re-samples image in (step 4)

image_rgb = np.array(image_rgb).astype(np.float32)
image_upscaled = np.array(image_upscaled).astype(np.float32) 

# calculation of the sum of the average of the squared differences
difference = image_rgb - image_upscaled
squared_diff = np.square(difference)
average = np.mean(squared_diff)
sum_of_squared_diff = np.sum(average)

print('Sum of average of the squared differences:', sum_of_squared_diff)

''' Output recieved:
    Sum of average of the squared differences: 
        Image of the lake:  23.493326
        Image of the mountain:  55.308125 
        Image of the dog:  143.88313 
        Image of the girl: 78.7881
        Image of the cameraman:  20.249739
        Image of the cube:  75.3132
        Image of the baby:  31.942661
        '''

#endregion



