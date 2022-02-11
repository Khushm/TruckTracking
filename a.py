import cv2
import imutils
import numpy as np

img = cv2.imread('73.jpg')
height, width, _ = np.shape(img)

# calculate the average color of each row of our image
avg_color_per_row = np.average(img, axis=0)

# calculate the averages of our rows
avg_colors = np.average(avg_color_per_row, axis=0)

# avg_color is a tuple in BGR order of the average colors
# but as float values
print(f'avg_colors: {avg_colors}')

# so, convert that array to integers
int_averages = np.array(avg_colors, dtype=np.uint8)
print(f'int_averages: {int_averages}')

# create a new image of the same height/width as the original
average_image = np.zeros((height, width, 3), np.uint8)
# and fill its pixels with our average color
average_image[:] = int_averages

# finally, show it side-by-side with the original
cv2.imshow("Avg Color", np.hstack([img, average_image]))
cv2.waitKey(0)
cv2.destroyAllWindows()



def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


# Load the two images
img1 = cv2.imread('6.jpg')
img2 = cv2.imread("74.jpg")
img3 = cv2.imread("73.jpg")

# print(unique_count_app(img1))
print(unique_count_app(img2))
print(unique_count_app(img3))


# Resize images if necessary
img1 = cv2.resize(img1, (600,360))
img2 = cv2.resize(img2, (600,360))

img_height = img1.shape[0]

# Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find the difference between the two images
# Calculate absolute difference between two arrays
diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1, img2)", diff)

# Apply threshold. Apply both THRESH_BINARY and THRESH_OTSU
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
