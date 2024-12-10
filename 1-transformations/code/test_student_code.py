import cv2
import random
from student_code import RandomRotate


def test_random_rotate():
    rot = RandomRotate(180)
    # random.seed(42)
    img = cv2.imread('../data/dog.bmp')
    img = rot(img)
    cv2.imwrite('dog_rotated.bmp', img)
    
if __name__ == '__main__':
    test_random_rotate()