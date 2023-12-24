import cv2 as cv
import numpy as np
import lanes



video = cv.VideoCapture('road2.mp4')


if not video.isOpened():
    print('error while opening the video')

cv.waitKey(1)

while video.isOpened():
    _,frame = video.read()

    copy_img = np.copy(frame)

    try:
        frame = lanes.canny(frame)
        frame = lanes.mask(frame)
        lines = cv.HoughLinesP(frame, 2, np.pi / 180, 100, np.array([()]), maxLineGap=5)
        average_lines = lanes.average_slope_intercept(frame, lines)
        line_image = lanes.display_lines(copy_img, average_lines)
    except:
        pass

    combo =cv.addWeighted(copy_img, 0.8, line_image, 0.5, 1)


    cv.imshow("Video", combo)


    cv.resizeWindow("Video", 1300, 800)




    if cv.waitKey(1) & 0xFF == ord('q'):
        break



video.release()

cv.destroyAllWindows()

