'''import pygame._sdl2.controller
from pygame.locals import *
from sys import exit
import numpy as np
from SwerveDrive import SwerveDrive
import Box2D

using_controller = True

fps = 1650
clock = pygame.time.Clock()

pygame.init()
running = True

if using_controller:
    pygame._sdl2.controller.init()
    controller = pygame._sdl2.controller.Controller(0)

screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("Swerve Sim")

left_x_deadzone = 0.05
left_y_deadzone = 0.05
right_x_deadzone = 0.05

robots = [SwerveDrive(position={"x": 200, "y": 200}, angle=0, velocity=7, angular_velocity=4),
          SwerveDrive(position={"x": 600, "y": 600}, angle=0, velocity=7, angular_velocity=4),
          SwerveDrive(position={"x": 400, "y": 400}, angle=0, velocity=7, angular_velocity=4)]

width = 100
height = 100

velocity = 5
angular_velocity = 5

imgs = [pygame.image.load("SwerveImages/swerve(1).png") for robot in robots]
print(imgs)
#img = pygame.transform.scale(img, (width, height))


def blitRotate(surf, image, pos, originPos, angle):
    # offset from pivot to center
    image_rect = image.get_rect(topleft=(pos[0] - originPos[0], pos[1] - originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center

    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center=rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)

    # draw rectangle around the image
    #pygame.draw.rect(surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()), 2)


def blitRotate2(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)

    surf.blit(rotated_image, new_rect.topleft)
    pygame.draw.rect(surf, (255, 0, 0), new_rect, 2)



w = [0 for robot in robots]
h = [0 for robot in robots]

for img in imgs:
    w[imgs.index(img)], h[imgs.index(img)] = img.get_size()

while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(0)

    for robot in robots:
        x = robot.get_position()["x"]
        y = robot.get_position()["y"]
        angle = robot.get_angle()
        velocity = robot.get_velocity()
        angular_velocity = robot.get_angular_velocity()
        if using_controller:
            x += (controller.get_axis(CONTROLLER_AXIS_LEFTX) / 32767 if np.abs(controller.get_axis(CONTROLLER_AXIS_LEFTX) / 32767) >  left_x_deadzone else 0) * velocity
            y += (controller.get_axis(CONTROLLER_AXIS_LEFTY) / 32767 if np.abs(controller.get_axis(CONTROLLER_AXIS_LEFTY) / 32767) >  left_y_deadzone else 0) * velocity
            angle += (-controller.get_axis(CONTROLLER_AXIS_RIGHTX) / 32767 if np.abs(controller.get_axis(CONTROLLER_AXIS_RIGHTX) / 32767) >  right_x_deadzone else 0) * angular_velocity

        # check for edge
        if x > 1920 - width:
            x = 1920 - width
        elif x < 0:
            x = 0

        if y > 1080 - height:
            y = 1080 - height
        elif y < 0:
            y = 0

        robot.set_position({"x": x, "y": y})
        robot.set_angle(angle)

        blitRotate(screen, imgs[robots.index(robot)], (x, y), (w[robots.index(robot)] / 2, h[robots.index(robot)] / 2), angle)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_ESCAPE]:
        running = False



    pygame.display.flip()
    #pygame.display.update()

    #clock.tick()

pygame.quit()'''