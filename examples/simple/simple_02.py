#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An attempt at some simple, self-contained pygame-based examples.
Example 02

In short:
One static body:
    + One fixture: big polygon to represent the ground
Two dynamic bodies:
    + One fixture: a polygon
    + One fixture: a circle
And some drawing code that extends the shape classes.

kne
"""
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)

from examples.framework import (Framework, Keys, main)
import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.Box2D import (b2World, b2PolygonShape, b2CircleShape, b2_staticBody, b2_dynamicBody)

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:



# --- main game loop ---

'''running = True
while running:
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    screen.fill((0, 0, 0, 0, 0))
    # Draw the world
    for body in world.bodies:
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)

        print(body.position.x, body.position.y)

    # Make Box2D simulate the physics of our world for one step.
    world.Step(TIME_STEP, 10, 10)

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
print('Done!')'''


class SimpleTest(Framework):
    name = "SimpleTest"
    def meters_to_pixels(self, meters):
        return int(meters * self.PPM)

    def __init__(self):
        super(SimpleTest, self).__init__()

        self.PPM = 100.0  # pixels per meter
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.meters_to_pixels(16.46), self.meters_to_pixels(8.23)

        # --- pybox2d world setup ---
        # Create the world
        self.world.gravity = (0, -10)

        self.boundary = self.world.CreateStaticBody(position=(0, 20))

        # And a static body to hold the ground shape
        self.lower_wall = self.world.CreateStaticBody(
            position=(0, -1),
            shapes=b2PolygonShape(box=(16.46, 1)),
        )
        self.left_wall = self.world.CreateStaticBody(
            position=(-1 - 0.02, 0),  # -1 is the actual but it hurts my eyes beacuase i can still see the wall
            shapes=b2PolygonShape(box=(1, 8.23)),
        )
        self.right_wall = self.world.CreateStaticBody(
            position=(16.47 + 1, 0),  # 16.46 is the actual but it hurts my eyes beacuase i can still see the wall
            shapes=b2PolygonShape(box=(1, 8.23)),
        )
        self.upper_wall = self.world.CreateStaticBody(
            position=(0, self.meters_to_pixels(8.23)),
            shapes=b2PolygonShape(box=(self.meters_to_pixels(16.46), 1)),
        )

        # Create a couple dynamic bodies
        body = self.world.CreateDynamicBody(position=(2, 45))
        self.circle = body.CreateCircleFixture(radius=0.5, density=1, friction=0.3)

        body = self.world.CreateDynamicBody(position=(16.46, 45), angle=15)
        self.box = body.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)

        self.colorss = {
            b2_staticBody: (255, 255, 255, 255),
            b2_dynamicBody: (127, 127, 127, 255),
        }

        def my_draw_polygon(polygon, body, fixture):
            vertices = [(body.transform * v) * self.PPM for v in polygon.vertices]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, self.colorss[body.type], vertices)

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            pygame.draw.circle(self.screen, self.colorss[body.type], [int(
                x) for x in position], int(circle.radius * self.PPM))
            # Note: Python 3.x will enforce that pygame get the integers it requests,
            #       and it will not convert from float.

        b2PolygonShape.draw = my_draw_polygon
        b2CircleShape.draw = my_draw_circle

    # Let's play with extending the shape classes to draw for us.

    def Step(self, settings):
        # Draw the world
        for body in self.world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

        super(SimpleTest, self).Step(settings)

if __name__ == "__main__":
    main(SimpleTest)
