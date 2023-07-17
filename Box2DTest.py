
import math
import time

import pygame
import random
# from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
from pygame.locals import *
import pygame._sdl2.controller
import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.Box2D import (b2World, b2PolygonShape, b2CircleShape, b2_staticBody, b2_dynamicBody, b2ContactListener)
from Box2D.Box2D import *

import main
from SwerveDrive import SwerveDrive

using_controller = False

if using_controller:
    pygame._sdl2.controller.init()
    controller = pygame._sdl2.controller.Controller(0)

left_x_deadzone = 0.15
left_y_deadzone = 0.15
right_x_deadzone = 0.15

velocity = 5
angular_velocity = 6

# --- constants ---
# Box2D deals with meters, but we want to display pixels,
# so define a conversion factor:
PPM = 100.0  # pixels per meter
def meters_to_pixels(meters):
    return int(meters * PPM)


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = meters_to_pixels(16.46), meters_to_pixels(8.23)

# --- pygame setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Multi Agent Swerve Env')
pygame.font.init()
clock = pygame.time.Clock()

teleop_time = 135 # 2:15

# --- pybox2d world setup ---
# Create the world

def destroy_body(body_to_destroy, team):
    body_to_destroy.__SetUserData({"ball": True, 'Team': team, "isFlaggedForDelete": True})

class ScoreHolder():
    def __init__(self):
        self.red_points = 0
        self.blue_points = 0

    def increase_points(self, team):
        match team:
            case 'Blue':
                self.blue_points += 1
            case 'Red':
                self.red_points += 1

    def reset_points(self):
        self.red_points = 0
        self.blue_points = 0

    def render_score(self):
        font = pygame.font.Font(None, 36)
        score_text_red = font.render(f'Red Points: {self.red_points}', True, (255, 0, 0))
        score_text_blue = font.render(f'Blue Points: {self.blue_points}', True, (0, 0, 255))
        return score_text_red, score_text_blue

scoreHolder = ScoreHolder()

class MyContactListener(b2ContactListener):
    def GetBodies(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a = fixture_a.body
        body_b = fixture_b.body

        return body_a, body_b

    def __init__(self, scoreHolder):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
        body_a, body_b = self.GetBodies(contact)
        main = None
        ball = None
        if body_a.userData is not None:
            main = body_a if 'robot' in body_a.userData else None
        if main is None:
            if body_b.userData is not None:
                main = body_b if 'robot' in body_b.userData else None
        if main is not None:
            if body_a.userData is not None:
                ball = body_a if 'ball' in body_a.userData else None
            if ball is None:
                if body_b.userData is not None:
                    ball = body_b if 'ball' in body_b.userData else None
            if ball is not None:
                new_ball_position = ((ball.position.x - main.position.x),
                                     (ball.position.y - main.position.y))

                angle_degrees = math.degrees(math.atan2(0 - new_ball_position[1],
                                                        0 - new_ball_position[0]) - np.pi)
                if angle_degrees < 0:
                    angle_degrees += 360

                if np.abs((math.degrees(main.angle) % 360) - angle_degrees) < 20:
                    '''print(main.angle)
                    print(angle_degrees)
                    print((math.degrees(main.angle) % 360) - angle_degrees)'''
                    # print("destroy")
                    if 'Team' in ball.userData:
                        scoreHolder.increase_points(ball.userData['Team'])
                    destroy_body(ball, ball.userData['Team'])

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


world = b2World(gravity=(0, 0), doSleep=True, contactListener=MyContactListener(scoreHolder))
hub_points = []

# main_robot_1 = world.CreateDynamicBody(position=(10, 5), angle=np.pi, userData={"main": 1, "robot": True, "isFlaggedForDelete": False, "Team": "Red"})
balls = []
robots = []
def sweepDeadBodies():
    for body in world.bodies:
        if body is not None:
            data = body.userData
            if data is not None:
                if "isFlaggedForDelete" in data:
                    if data["isFlaggedForDelete"]:
                        choice = random.randint(1, 4)
                        if 'ball' in body.userData and 'Team' in body.userData:
                            # create_new_ball((hub_points[choice - 1].x, hub_points[choice - 1].y),(choice * (np.pi / 2)))
                            create_new_ball((hub_points[choice-1].x, hub_points[choice-1].y), ((choice * (np.pi/2)) + np.pi) + 1.151917/4, data["Team"])
                            balls.remove(body)
                            world.DestroyBody(body)
                            body.__SetUserData(None)
                            body = None


carpet = world.CreateStaticBody(
    position=(-3, -3),
)

carpet_fixture = carpet.CreatePolygonFixture(box=(1, 1), density=1, friction=0.3)

# And a static body to hold the ground shape
lower_wall = world.CreateStaticBody(
    position=(0, -1),
    shapes=b2PolygonShape(box=(16.46, 1)),
)
left_wall = world.CreateStaticBody(
    position=(-1 - 0.02, 0),  # -1 is the actual but it hurts my eyes beacuase i can still see the wall
    shapes=b2PolygonShape(box=(1, 8.23)),
)
right_wall = world.CreateStaticBody(
    position=(16.47 + 1, 0),  # 16.46 is the actual but it hurts my eyes beacuase i can still see the wall
    shapes=b2PolygonShape(box=(1, 8.23)),
)
upper_wall = world.CreateStaticBody(
    position=(0, 8.23 + 1),
    shapes=b2PolygonShape(box=(16.46, 1)),
)

terminal_blue = world.CreateStaticBody(
    position=((0.247) / math.sqrt(2), (0.247) / math.sqrt(2)),
    angle=np.pi / 4,
    shapes=b2PolygonShape(box=(0.99, 2.47)),
)

terminal_red = world.CreateStaticBody(
    position=((16.46 - (0.247/ math.sqrt(2))), (8.23 - (0.247/ math.sqrt(2)))),
    angle=np.pi / 4,
    shapes=b2PolygonShape(box=(0.99, 2.47)),
)

print(terminal_red.position)

hub = world.CreateStaticBody(
    position=(16.46/2, 8.23/2),
    angle=1.151917,
    shapes=b2PolygonShape(box=(0.86, 0.86)),
)

for vertex in hub.fixtures[0].shape.vertices:
    new_vertex = hub.GetWorldPoint(vertex)
    offset = 0
    if new_vertex.x < 0:
        new_vertex.x -= offset
    else:
        new_vertex.x += offset

    if new_vertex.y < 0:
        new_vertex.y -= offset
    else:
        new_vertex.y += offset

    hub_points.append(new_vertex)
# hub_points = [hub.GetWorldPoint(vertex) for vertex in hub.fixtures[0].shape.vertices]
# Create a couple dynamic bodies -- in meters


def return_closest_ball(robot):
    LL_FOV = 31.65
    closest_ball = None
    angle_offset = 0

    for ball in balls:
        if ball.userData['Team'] == robot.userData['Team']:
            new_ball_position = ((ball.position.x - robot.position.x),
                                 (ball.position.y - robot.position.y))

            angle_degrees = math.degrees(math.atan2(0 - new_ball_position[1],
                                                    0 - new_ball_position[0]) - np.pi)
            if angle_degrees < 0:
                angle_degrees += 360
            if np.abs((math.degrees(robot.angle) % 360) - angle_degrees) < LL_FOV:
                if closest_ball is None:
                    closest_ball = ball
                    angle_offset = (math.degrees(robot.angle) % 360) - angle_degrees
                elif (new_ball_position[0] ** 2 + new_ball_position[1] ** 2) < (closest_ball.position.x ** 2 + closest_ball.position.y ** 2):
                    closest_ball = ball
                    angle_offset = (math.degrees(robot.angle) % 360) - angle_degrees

    return closest_ball, angle_offset


def return_robots_in_sight(robot_main):
    LL_FOV = 31.65
    found_robots = []
    angle_offset = 0

    for robot in robots:
        new_robot_position = ((robot.position.x - robot_main.position.x),
                             (robot.position.y - robot_main.position.y))

        angle_degrees = math.degrees(math.atan2(0 - new_robot_position[1],
                                                0 - new_robot_position[0]) - np.pi)
        if angle_degrees < 0:
            angle_degrees += 360
        if np.abs((math.degrees(robot_main.angle) % 360) - angle_degrees) < LL_FOV:
            angle_offset = (math.degrees(robot_main.angle) % 360) - angle_degrees
            found_robots.append([robot.userData['Team'], angle_offset])

    return found_robots


for robot in robots:
    robot.CreatePolygonFixture(box=(0.56/2, 0.56/2), density=12, friction=0.01)
    friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=robot, bodyB=carpet, maxForce=10, maxTorque=10)
    world.CreateJoint(friction_joint_def)


for ball in balls:
    ball.CreateCircleFixture(radius=0.12, density=0.1, friction=0.01)
    friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=ball, bodyB=carpet,
                                            maxForce=0.05, maxTorque=5)
    world.CreateJoint(friction_joint_def)

def create_new_ball(position, force_direction, team, force=0.014 - ((random.random() / 100))):
    x = position[0]
    y = position[1]

    new_ball = world.CreateDynamicBody(position=(x, y),
                                         userData={"ball": True,
                                                   "Team": team,
                                                   "isFlaggedForDelete": False})

    new_ball.CreateCircleFixture(radius=0.12, density=0.1, friction=0.001)
    friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_ball, bodyB=carpet,
                                            maxForce=0.01, maxTorque=5)
    world.CreateJoint(friction_joint_def)

    balls.append(new_ball)

    pos_or_neg = random.randint(0, 1)

    # force_direction = force_direction + (random.random()/36 if pos_or_neg == 0 else force_direction - random.random()) / 36 #  small random
    # force_direction = force_direction + (random.random()/18 if pos_or_neg == 0 else force_direction - random.random()) / 18 #  medium random
    force_direction = force_direction + (random.random()/9 if pos_or_neg == 0 else force_direction - random.random()) / 9 #  large random
    new_ball.ApplyLinearImpulse((np.cos(force_direction) * force, np.sin(force_direction) * force), point=new_ball.__GetWorldCenter(), wake=True)


def create_new_robot(**kwargs):
    position = kwargs['position'] or (0, 0)
    angle = kwargs['angle'] or 0
    team = kwargs['team'] or "Red"

    new_robot = world.CreateDynamicBody(position=position,
                            angle=angle,
                            userData={"robot": True,
                                      "isFlaggedForDelete": False,
                                      "Team": team})

    new_robot.CreatePolygonFixture(box=(0.56/2, 0.56/2), density=30, friction=0.01)
    friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_robot, bodyB=carpet,
                                            maxForce=10, maxTorque=10)
    world.CreateJoint(friction_joint_def)

    robots.append(new_robot)

ball_circle_diameter = 7.77
ball_circle_radius = ball_circle_diameter / 2
ball_circle_center = (16.46/2, 8.23/2)


'''create_new_ball(position=(0.658 + ball_circle_center[0], 3.830 + ball_circle_center[1]), force_direction=0, team="Red", force=0)
create_new_ball(position=(-0.858 + ball_circle_center[0], 3.790 + ball_circle_center[1]), force_direction=0, team="Blue", force=0)
create_new_ball(position=(-2.243 + ball_circle_center[0], 3.174 + ball_circle_center[1]), force_direction=0, team="Red", force=0)
create_new_ball(position=(-3.287 + ball_circle_center[0], 2.074 + ball_circle_center[1]), force_direction=0, team="Blue", force=0)
create_new_ball(position=(-3.790 + ball_circle_center[0], -0.858 + ball_circle_center[1]), force_direction=0, team="Red", force=0)
create_new_ball(position=(-3.174 + ball_circle_center[0], -2.243 + ball_circle_center[1]), force_direction=0, team="Blue", force=0)
create_new_ball(position=(-0.658 + ball_circle_center[0], -3.830 + ball_circle_center[1]), force_direction=0, team="Blue", force=0)
create_new_ball(position=(0.858 + ball_circle_center[0], -3.790 + ball_circle_center[1]), force_direction=0, team="Red", force=0)
create_new_ball(position=(2.243 + ball_circle_center[0], -3.174 + ball_circle_center[1]), force_direction=0, team="Blue", force=0)
create_new_ball(position=(3.287 + ball_circle_center[0], -2.074 + ball_circle_center[1]), force_direction=0, team="Red", force=0)
create_new_ball(position=(3.790 + ball_circle_center[0], 0.858 + ball_circle_center[1]), force_direction=0, team="Blue", force=0)
create_new_ball(position=(3.174 + ball_circle_center[0], 2.243 + ball_circle_center[1]), force_direction=0, team="Red", force=0)'''

ball_x_coords = [0.658, -0.858, -2.243, -3.287, -3.790, -3.174, -0.658, 0.858, 2.243, 3.287, 3.790, 3.174, -7.165, 7.165]
ball_y_coords = [3.830, 3.790, 3.174, 2.074, -0.858, -2.243, -3.830, -3.790, -3.174, -2.074, 0.858, 2.243, -2.990, 2.990]

ball_teams = ["Red", "Blue", "Red", "Blue", "Red", "Blue", "Blue", "Red", "Blue", "Red", "Blue", "Red", "Blue", "Red"]

for x_coord, y_coord, team in zip(ball_x_coords, ball_y_coords, ball_teams):
    position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
    create_new_ball(position=position, force_direction=0, team=team, force=0)

robot_x_coords = [-1.3815, -0.941, 0, 1.381, 0.941, 0]
robot_y_coords = [0.5305, -0.9915, -1.3665, -0.53, 0.9915, 1.3665]

robot_teams = ["Blue", "Blue", "Blue", "Red", "Red", "Red"]

'''robot_x_coords = [0]
robot_y_coords = [0]

robot_teams = ["Red"]''' # for testing


for x_coord, y_coord, team in zip(robot_x_coords, robot_y_coords, robot_teams):
    position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
    create_new_robot(position=position, angle=0, team=team)

colors = {
    b2_staticBody: (255, 255, 255, 255),
    b2_dynamicBody: (127, 127, 127, 255),
}


# Let's play with extending the shape classes to draw for us.



def my_draw_polygon(polygon, body, fixture):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    if body.userData is not None:
        pygame.draw.polygon(screen, (0, 0, 255, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255), vertices)
    else:
        pygame.draw.polygon(screen, colors[body.type], vertices)
b2PolygonShape.draw = my_draw_polygon


def my_draw_circle(circle, body, fixture):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(screen, (0, 0, 255, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255), [int(
        x) for x in position], int(circle.radius * PPM))
    # Note: Python 3.x will enforce that pygame get the integers it requests,
    #       and it will not convert from float.
b2CircleShape.draw = my_draw_circle

# --- main game loop ---
red_spawned = False
blue_spawned = False


def isCloseToTerminal(robot, red_spawned, blue_spawned):
    distance = 2.5
    force = 0.017

    if robot.userData['Team'] == 'Blue':
        if math.sqrt(robot.position.x ** 2 + robot.position.y ** 2) < distance and not blue_spawned:
            create_new_ball(position=(0, 0), force_direction=np.pi / 4, team='Blue', force=force)
            return 'Blue'

    else:
        if math.sqrt(robot.position.x ** 2 + robot.position.y ** 2) > -distance + math.sqrt(terminal_red.position.x ** 2 + terminal_red.position.y ** 2) and not red_spawned:
            create_new_ball(position=(terminal_red.position.x - (0.4 * math.sqrt(np.pi)), terminal_red.position.y - (0.4 * math.sqrt(np.pi))), force_direction=(np.pi / 4) + np.pi, team='Red', force=force)
            return 'Red'


swerve_instances = [SwerveDrive(robot, robot.userData['Team'], (1, 1), 1, velocity_factor=velocity, angular_velocity_factor=angular_velocity) for robot in robots]




current_time = time.time()
game_time = time.time() - current_time
running = True

while running:


    game_time = time.time() - current_time
    if game_time > teleop_time:
        running = False
    # Check the event queue
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

    screen.fill((0, 0, 0, 0))

    x, y, av = 0, 0, 0
    sweepDeadBodies()
    '''if using_controller:
        x = (controller.get_axis(CONTROLLER_AXIS_LEFTX) / 32767 if np.abs(
            controller.get_axis(CONTROLLER_AXIS_LEFTX) / 32767) > left_x_deadzone else 0) * velocity
        y = -(controller.get_axis(CONTROLLER_AXIS_LEFTY) / 32767 if np.abs(
            controller.get_axis(CONTROLLER_AXIS_LEFTY) / 32767) > left_y_deadzone else 0) * velocity
        av = (-controller.get_axis(CONTROLLER_AXIS_RIGHTX) / 32767 if np.abs(
            controller.get_axis(CONTROLLER_AXIS_RIGHTX) / 32767) > right_x_deadzone else 0) * angular_velocity'''

    '''count = 0
    # Draw the world
    for body in world.bodies:
        if count == 0:
            body.__SetLinearVelocity((x, y))
            body.__SetAngularVelocity(av)
        else:
            body.__SetLinearVelocity((0, 0))
            body.__SetAngularVelocity(0)
        for fixture in body.fixtures:
            fixture.shape.draw(body, fixture)

        count += 1'''

    '''main_robot.__SetLinearVelocity((x, y))
    main_robot.__SetAngularVelocity(av)'''

    for fixture in carpet.fixtures:
        fixture.shape.draw(carpet, fixture)

    for fixture in hub.fixtures:
        fixture.shape.draw(hub, fixture)

    for fixture in terminal_red.fixtures:
        fixture.shape.draw(terminal_red, fixture)

    for fixture in terminal_blue.fixtures:
        fixture.shape.draw(terminal_blue, fixture)

    # robots[1].__SetLinearVelocity((random.randint(-5, 5), random.randint(-5, 5)))
    # robots[1].__SetAngularVelocity(random.randint(-5, 5))

    for ball in balls:
        #adjusted ball position where the robot is centered at (0, 0)
        for fixture in ball.fixtures:
            fixture.shape.draw(ball, fixture)

    '''for body in world.bodies:
        if body.userData is not None:
            if 'robot' in body.userData:
                vel = body.GetLinearVelocityFromWorldPoint(body.position)
                angular_vel = body.__GetAngularVelocity()

                desired_x = 0
                desired_y = 0
                desired_theta = 0

                if x < 0:
                    desired_x = np.max([vel.x - 0.3, x])
                elif x == 0:
                    desired_x = vel.x * 0.9
                elif x > 0:
                    desired_x = np.min([vel.x + 0.3, x])

                if y < 0:
                    desired_y = np.max([vel.y - 0.3, y])
                elif y == 0:
                    desired_y = vel.y * 0.9
                elif y > 0:
                    desired_y = np.min([vel.y + 0.3, y])

                if av < 0:
                    desired_theta = np.max([angular_vel - 0.3, av])
                elif av == 0:
                    desired_theta = angular_vel * 0.97
                elif av > 0:
                    desired_theta = np.min([angular_vel + 0.3, av])

                vel_change_x = desired_x - vel.x
                vel_change_y = desired_y - vel.y
                vel_change_av = desired_theta - angular_vel

                impulse_x = body.mass * vel_change_x
                impulse_y = body.mass * vel_change_y
                impulse_av = body.mass * vel_change_av

                max_impulse_av = 1.4

                if impulse_av > 0:
                    impulse_av = np.min([impulse_av, max_impulse_av])
                elif impulse_av < 0:
                    impulse_av = np.max([impulse_av, -max_impulse_av])

                closest_ball, offset_angle = return_closest_ball(body)
                if closest_ball is not None:
                    print(f'LL.getXAngle = {offset_angle}')

                print(impulse_av)
                body.ApplyLinearImpulse((impulse_x, impulse_y), point=body.__GetWorldCenter(), wake=True)
                body.ApplyAngularImpulse(impulse_av, wake=True)

                # main_robot.ApplyForce(force=(x, y), point=main_robot.__GetWorldCenter(), wake=True)

                angle_degrees = (body.angle / np.pi * 180)  # radians to degrees
                #print((angle_degrees % 360))  # converts to 0 to 360

                match isCloseToTerminal(body, red_spawned, blue_spawned):
                    case 'Red':
                        red_spawned = True
                    case 'Blue':
                        blue_spawned = True



                for fixture in body.fixtures:
                    fixture.shape.draw(body, fixture)

            elif 'robot' in body.userData:
                for fixture in body.fixtures:
                    fixture.shape.draw(body, fixture)'''

    for swerve in swerve_instances:
        swerve.update()

        match isCloseToTerminal(swerve.get_box2d_instance(), red_spawned, blue_spawned):
            case 'Red':
                red_spawned = True
            case 'Blue':
                blue_spawned = True

        for fixture in swerve.get_box2d_instance().fixtures:
            fixture.shape.draw(swerve.get_box2d_instance(), fixture)

    game_time_font = pygame.font.SysFont('Arial', 30)

    screen.blit(scoreHolder.render_score()[1], (10, 10))
    screen.blit(scoreHolder.render_score()[0], (screen.get_width() - 180, 10))
    screen.blit(game_time_font.render(f'{int(teleop_time - game_time)}', True, (255, 255, 255)), (screen.get_width() / 2 - 20, 10))


    # Make Box2D simulate the physics of our world for one step.
    world.Step(TIME_STEP, 10, 10)

    # Flip the screen and try to keep at the target FPS
    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
print('Done!')
