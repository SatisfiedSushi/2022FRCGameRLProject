import functools
import math
import random
import time
from copy import copy

import gymnasium
from gymnasium.spaces import Dict, Box, MultiDiscrete
# from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
import pygame
import pygame._sdl2.controller
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.Box2D import *
from pettingzoo.utils.env import ParallelEnv

from SwerveDrive import SwerveDrive


class ScoreHolder:
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

    def get_score(self, team):
        match team:
            case 'Blue':
                return self.blue_points
            case 'Red':
                return self.red_points


class MyContactListener(b2ContactListener):
    def destroy_body(self, body_to_destroy, team):
        body_to_destroy.userData({"ball": True, 'Team': team, "isFlaggedForDelete": True})

    def GetBodies(self, contact):
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a = fixture_a.body
        body_b = fixture_b.body

        return body_a, body_b

    def __init__(self, scoreHolder):
        b2ContactListener.__init__(self)
        self.scoreHolder = scoreHolder

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
                        self.scoreHolder.increase_points(ball.userData['Team'])
                    self.destroy_body(ball, ball.userData['Team'])

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


class env(ParallelEnv):
    metadata = {
        'render.modes': ['human'],
        'name': 'FRCGameEnv-v0'
    }

    def meters_to_pixels(self, meters):
        return int(meters * self.PPM)

    def sweep_dead_bodies(self):
        for body in self.world.bodies:
            if body is not None:
                data = body.userData
                if data is not None:
                    if "isFlaggedForDelete" in data:
                        if data["isFlaggedForDelete"]:
                            choice = random.randint(1, 4)
                            if 'ball' in body.userData and 'Team' in body.userData:
                                self.create_new_ball((self.hub_points[choice - 1].x, self.hub_points[choice - 1].y),
                                                     ((choice * (np.pi / 2)) + np.pi) + 1.151917 / 4, data["Team"])
                                self.balls.remove(body)
                                self.world.DestroyBody(body)
                                body.userData(None)
                                body = None

    def return_closest_ball(self, robot):
        LL_FOV = 31.65
        closest_ball = None
        angle_offset = 0

        for ball in self.balls:
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
                    elif (new_ball_position[0] ** 2 + new_ball_position[1] ** 2) < (
                            closest_ball.position.x ** 2 + closest_ball.position.y ** 2):
                        closest_ball = ball
                        angle_offset = (math.degrees(robot.angle) % 360) - angle_degrees

        return closest_ball, angle_offset

    def return_robots_in_sight(self, robot_main):
        LL_FOV = 31.65  # 31.65 degrees off the center of the LL
        found_robots = []
        angle_offset = 0

        for robot in self.robots:
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

    def create_new_ball(self, position, force_direction, team, force=0.014 - ((random.random() / 100))):
        x = position[0]
        y = position[1]

        new_ball = self.world.CreateDynamicBody(position=(x, y),
                                                userData={"ball": True,
                                                          "Team": team,
                                                          "isFlaggedForDelete": False})

        new_ball.CreateCircleFixture(radius=0.12, density=0.1, friction=0.001)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_ball,
                                                bodyB=self.carpet,
                                                maxForce=0.01, maxTorque=5)
        self.world.CreateJoint(friction_joint_def)

        self.balls.append(new_ball)

        pos_or_neg = random.randint(0, 1)

        # force_direction = force_direction + (random.random()/36 if pos_or_neg == 0 else force_direction - random.random()) / 36 #  small random
        # force_direction = force_direction + (random.random()/18 if pos_or_neg == 0 else force_direction - random.random()) / 18 #  medium random
        force_direction = force_direction + (
            random.random() / 9 if pos_or_neg == 0 else force_direction - random.random()) / 9  # large random
        new_ball.ApplyLinearImpulse((np.cos(force_direction) * force, np.sin(force_direction) * force),
                                    point=new_ball.worldCenter, wake=True)

    def create_new_robot(self, **kwargs):
        position = kwargs['position'] or (0, 0)
        angle = kwargs['angle'] or 0
        team = kwargs['team'] or "Red"

        new_robot = self.world.CreateDynamicBody(position=position,
                                                 angle=angle,
                                                 userData={"robot": True,
                                                           "isFlaggedForDelete": False,
                                                           "Team": team})

        new_robot.CreatePolygonFixture(box=(0.56 / 2, 0.56 / 2), density=30, friction=0.01)
        friction_joint_def = b2FrictionJointDef(localAnchorA=(0, 0), localAnchorB=(0, 0), bodyA=new_robot,
                                                bodyB=self.carpet,
                                                maxForce=10, maxTorque=10)
        self.world.CreateJoint(friction_joint_def)

        self.robots.append(new_robot)

    def is_close_to_terminal(self, robot, red_spawned, blue_spawned):
        distance = 2.5
        force = 0.017

        if robot.userData['Team'] == 'Blue':
            if math.sqrt(robot.position.x ** 2 + robot.position.y ** 2) < distance and not blue_spawned:
                self.create_new_ball(position=(0, 0), force_direction=np.pi / 4, team='Blue', force=force)
                return 'Blue'

        else:
            if math.sqrt(robot.position.x ** 2 + robot.position.y ** 2) > -distance + math.sqrt(
                    self.terminal_red.position.x ** 2 + self.terminal_red.position.y ** 2) and not red_spawned:
                self.create_new_ball(position=(
                    self.terminal_red.position.x - (0.4 * math.sqrt(np.pi)),
                    self.terminal_red.position.y - (0.4 * math.sqrt(np.pi))),
                    force_direction=(np.pi / 4) + np.pi, team='Red', force=force)
                return 'Red'

    def __init__(self, render_mode="human"):
        # --- pygame setup ---
        self.PPM = 100.0  # pixels per meter
        self.TARGET_FPS = 60
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.meters_to_pixels(16.46), self.meters_to_pixels(8.23)
        self.screen = None
        self.clock = None
        self.teleop_time = 30  # 135 default

        # RL variables
        self.render_mode = render_mode
        self.possible_agents = ["red_1", "red_2", "red_3", "blue_1", "blue_2", "blue_3"]

        self.red_Xs = None
        self.red_Ys = None
        self.red_angles = None
        self.red_LL_x_angles = None
        self.red_LL_robot_x_angles = None
        self.red_LL_robot_teams = None

        self.blue_Xs = None
        self.blue_Ys = None
        self.blue_angles = None
        self.blue_LL_x_angles = None
        self.blue_LL_robot_x_angles = None
        self.blue_LL_robot_teams = None

        self.timestep = None
        self.current_time = None
        self.game_time = None

        self.scoreHolder = None

        self.red_spawned = None
        self.blue_spawned = None

        # --- other ---
        self.velocity_factor = 5
        self.angular_velocity_factor = 6
        self.balls = None
        self.robots = None
        self.swerve_instances = None

        # --- Box2d ---
        self.world = None
        self.hub_points = None
        self.carpet = None
        self.carpet_fixture = None
        self.lower_wall = None
        self.left_wall = None
        self.right_wall = None
        self.upper_wall = None
        self.terminal_blue = None
        self.terminal_red = None
        self.hub = None
        self.colors = {
            b2_staticBody: (255, 255, 255, 255),
            b2_dynamicBody: (127, 127, 127, 255),
        }

        def my_draw_polygon(polygon, body, fixture):
            vertices = [(body.transform * v) * self.PPM for v in polygon.vertices]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            if body.userData is not None:
                pygame.draw.polygon(self.screen,
                                    (0, 0, 255, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255),
                                    vertices)
            else:
                pygame.draw.polygon(self.screen, self.colors[body.type], vertices)

        b2PolygonShape.draw = my_draw_polygon

        def my_draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            pygame.draw.circle(self.screen, (0, 0, 255, 255) if body.userData['Team'] == 'Blue' else (255, 0, 0, 255),
                               [int(
                                   x) for x in position], int(circle.radius * self.PPM))
            # Note: Python 3.x will enforce that pygame get the integers it requests,
            #       and it will not convert from float.

        b2CircleShape.draw = my_draw_circle

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ):

        # --- RL variables ---
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.scoreHolder = ScoreHolder()
        self.current_time = time.time()
        self.game_time = self.teleop_time - (time.time() - self.current_time)

        # --- other ---
        self.balls = []
        self.robots = []  # TODO: find out how the fuck this works
        self.red_spawned = False
        self.blue_spawned = False

        # --- FRC game setup ---
        self.hub_points = []
        self.world = b2World(gravity=(0, 0), doSleep=True, contactListener=MyContactListener(self.scoreHolder))

        self.carpet = self.world.CreateStaticBody(
            position=(-3, -3),
        )

        self.carpet_fixture = self.carpet.CreatePolygonFixture(box=(1, 1), density=1, friction=0.3)

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
            position=(0, 8.23 + 1),
            shapes=b2PolygonShape(box=(16.46, 1)),
        )

        self.terminal_blue = self.world.CreateStaticBody(
            position=((0.247) / math.sqrt(2), (0.247) / math.sqrt(2)),
            angle=np.pi / 4,
            shapes=b2PolygonShape(box=(0.99, 2.47)),
        )

        self.terminal_red = self.world.CreateStaticBody(
            position=((16.46 - (0.247 / math.sqrt(2))), (8.23 - (0.247 / math.sqrt(2)))),
            angle=np.pi / 4,
            shapes=b2PolygonShape(box=(0.99, 2.47)),
        )

        self.hub = self.world.CreateStaticBody(
            position=(16.46 / 2, 8.23 / 2),
            angle=1.151917,
            shapes=b2PolygonShape(box=(0.86, 0.86)),
        )

        for vertex in self.hub.fixtures[0].shape.vertices:
            new_vertex = self.hub.GetWorldPoint(vertex)
            offset = 0
            if new_vertex.x < 0:
                new_vertex.x -= offset
            else:
                new_vertex.x += offset

            if new_vertex.y < 0:
                new_vertex.y -= offset
            else:
                new_vertex.y += offset

            self.hub_points.append(new_vertex)

        ball_circle_diameter = 7.77
        ball_circle_center = (16.46 / 2, 8.23 / 2)

        ball_x_coords = [0.658, -0.858, -2.243, -3.287, -3.790, -3.174, -0.658, 0.858, 2.243, 3.287, 3.790, 3.174,
                         -7.165,
                         7.165]
        ball_y_coords = [3.830, 3.790, 3.174, 2.074, -0.858, -2.243, -3.830, -3.790, -3.174, -2.074, 0.858, 2.243,
                         -2.990,
                         2.990]

        ball_teams = ["Red", "Blue", "Red", "Blue", "Red", "Blue", "Blue", "Red", "Blue", "Red", "Blue", "Red", "Blue",
                      "Red"]

        for x_coord, y_coord, team in zip(ball_x_coords, ball_y_coords, ball_teams):
            position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
            self.create_new_ball(position=position, force_direction=0, team=team, force=0)

        robot_x_coords = [-1.3815, -0.941, 0, 1.381, 0.941, 0]
        robot_y_coords = [0.5305, -0.9915, -1.3665, -0.53, 0.9915, 1.3665]

        robot_teams = ["Blue", "Blue", "Blue", "Red", "Red", "Red"]

        for x_coord, y_coord, team in zip(robot_x_coords, robot_y_coords, robot_teams):
            position = (x_coord + ball_circle_center[0], y_coord + ball_circle_center[1])
            self.create_new_robot(position=position, angle=0, team=team)

        self.swerve_instances = [
            SwerveDrive(robot, robot.userData['Team'], (1, 1), 1, velocity_factor=self.velocity_factor,
                        angular_velocity_factor=self.angular_velocity_factor) for robot in
            self.robots]  # TODO: find out how the fuck this works

        observations = {agent: None for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def reset_pygame(self):
        # --- pygame setup ---

        pygame.display.set_caption('Multi Agent Swerve Env')
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)

    def step(self, actions):  # TODO: change action dictionary
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.game_time = self.teleop_time - (time.time() - self.current_time)
        self.sweep_dead_bodies()

        for agent in self.agents:
            swerve = self.swerve_instances[self.agents.index(agent)]
            swerve.set_velocity(actions[agent]['velocity'])
            swerve.set_angular_velocity(actions[agent]['angular_velocity'])
            swerve.update()

            match self.is_close_to_terminal(swerve.get_box2d_instance(), self.red_spawned, self.blue_spawned):
                case 'Red':
                    self.red_spawned = True
                case 'Blue':
                    self.blue_spawned = True

        self.world.Step(self.TIME_STEP, 10, 10)
        self.clock.tick(self.TARGET_FPS)

        rewards = {
            agent: {
                self.scoreHolder.get_score(
                    self.swerve_instances[self.agents.index(agent)].get_box2d_instance().userData['Team'])
            }
            for agent in self.agents
        }

        terminations = {agent: False for agent in self.agents}
        env_truncation = self.game_time > self.teleop_time
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {
            agent: {
                'observation': {
                    'velocity': self.swerve_instances[self.agents.index(agent)].get_velocity(),
                    'angular_velocity': self.swerve_instances[self.agents.index(agent)].get_angular_velocity(),
                    'angle': self.swerve_instances[self.agents.index(agent)].get_angle(),
                    'closest_ball':
                        self.return_closest_ball(self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[
                            1],
                    'robots_in_sight': {
                        'teams': 0 if self.return_robots_in_sight(
                            self.swerve_instances[self.agents.index(agent)].get_box2d_instance()) == [] else 1 if
                        self.return_robots_in_sight(
                            self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[0][
                            0] == 'Blue' else 2,
                        'angle': 0 if self.return_robots_in_sight(
                            self.swerve_instances[self.agents.index(agent)].get_box2d_instance()) == [] else
                        self.return_robots_in_sight(
                            self.swerve_instances[self.agents.index(agent)].get_box2d_instance())[0][1],
                    },
                }
            }

            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []
            pygame.quit()

        if self.render_mode == 'human':
            self.render()
        return observations, rewards, terminations, truncations, infos

    def close(self):
        pygame.quit()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.screen.fill((0, 0, 0, 0))

        for fixture in self.hub.fixtures:
            fixture.shape.draw(self.hub, fixture)

        for fixture in self.terminal_red.fixtures:
            fixture.shape.draw(self.terminal_red, fixture)

        for fixture in self.terminal_blue.fixtures:
            fixture.shape.draw(self.terminal_blue, fixture)

        for ball in self.balls:
            # adjusted ball position where the robot is centered at (0, 0)
            for fixture in ball.fixtures:
                fixture.shape.draw(ball, fixture)

        for agent in self.agents:
            swerve = self.swerve_instances[self.agents.index(agent)]
            for fixture in swerve.get_box2d_instance().fixtures:
                fixture.shape.draw(swerve.get_box2d_instance(), fixture)

        game_time_font = pygame.font.SysFont('Arial', 30)

        self.screen.blit(self.scoreHolder.render_score()[1], (10, 10))
        self.screen.blit(self.scoreHolder.render_score()[0], (self.screen.get_width() - 180, 10))
        self.screen.blit(game_time_font.render(f'{int(self.teleop_time - self.game_time)}', True, (255, 255, 255)),
                         (self.screen.get_width() / 2 - 20, 10))

        pygame.display.flip()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observations = Dict(
            {
                'velocity': Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,)),
                'angular_velocity': Box(low=np.array([-1]), high=np.array([1]), shape=(1,)),
                'angle': Box(low=np.array([-1]), high=np.array([360]), shape=(1,)),
                'closest_ball': Dict(
                    {
                        'angle': Box(low=np.array([-180]), high=np.array([180]), shape=(1,)),
                    }
                ),
                'robots_in_sight': Dict(
                    {
                        'angles': Box(low=np.array([-180, -180, -180, -180, -180]),
                                      high=np.array([180, 180, 180, 180, 180]), shape=(5,)),
                        'teams': MultiDiscrete(np.array([3, 3, 3, 3, 3])),
                    }
                ),
            }
        )
        return observations

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        actions = Dict(
            {
                'velocity': Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,)),
                'angular_velocity': Box(low=np.array([-1]), high=np.array([1]), shape=(1,)),
            }
        )
        return actions