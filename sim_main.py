import sim_utiles as utiles
import pygame as pg
import CONST
import math
## INITILIZATION ##
pg.init()
## VARIABLES ##
running = 1
STANDARD_SCREEN_DIMENSIONS = [800, 800]
STANDARD_DIMENSIONS = [100, 100]
WINDOW = pg.display.set_mode(tuple(STANDARD_SCREEN_DIMENSIONS)); pg.display.set_caption("Calculus-tool : sim_main.py : ln 10 : Simulation tool")
WINDOW.fill((255, 255, 255))
CENTRE_POINT = [int(STANDARD_DIMENSIONS[0] / 2), int(STANDARD_DIMENSIONS[1] / 2)]
pg.draw.line(WINDOW, (0, 0, 0), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), STANDARD_SCREEN_DIMENSIONS[1]), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), 0))
pg.draw.line(WINDOW, (0, 0, 0), (0, int(STANDARD_SCREEN_DIMENSIONS[1] / 2)), (STANDARD_SCREEN_DIMENSIONS[0], int(STANDARD_SCREEN_DIMENSIONS[1] / 2)))
pg.transform.scale(WINDOW, (1, 1))

def convert(seconds):
   seconds = seconds % (24 * 3600)
   hour = seconds // 3600
   seconds %= 3600
   minutes = seconds // 60
   seconds %= 60
   return "%02d:%02d:%02d" % (hour, minutes, seconds) #formatting

def transform_coordinates(point, centre=CENTRE_POINT, dims_screen=STANDARD_SCREEN_DIMENSIONS, dims=STANDARD_DIMENSIONS): # transform point in cartesian coords with respec to centre of screen to pygame coords
    return [dims_screen[0]/dims[0] * (centre[0] + point[0]), dims_screen[1]/dims[1] * (centre[1] - point[1])]

def draw_arrow(screen, colour, start, end):
    pg.draw.line(screen,colour,start,end,2)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pg.draw.polygon(screen, (255, 0, 0), ((end[0]+5*math.sin(math.radians(rotation)), end[1]+5*math.cos(math.radians(rotation))), (end[0]+5*math.sin(math.radians(rotation-120)), end[1]+5*math.cos(math.radians(rotation-120))), (end[0]+5*math.sin(math.radians(rotation+120)), end[1]+5*math.cos(math.radians(rotation+120)))))

## MAINLOOP ##
mass1 = 330000000
mass2 = 1000
body1 = utiles.Body(utiles.Vector([0, 0]), utiles.Vector([0, 0.0]), mass1, dt=CONST.dt, q=10 ** -4)
body2 = utiles.Body(utiles.Vector([-15, 0]), utiles.Vector([.01, .02]), mass2, dt=CONST.dt, q=-10 ** -5)
field1 = utiles.GravityField(body1)#;field1E = utiles.ElecField(body1); 
field2 = utiles.GravityField(body2)#;field2E = utiles.ElecField(body2);
force21 = utiles.GravForce(body2, field1)#; force21E = utiles.ElecForce(body2, field1E);
force12 = utiles.GravForce(body1, field2)#; force12E = utiles.ElecForce(body1, field2E);


time = 0
paused = 0
while running:
	for ev in pg.event.get():
		if ev.type == pg.QUIT:
			running = 0
		if ev.type == pg.KEYDOWN:
			if ev.key == pg.K_SPACE:
				paused = 1 - paused
			
			if ev.key == pg.K_ESCAPE:
				running = 0
			
			if ev.key == pg.K_RIGHT:
				CONST.dt *= 2
			
			if ev.key == pg.K_LEFT:
				CONST.dt /= 2

	if paused :
		WINDOW.fill((255, 255, 255))
		pg.draw.line(WINDOW, (0, 0, 0), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), STANDARD_SCREEN_DIMENSIONS[1]), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), 0))
		pg.draw.line(WINDOW, (0, 0, 0), (0, int(STANDARD_SCREEN_DIMENSIONS[1] / 2)), (STANDARD_SCREEN_DIMENSIONS[0], int(STANDARD_SCREEN_DIMENSIONS[1] / 2)))
		draw_arrow(WINDOW, (255, 0, 0), transform_coordinates(body1.pos.val), transform_coordinates((body1.pos + f12.force * (3 / abs(f12.force))).val))
		draw_arrow(WINDOW, (0, 0, 0), transform_coordinates(body1.pos.val), transform_coordinates((body1.pos + body1.vel * (3 / abs(body1.vel))).val))
		draw_arrow(WINDOW, (255, 0, 0), transform_coordinates(body2.pos.val), transform_coordinates((body2.pos + f21.force * (3 / abs(f21.force))).val))
		draw_arrow(WINDOW, (0, 0, 0), transform_coordinates(body2.pos.val), transform_coordinates((body2.pos + body2.vel * (3 / abs(body2.vel))).val))
		pg.draw.circle(WINDOW, (0, 0, 255), transform_coordinates(body1.pos.val[:]), 5)
		pg.draw.circle(WINDOW, (255, 0, 0), transform_coordinates(body2.pos.val[:]), 5)
		font = pg.font.SysFont("courier new", 15)
		img = font.render("time elapsed: %s"%convert(time), True, (0, 0, 0))
		img2 = font.render("Simulation frame time quanta / sec: %5.4f"%CONST.dt, True, (0, 0, 0))
		img3 = font.render("seconds elapsed: %5.4f"%time, True, (0, 0, 0))
		WINDOW.blit(img, (0, 0))
		WINDOW.blit(img3, (0, 15))
		WINDOW.blit(img2, (0, 30))

		font = pg.font.SysFont("courier", 15)
		img = font.render("paused", True, (255, 0, 0))
		WINDOW.blit(img, (0, 45))
		pg.display.update()

		continue;

	WINDOW.fill((255, 255, 255))
	pg.draw.line(WINDOW, (0, 0, 0), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), STANDARD_SCREEN_DIMENSIONS[1]), (int(STANDARD_SCREEN_DIMENSIONS[0] / 2), 0))
	pg.draw.line(WINDOW, (0, 0, 0), (0, int(STANDARD_SCREEN_DIMENSIONS[1] / 2)), (STANDARD_SCREEN_DIMENSIONS[0], int(STANDARD_SCREEN_DIMENSIONS[1] / 2)))
	field1.update()#;field1E.update();
	field2.update()#;field2E.update();
	force12.update();f12 = force12#;force12E.update();
	force21.update();f21 = force21#;force21E.update();
	#body1.applyForce(force12E);draw_arrow(WINDOW, (0, 0, 255), transform_coordinates(body1.pos.val), transform_coordinates((body1.pos + force12E.force * (5 / abs(force12E.force))).val))
	body1.applyForce(force12); draw_arrow(WINDOW, (255, 0, 0), transform_coordinates(body1.pos.val), transform_coordinates((body1.pos + force12.force * (3 / abs(force12.force))).val))
	body1.applyVel();draw_arrow(WINDOW, (0, 0, 0), transform_coordinates(body1.pos.val), transform_coordinates((body1.pos + body1.vel * (3 / abs(body1.vel))).val))
	#body2.applyForce(force21E);draw_arrow(WINDOW, (0, 0, 255), transform_coordinates(body2.pos.val), transform_coordinates((body2.pos + force21E.force * (5 / abs(force21E.force))).val))
	body2.applyForce(force21);draw_arrow(WINDOW, (255, 0, 0), transform_coordinates(body2.pos.val), transform_coordinates((body2.pos + force21.force * (3 / abs(force21.force))).val))
	body2.applyVel();draw_arrow(WINDOW, (0, 0, 0), transform_coordinates(body2.pos.val), transform_coordinates((body2.pos + body2.vel * (3 / abs(body2.vel))).val))
	pg.draw.circle(WINDOW, (0, 0, 255), transform_coordinates(body1.pos.val[:]), 5)
	pg.draw.circle(WINDOW, (255, 0, 0), transform_coordinates(body2.pos.val[:]), 5)
	font = pg.font.SysFont("courier new", 15)
	img = font.render("time elapsed: %s"%convert(time), True, (0, 0, 0))
	img2 = font.render("Simulation frame time quanta / sec: %5.4f"%CONST.dt, True, (0, 0, 0))
	img3 = font.render("seconds elapsed: %5.4f"%time, True, (0, 0, 0))
	WINDOW.blit(img, (0, 0))
	WINDOW.blit(img3, (0, 15))
	WINDOW.blit(img2, (0, 30))

	pg.display.update()
	time += CONST.dt


pg.quit()
