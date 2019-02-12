from playsound import playsound
import pygame
from pygame.locals import*
import time

import pygame
from pygame.locals import*
from glob import glob
import os
import sys
from optparse import OptionParser

PAUSE1 = 3
PAUSE2 = 10
PAUSE3 = 5

def show_image(path, ecran):
	try :
		image = pygame.image.load(path).convert_alpha()
	except pygame.error:
		print("Image not found")
		return

	ecran.blit(image, (0, 0))                       
	pygame.display.flip()


def main(acoustic, folder="images/", k=0):
	pygame.init()

	ecran = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
	continuer = True

	show_image("images/Start.png", ecran)

	while continuer:

		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == K_ESCAPE:
					pygame.quit()
					exit()
				if event.key == K_RETURN:
					continuer = False
					
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()


	images = ["Algo1_original.png", "Algo1_Mix1.png", "Algo1_Mix2.png", "Algo1_Mix3.png", \
			  "Algo2_original.png", "Algo2_Mix1.png", "Algo2_Mix2.png", "Algo2_Mix3.png", \
			  "Algo3_original.png", "Algo3_Mix1.png", "Algo3_Mix2.png", "Algo3_Mix3.png", \
			  "Algo4_original.png", "Algo4_Mix1.png", "Algo4_Mix2.png", "Algo4_Mix3.png"]

	audio = ["Algo1_original_normalized.wav", "Algo1_Mix1_normalized.wav", "Algo1_Mix2_normalized.wav", "Algo1_Mix3_normalized.wav", \
			  "Algo2_original_normalized.wav", "Algo2_Mix1_normalized.wav", "Algo2_Mix2_normalized.wav", "Algo2_Mix3_normalized.wav", \
			  "Algo3_original_normalized.wav", "Algo3_Mix1_normalized.wav", "Algo3_Mix2_normalized.wav", "Algo3_Mix3_normalized.wav", \
			  "Algo4_original_normalized.wav", "Algo4_Mix1_normalized.wav", "Algo4_Mix2_normalized.wav", "Algo4_Mix3_normalized.wav"]	
	k = 0
	continuer = True

	while continuer:

		show_image("images/" + images[k], ecran)

		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == K_ESCAPE:
					pygame.quit()
					exit()
				if event.key == K_RETURN:
					continuer = False
					
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()

		playsound("audio/" + acoustic + "/" + audio[k])
		k += 1

		show_image("images/Pause.png", ecran)

		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == K_ESCAPE:
					pygame.quit()
					exit()
				if event.key == K_RETURN:
					continuer = False
					
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()

		if k % 4 == 1:
			print("pause après original, PAUSE1")
			time.sleep(PAUSE1)
		elif k % 4 == 0 and k != 0:
			print("pause entre algo, PAUSE2")
			time.sleep(PAUSE2)
		else:
			print("Pause entre écoute, PAUSE3")
			time.sleep(PAUSE3)


		if k >= len(images):
			continuer = False


	pygame.quit()

if __name__ == "__main__":

	if len(sys.argv) == 2 :
		main(sys.argv[1])

	else:
		print("Usage: Python3 Main.py <ROOM or ANECHOIC>")  
