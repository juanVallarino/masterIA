import pygame
import numpy as np
import time

pygame.init()

width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))

background = 25, 25, 25
screen.fill(background)
cellX, cellY = 50, 50

dimCW = width / cellX
dimCH = height / cellY

#estado de las celdas vivas = 1, muertas = 0
gameState = np.zeros((cellX, cellY))
pauseExec = False

#automata
gameState[21,21] = 1
gameState[22,22] = 1
gameState[22,23] = 1
gameState[21,23] = 1
gameState[20,23] = 1
#BUCLE DE EJECUCION
while True:

    newGameState = np.copy(gameState)
    screen.fill(background)
    time.sleep(0.1)

    ev = pygame.event.get()

    for event in ev:
        if event.type == pygame.KEYDOWN:
            pauseExec = not pauseExec

        mouseClick = pygame.mouse.get_pressed()

        if sum(mouseClick) > 0:
            posX, posY = pygame.mouse.get_pos()
            celdaX, celdaY = int(np.floor(posX / dimCW)), int(np.floor(posY / dimCH))
            newGameState[celdaX, celdaY] = not mouseClick[2]

    for y in range(0, cellX):
        for x in range(0, cellY):

            if not pauseExec:
                n_vecinos = gameState[(x-1) % cellX, (y-1) % cellY] + \
                            gameState[(x)   % cellX, (y-1) % cellY] + \
                            gameState[(x+1) % cellX, (y-1) % cellY] + \
                            gameState[(x-1) % cellX, (y)   % cellY] + \
                            gameState[(x+1) % cellX, (y)   % cellY] + \
                            gameState[(x-1) % cellX, (y+1) % cellY] + \
                            gameState[(x)   % cellX, (y+1) % cellY] + \
                            gameState[(x+1) % cellX, (y+1) % cellY]

                #regla 1: una celula muerta con exactamente 3 vecinos vivos, revive
                if gameState[x,y] == 0 and n_vecinos == 3:
                    newGameState[x,y] = 1

                #regla 2: una celula viva con menos de 2 o mas de 3 vecinos vivos, muere.
                elif gameState[x,y] == 1 and (n_vecinos < 2 or n_vecinos > 3):
                    newGameState[x,y] = 0

            poly = [((x) * dimCW, y * dimCH),
                    ((x+1) * dimCW, y * dimCH),
                    ((x+1) * dimCW, (y+1) * dimCH),
                    ((x) * dimCW, (y+1) * dimCH)
                    ]
            #dibujar la cerla por cad par x, y
            if newGameState[x,y] == 0:
                pygame.draw.polygon(screen, (128,128,128), poly, 1)
            else:
                pygame.draw.polygon(screen, (255,255,255), poly, 0)

    #actualizar el estado
    gameState = np.copy(newGameState)


    pygame.display.flip()
    pass