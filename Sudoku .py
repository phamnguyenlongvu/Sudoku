import pygame
import numpy
import random
import sys
pygame.init()
pygame.font.init()
random.seed()
RED = (255, 0, 0)
BLUE = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)

Nd = 9

# def main():
#     for case in range(1, 4):
#
#         r_filename = "testcase" + str(case)
#         w_filename = "solution" + str(case)
#         initial_state = []
#         with open(r_filename, "r") as r_file:
#             initial_state = [[int(num) for num in line.split(' ')] for line in r_file]
#             with open(w_filename,"w") as w_file:
#                 solution = solve(initial_state, w_file)
#                 if solution != None:
#                     w_file.writelines("result: \n")
#                     w_file.write(str(solution))
#                 else:
#                     w_file.write("result: can not solve\n")


class Grid:
    values = [([0] * 9) for i in range(9)]

    def __init__(self, rows, cols, width, height):
        self.rows = rows
        self.cols = cols
        self.cubes = [[Cube(self.values[i][j], i, j, width, height) for j in range(cols)] for i in range(rows)]
        self.width = width
        self.height = height
        self.model = None
        self.selected = None

    def update_model(self):
        self.model = [[self.cubes[i][j].value for j in range(self.cols)] for i in range(self.rows)]

    def place(self, val):
        row, col = self.selected
        self.values[row][col] = val
        self.cubes[row][col].set(val)
        self.update_model()

    def draw(self, win):
        # Draw Grid Lines
        gap = self.width / 9
        for i in range(self.rows + 1):
            if i % 3 == 0 and i != 0:
                thick = 4
            else:
                thick = 1
            pygame.draw.line(win, BLACK, (0, i * gap), (self.width, i * gap), thick)
            pygame.draw.line(win, BLACK, (i * gap, 0), (i * gap, self.height), thick)
        # Draw Cubes
        for i in range(self.rows):
            for j in range(self.cols):
                self.cubes[i][j].draw(win)

    def select(self, row, col):
        # Reset all other
        for i in range(self.rows):
            for j in range(self.cols):
                self.cubes[i][j].selected = False

        self.cubes[row][col].selected = True
        self.selected = (row, col)

    def clear(self):
        row, col = self.selected
        self.values[row][col] = 0
        self.cubes[row][col].set(0)

    def click(self, pos):
        """
        :param: pos
        :return: (row, col)
        """
        if pos[0] < self.width and pos[1] < self.height:
            gap = self.width / 9
            x = pos[0] // gap
            y = pos[1] // gap
            return int(y), int(x)
        else:
            return None

    def is_finished(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.cubes[i][j].value == 0:
                    return False
        return True

    def update_result(self):
        self.cubes = [[Cube(self.values[i][j], i, j, self.width, self.height) for j in range(self.cols)] for i in
                      range(self.rows)]
        self.update_model()


class Cube:
    rows = 9
    cols = 9

    def __init__(self, value, row, col, width, height):
        self.value = value
        self.temp = 0
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.selected = False

    def draw(self, win):
        font = pygame.font.SysFont("comicsans", 40)

        gap = self.width / 9
        x = self.col * gap
        y = self.row * gap

        if self.value != 0:
            text = font.render(str(self.value), True, BLACK)
            win.blit(text, (x + (gap / 2 - text.get_width() / 2), y + (gap / 2 - text.get_height() / 2)))

        if self.selected:
            pygame.draw.rect(win, RED, (x, y, gap, gap), 3)

    def set(self, val):
        self.value = val


def win_draw(board, win, word):
    win.fill(WHITE)
    board.draw(win)
    font = pygame.font.SysFont('consolas', 30)
    text = font.render(word, True, BLUE, RED)
    win.blit(text, (100, 555))


def main():
    win = pygame.display.set_mode((540, 600))
    pygame.display.set_caption("Sudoku")
    board = Grid(9, 9, 540, 540)
    key = None
    run = True
    text = 'Press SPACE to solve'
    while run:
        win_draw(board, win, text)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                clicked = board.click(pos)
                if clicked:
                    board.select(clicked[0], clicked[1])
                    key = None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    key = 1
                if event.key == pygame.K_2:
                    key = 2
                if event.key == pygame.K_3:
                    key = 3
                if event.key == pygame.K_4:
                    key = 4
                if event.key == pygame.K_5:
                    key = 5
                if event.key == pygame.K_6:
                    key = 6
                if event.key == pygame.K_7:
                    key = 7
                if event.key == pygame.K_8:
                    key = 8
                if event.key == pygame.K_9:
                    key = 9
                if event.key == pygame.K_DELETE:
                    board.clear()
                    key = None
                if event.key == pygame.K_SPACE:
                    board.values = solve(board.values)
                    # print(board.values)
                    board.update_result()
                    text = 'PROBLEM SOLVED'
                    win_draw(board, win, text)

        if board.selected and key!=None:
            board.place(key)
        pygame.display.update()


# Solving
def is_row_duplicate(board, pos, num):
    # Check row
    for i in range(0, 9):
        if board[pos[0]][i] == num:
            return True
    return False


def is_col_duplicate(board, pos, num):
    # Check column
    for i in range(0, 9):
        if board[i][pos[1]] == num:
            return True
    return False


def is_block_duplicate(board, pos, num):
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num:
                return True
    return False


def sort_fitness(c):
    return c.fitness


def valid(board, num, pos):

    if (is_row_duplicate(board, pos, num) 
        or is_col_duplicate(board, pos, num) 
        or is_block_duplicate(board, pos, num)):
        return False
    return True


class Population:
    """ 
    A set of candidate solutions to the Sudoku puzzle. 
    These candidates are also known as the chromosomes in the population.
    """

    def __init__(self):
        self.candidates = []
        # self.helper = Candidate()
        return

    """def auto_fill(self, given):
        list_appear_in_row = [[[] for i in range (0, Nd)] for j in range (0, Nd)]

        for row in range(0, Nd):
            for col in range(0, Nd):
                for value in range(1, Nd + 1):
                    if((given[row][col] == 0) and valid(given, value, (row, col))):
                        list_appear_in_row[row][value - 1].append(col)
                    elif(given[row][col] != 0):
                        break
       
        update = True
        while update:
            update = False
            for row in range(0, Nd):    
                for value in range(1, Nd + 1):
                    if len(list_appear_in_row[row][value - 1]) == 1:
                        col = list_appear_in_row[row][value - 1][0]

                        for _row in range(0, Nd):
                            if col in list_appear_in_row[_row][value - 1] and _row != row:
                                list_appear_in_row[_row][value - 1].remove(col)

                        row_in_block = [i for i in range(row // 3 * 3, row // 3 * 3 + 3)]
                        row_in_block.remove(row)
                        col_in_block = [j for j in range(col // 3 * 3, col // 3 * 3 + 3)]
                        col_in_block.remove(col)
                        for _row in row_in_block:
                            for _col in col_in_block:
                                if _col in list_appear_in_row[_row][value - 1]:
                                    list_appear_in_row[_row][value - 1].remove(_col)

                        for value_delete_appear in range(1, Nd + 1):
                            if col in list_appear_in_row[row][value_delete_appear - 1]:
                                list_appear_in_row[row][value_delete_appear - 1].remove(col)

                        update = True
                        given[row][col] = value
                        print("Update row:", row, col, value)
    """

    def seed(self, Nc, given):
       # Determine the legal values that each square can take.
       helper = Candidate()
       helper.values = [[[] for j in range(0, 9)] for i in range(0, 9)]
       seeding = True
       while seeding:
           for row in range(0, 9):
               for col in range(0, 9):
                   for value in range(1, 10):
                       if given[row][col] == 0 and valid(given, value, (row, col)):
                           # Value is available.
                           helper.values[row][col].append(value)
                       elif given[row][col] != 0:
                           # Given/known value from file.
                           helper.values[row][col].append(given[row][col])
                           break
           for p in range(0, Nc):
               g = Candidate()
               for i in range(0, 9): # New row in candidate.
                   row = [0 for i in range(9)]
                      
                   # Fill in the givens.
                   for j in range(0, 9): # New column j value in row i.
                      
                       # If value is already given, don't change it.
                       if(given[i][j] != 0):
                           row[j] = given[i][j]
                           # Fill in the gaps using the helper board.
                       elif(given[i][j] == 0):
                           row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]
  
                   # If we don't have a valid board, then try again. There must be no duplicates in the row.
                   while(len(list(set(row))) != 9):
                       for j in range(0, 9):
                           if(given[i][j] == 0):
                               row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]
                  
                   g.values[i] = row
                  
               self.candidates.append(g)          
               # Compute the fitness of all candidates in the population.
           self.update_fitness()
           seeding=False
      
       print("Seeding complete.")
       return

        
    def update_fitness(self):
        #Update fitness of every candidate/chromosome
        for candidate in self.candidates:
            candidate.update_fitness()
        return
    def sort(self):
        self.candidates = sorted(self.candidates, key= sort_fitness, reverse= True)


class Candidate(object):
    """ A candidate solutions to the Sudoku puzzle. """
    def __init__(self):
        self.values = [([0]*9) for i in range(9)]
        self.fitness = 0.0
        return

    def complete(self):
        if self.fitness == 1.0:
            return True

    def update_fitness(self):
        """ 
        The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. 
        The actual solution (i.e. the 'fittest') is defined as a 9x9 grid of numbers in the range [1, 9] 
        where each row, column and 3x3 block contains the numbers [1, 9] without any duplicates 
        if there are any duplicates then the fitness will be lower. 
        """
        
        col_count = [0 for i in range(9)]
        block_count = [0 for i in range(9)]
        col_sum = 0
        block_sum = 0
        for i in range(0, 9):  # For each col...
            for j in range(0, 9):  # For each number within it...
                col_count[self.values[j][i]-1] += 1  # ...Update list with occurrence of a particular number.
            col_sum += (1.0 / len(set(col_count)))/9
            col_count = [0 for i in range(9)]

        # For each block...
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                block_count[self.values[i][j]-1] += 1
                block_count[self.values[i][j+1]-1] += 1
                block_count[self.values[i][j+2]-1] += 1
                
                block_count[self.values[i+1][j]-1] += 1
                block_count[self.values[i+1][j+1]-1] += 1
                block_count[self.values[i+1][j+2]-1] += 1
                
                block_count[self.values[i+2][j]-1] += 1
                block_count[self.values[i+2][j+1]-1] += 1
                block_count[self.values[i+2][j+2]-1] += 1

                block_sum += (1.0/len(set(block_count)))/9
                block_count = [0 for i in range(9)]

        # Calculate overall fitness.
        if int(col_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = col_sum * block_sum
        
        self.fitness = fitness
        return
        
    def mutate(self, mutation_rate, given):
        """ Mutate a candidate by picking a row, and then picking two values within that row to swap. """

        r = random.uniform(0,1.0000000000001)
    
        success = False
        if r < mutation_rate:  # Mutate.
            while not success:
                row1 = random.randint(0, 8)
                row2 = row1
                from_col = random.randint(0, 8)
                to_col = random.randint(0, 8)
                while from_col == to_col:
                    from_col = random.randint(0, 8)
                    to_col = random.randint(0, 8)   

                # Check if the two places are free...
                if given[row1][from_col] == 0 and given[row1][to_col] == 0:
                    # ...and that we are not causing a duplicate in the rows' cols.
                    if(not     is_col_duplicate(given, (row2,to_col),  self.values[row1][from_col])
                       and not is_col_duplicate(given, (row1,from_col), self.values[row2][to_col])
                       and not is_block_duplicate(given, (row2,to_col), self.values[row1][from_col])
                       and not is_block_duplicate(given, (row1,from_col), self.values[row2][to_col])):
                    
                        # Swap values.
                        temp = self.values[row2][to_col]
                        self.values[row2][to_col] = self.values[row1][from_col]
                        self.values[row1][from_col] = temp
                        success = True
    
        return success


def compete(candidates):
        """ Pick 2 random candidates from the population and get them to compete against each other. """
        c1 = candidates[random.randint(0, len(candidates)-1)]
        c2 = candidates[random.randint(0, len(candidates)-1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Find the fittest and the weakest.
        if f1 > f2:
            fittest = c1
            weakest = c2
        else:
            fittest = c2
            weakest = c1

        selection_rate = 0.85
        r = random.uniform(0, 1.00000000000001)
        if r < selection_rate:
            return fittest
        else:
            return weakest


class CycleCrossover(object):
    """ Crossover relates to the analogy of genes within each parent candidate mixing together in the hopes of creating a fitter child candidate. 
    Cycle crossover is used here (see e.g. A. E. Eiben, J. E. Smith. Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self):
        return
    
    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate()
        child2 = Candidate()
        
        # Make a copy of the parent genes.
        child1.values = list(parent1.values)
        child2.values = list(parent2.values)

        r = random.uniform(0, 1.1)
        while r > 1:  # Outside [0, 1] boundary. Choose another.
            r = random.uniform(0, 1.1)
            
        # Perform crossover.
        if r < crossover_rate:
            # Pick a crossover point. Crossover must have at least 1 row (and at most 9-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while crossover_point1 == crossover_point2:
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)
                
            if crossover_point1 > crossover_point2:
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
                
            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2): 
        child_row1 = [0 for i in range(9)]
        child_row2 = [0 for i in range(9)]

        remaining = [i for i in range(1,9+1)]
        cycle = 0
        
        while((0 in child_row1) and (0 in child_row2)):  # While child rows not complete...
            if(cycle % 2 == 0):  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]
                
                while(next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]
                
                while(next != start):  # While cycle not done...
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]
                    
                cycle += 1
        return child_row1, child_row2  
    
    # return index
    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if parent_row[i] in remaining:
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if parent_row[i] == value:
                return i

def solve(board):
    # num_round = 100
    tol_gen = 0
    solution = None
    # for run_time in range(0, num_round):
        # w_file.write("Round "+str(run_time)+"\n")
    Nc = 1000  # Number of candidates (i.e. population size).
    Ne = int(0.05*Nc)  # Number of elites.
    Ng = 10000  # Number of generations.
    Nm = 0  # Number of mutations.
    # Mutation parameters.
    phi = 0
    sigma = 1
    mutation_rate = 0.06
    mutation_rate_elite = 0.5
    # Create an initial population.
    population = Population()
    population.seed(Nc, board)
    best_fitness = 0
    count_best_fit_unchange = 0
    count_bound = 0
    # For up to Ng generations...
    for generation in range(0, Ng):
        print("Generation %d" % generation)
        # Check for a solution.
        population.update_fitness()
        population.sort()
        if best_fitness == population.candidates[0].fitness:
            count_best_fit_unchange += 1
        else:
            count_best_fit_unchange = 0
            if best_fitness < population.candidates[0].fitness:
                best_fitness = population.candidates[0].fitness
        print("Fitness: ", best_fitness)
        # w_file.write("Generation "+str(generation)+": "+str(best_fitness)+"\n")
        # Maybe the generation stuck on a local maximum, so let's shock them
        if count_best_fit_unchange >= 1000:
            for e in range(0, Ne):
                elite = population.candidates[e]
                old_fitness = elite.fitness
                success = elite.mutate(mutation_rate_elite, board)
                if success:
                    elite.update_fitness()
                    Nm += 1
                    if elite.fitness > old_fitness:  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1
            count_best_fit_unchange = 0

        if best_fitness == 1:
            # print("Solution found at generation %d!" % generation)
            tol_gen += generation
            solution = population.candidates[0].values
            break
        # Create the next population.
        next_population = []

        # Select elites (the fittest candidates) and preserve them for the next generation.
        for e in range(0, Ne):
            elite = population.candidates[e]
            next_population.append(elite)

        # Create the rest of the candidates.
        new_turn = False
        for count in range(Ne, Nc, 2):
            # Select parents from population via a tournament.
            parent1 = compete(population.candidates)
            parent2 = compete(population.candidates)

            # Cross-over.
            cc = CycleCrossover()
            child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

            # Mutate child1.
            child1.update_fitness()
            old_fitness = child1.fitness
            if old_fitness == 1.0:
                tol_gen += generation
                new_turn = True
                solution = child1.values
                break
                # print("solution found at: ", generation)
                # return child1.values
            success = child1.mutate(mutation_rate, board)
            child1.update_fitness()
            if success:
                Nm += 1
                if child1.fitness > old_fitness:  # Used to calculate the relative success rate of mutations.
                    phi = phi + 1

            # Mutate child2.
            child2.update_fitness()
            old_fitness = child2.fitness
            if old_fitness == 1.0:
                tol_gen += generation
                new_turn = True
                solution = child2.values
                break
                # print("solution found at: ", generation)
                # return child2.values
            success = child2.mutate(mutation_rate, board)
            child2.update_fitness()
            if success:
                Nm += 1
                if(child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                    phi = phi + 1

            # Add children to new population.
            next_population.append(child1)
            next_population.append(child2)

        if new_turn == True:
            break
        # Select next generation.
        population.candidates = next_population

        """
        Calculate new adaptive mutation rate (based on Rechenberg's 1/5 success rule). 
        This is to stop too much mutation as the fitness progresses towards unity.
        """
        if Nm == 0:
            phi = 0  # Avoid divide by zero.
        else:
            phi = phi / Nm

        if phi > 0.2:
            sigma = sigma/0.998
        elif phi < 0.2:
            sigma = sigma*0.998

        mutation_rate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
        Nm = 0
        phi = 0

        # Re-seed the population if 2000 generations have passed and the solution is'n found
        up_bound = 2000
        if count_bound >= up_bound:
            print("The population has gone stale. Re-seeding...")
            population.seed(Nc, board)
            sigma = 1
            phi = 0
            Nm = 0
            mutation_rate = 0.06
            count_bound = -1
            # run_time += 1
        count_bound += 1
    if solution == None:
        print('No found solution')
        pygame.quit()
        sys.exit()
    print("The average generation is: ", tol_gen/100)
    return solution

main()
pygame.quit()
sys.exit()
