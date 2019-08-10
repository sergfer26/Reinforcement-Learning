import numpy as np

class Board: 

    def __init__(self):
        self.items = ['']*9
        self.canon_items = 
        self.syms = {'': 0, 'O': 1, 'X': 2}

    def show_board(self):
        board = """
             {} | {} | {}
            ----------
             {} | {} | {}
            ----------
             {} | {} | {}
        """.format(*self.items)
        print(board)

    def mark_(self, space, mark='X'):
        if self.items[space] == '':
            self.items[space] = mark
        else:
            print('espacio ocupado')

    def items_to_state(self):
        '''
            pasa los elementos de tablero a un estado
        '''
        state = [self.syms.get(sym) for sym in self.items]
        return state

    def state_to_items(self, state):
        '''
            recibe un estado y actualiza los elemetos del tablero
        '''
        self.items = [self.syms[key] for key in state]

    




        



