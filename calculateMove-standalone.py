import chess
import chess.engine

sideToPlay = 'w' #'b'

# Initialize the chess board using a FEN string
#fen = 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1'
#fen = 'rn1qkb1r/ppp1pppp/5n2/3p1b2/3P1B2/5N2/PPP1PPPP/RN1QKB1R '+sideToPlay+' KQkq - 0 1'
fen = 'r3kb1r/ppp2ppp/2n1pn2/3p1bq1/1P1P1B2/2N2N2/P1P1PPPP/R2QKB1R ' + sideToPlay + ' KQkq - 0 1'
board = chess.Board(fen)

# Create and configure the chess engine
engine = chess.engine.SimpleEngine.popen_uci("C:/Users/danie/OneDrive/Ambiente de Trabalho/VSCode/AI/chessMind/chessMind/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")
engine.configure({"Threads": 1})

# Search for the best move
info = engine.analyse(board, chess.engine.Limit(time=0.1))
bestMove = info["pv"][0]

bestMoveStr = bestMove.uci()
pieceToMove = bestMoveStr[:2]
postion = bestMoveStr[2:]

result = "Move " + pieceToMove + " to " + postion 

# Print the best move
print(result)
# Close the engine
engine.quit()