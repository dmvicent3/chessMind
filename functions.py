import numpy as np
import cv2
import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
from functools import partial
from base64 import b64encode
import io
from PIL import Image
from keras.models import load_model
import chess
import chess.engine

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

def findBoard(fname): 
    img = cv2.imdecode(fname, 1)

    if img is None:
        print('No image found')
        return None
    #print(img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    
    # Canny edge detection
    edges = autoCanny(gray)
    #print(np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]))
    if np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) > 0.35: #0.015 by default
        print ('too many edges')
        return None
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is None:
        print ('no lines')
        return None
    
    lines = np.reshape(lines, (-1, 2))
    
    # Compute intersection points
    h, v = horVertLines(lines)
    if len(h) < 9 or len(v) < 9:
        print ('too few lines')
        return None
    points = intersections(h, v)
    
    # Cluster intersection points
    points = cluster(points)

    # Find corners
    img_shape = np.shape(img)
    points = findCorners(points, (img_shape[1], img_shape[0]))
    
    # Perspective transform
    new_img = fourPointTransform(img, points)

    return new_img



def autoCanny(image, sigma=0.33):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def horVertLines(lines):
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v

def intersections(h, v):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)

def cluster(points, max_dist=50):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = list(map(lambda arr: (np.mean(np.array(arr)[:,0]), np.mean(np.array(arr)[:,1])), clusters))
    return clusters

def closestPoint(points, loc):
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(list(map(partial(spatial.distance.euclidean, loc), points)))
    return points[dists.argmin()]

def findCorners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closestPoint(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closestPoint(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))
    
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25 # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closestPoint(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closestPoint(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner), np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners

def fourPointTransform(img, points, squareLength=1816):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, squareLength], [squareLength, squareLength], [squareLength, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (squareLength, squareLength))

def segmentBoard(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sqrLen = img.shape[0] / 8
    for i in range(8):
        for j in range(8):
            arr.append(img[i * int(sqrLen) : (i + 1) * int(sqrLen), j * int(sqrLen) : (j + 1) * int(sqrLen)])
    return arr

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def uint8ToBase64(image):
    rawBytes = io.BytesIO()
    image = Image.fromarray(image.astype("uint8"))
    image.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    return "data:image/jpeg;base64,"+b64encode(rawBytes.getvalue()).decode('ascii')


def prepare_image(image, size=(224,224)):
   
    #image = cv2.resize(image, size)
    #image = np.reshape(image, (None, 224, 224, 3))
    image = Image.fromarray(image).resize(size)
    
    #image = np.flip(image, axis=2)
    #image = np.expand_dims(image, axis=-1)
    image = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    image = np.flip(image, axis=2)
    image = np.expand_dims(image, axis=0)
    image[:, :, 0] -= 103.939
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 123.68
    return image

def retFEN(pred_list):
    fen = ''.join(pred_list)
    fen = fen[::-1]
    fen = '/'.join(fen[i:i + 8] for i in range(0, len(fen), 8))
    sum_digits = 0
    for i, p in enumerate(fen):
        if p.isdigit():
            sum_digits += 1
        elif p.isdigit() is False and (fen[i - 1].isdigit() or i == len(fen)):
            fen = fen[:(i - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1)) + fen[i:]
            sum_digits = 0
    if sum_digits > 1:
        fen = fen[:(len(fen) - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1))
    fen = fen.replace('D', '')
    print(fen)
    return fen

def fen_to_board(fen):
    board = []
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend( ['--'] * int(c) )
            elif c == 'p':
                brow.append( 'bp' )
            elif c == 'P':
                brow.append( 'wp' )
            elif c > 'Z':
                brow.append( 'b'+c.upper() )
            else:
                brow.append( 'w'+c )

        board.append( brow )
    return board

model = load_model('C:/Users/Daniel/Desktop/VSCode/chessMind/model_digital.h5')

def predictImages(imgs):
    category_reference = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 6: '1', 7: 'B', 8: 'K', 9: 'N', 10: 'P',
        11: 'Q', 12: 'R'}
    pred_list = []

    for img in imgs:
        preparedImg = prepare_image(img)
        out = model.predict(preparedImg)
        top_pred = np.argmax(out)
        pred = category_reference[top_pred]
        pred_list.append(pred)
    return pred_list

def calculateBestMove(fen, lastSideToPlay):

    # Initialize the chess board using a FEN string
    fen = fen + ' ' + lastSideToPlay + ' KQkq - 0 1'
    board = chess.Board(fen)

    # Create and configure the chess engine
    engine = chess.engine.SimpleEngine.popen_uci("C:/Users/Daniel/Desktop/VSCode/chessMind/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")
    engine.configure({"Threads": 1})

    # Search for the best move
    info = engine.analyse(board, chess.engine.Limit(time=0.5))
    bestMove = info["pv"][0]
    print(bestMove)
    bestMoveStr = bestMove.uci()
    pieceToMove = bestMoveStr[:2]
    postion = bestMoveStr[2:]

    result = "Move " + pieceToMove + " to " + postion
    # Close the engine
    engine.quit()
    
    return result
    